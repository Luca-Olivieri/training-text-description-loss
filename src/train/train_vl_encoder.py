from config import *
from data import JSONLDataset, ImageDataset, ImageCaptionDataset, crop_image_preprocess_image_text_batch
from models.vl_encoders import VLE_REGISTRY, VLEncoder
from viz import get_layer_numel_str
from utils import compile_torch_model, get_torch_gen # Assuming get_torch_gen is in utils
from logger import LogManager

import torch
from torch import nn
from functools import partial
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, ConcatDataset
import math
from open_clip_train.precision import get_autocast # for AMP
from torch.amp import GradScaler # for AMP
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown

from typing import Callable, Optional
import dataclasses
from pathlib import Path

from vendors.flair.src.flair.train import backward

VLE_CONFIG = CONFIG['vle']
VLE_TRAIN_CONFIG = VLE_CONFIG['train']

if VLE_TRAIN_CONFIG['log_only_to_stdout']:
    log_manager = LogManager(
        exp_name=VLE_TRAIN_CONFIG['exp_name'],
        exp_desc=VLE_TRAIN_CONFIG['exp_desc'],
    )
else:
    log_manager = LogManager(
        exp_name=VLE_TRAIN_CONFIG['exp_name'],
        exp_desc=VLE_TRAIN_CONFIG['exp_desc'],
        file_logs_dir_path=VLE_TRAIN_CONFIG['file_logs_dir_path'],
        tb_logs_dir_path=VLE_TRAIN_CONFIG['tb_logs_dir_path']
    )

def train_loop(
        vle: VLEncoder,
        train_dl: DataLoader,
        val_dl: DataLoader,
        criterion: Callable,
        checkpoint_dict: Optional[dict] = None
) -> None:
    
    # --- 1. Initialization and State Restoration ---
    
    start_epoch = 0
    global_step = 0
    if checkpoint_dict:
        start_epoch = checkpoint_dict['epoch'] + 1
        global_step = checkpoint_dict['global_step']
        log_manager.log_line(f"Resuming training from epoch {start_epoch}, global step {global_step}.")

    grad_accum_steps = VLE_TRAIN_CONFIG['grad_accum_steps']
    num_batches_per_epoch = len(train_dl)
    # The number of optimizer steps per epoch
    num_steps_per_epoch = math.ceil(num_batches_per_epoch / grad_accum_steps)
    
    # --- 2. Optimizer Setup ---
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(vle.model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": 0.5},
        ],
        lr=VLE_TRAIN_CONFIG['lr_schedule']['base_lr'],
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    if checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

    # --- 3. Scheduler Setup ---
    total_steps = num_steps_per_epoch * VLE_TRAIN_CONFIG['num_epochs']
    sched_config = VLE_TRAIN_CONFIG['lr_schedule']
    
    scheduler = None
    if sched_config['policy'] == 'const':
        scheduler = const_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)
    elif sched_config['policy'] == 'const-cooldown':
        cooldown_steps = num_steps_per_epoch * sched_config['epochs_cooldown']
        scheduler = const_lr_cooldown(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps, cooldown_steps, sched_config['lr_cooldown_power'], sched_config['lr_cooldown_end'])
    elif sched_config['policy'] == 'cosine':
        scheduler = cosine_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)
    
    # --- 4. AMP and Model Compilation Setup ---
    autocast = get_autocast(VLE_TRAIN_CONFIG['precision'])
    scaler = GradScaler() if VLE_TRAIN_CONFIG['precision'] == "amp" else None
    if scaler and checkpoint_dict:
        scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])
    
    vle.model = compile_torch_model(vle.model)

    # --- 5. Initial Validation ---
    log_manager.log_title("Initial Validation")
    val_loss = vle.evaluate(val_dl, criterion)
    log_manager.log_scores(f"Before any weight update, VALIDATION", val_loss, None, start_epoch, "val", None, "val_")
    best_val_loss = val_loss

    log_manager.log_title("Training Start")
    
    # --- 6. Main Training Loop ---
    for epoch in range(start_epoch, VLE_TRAIN_CONFIG["num_epochs"]):
        
        vle.model.train()
        accum_features = {}
        if grad_accum_steps > 1:
            accum_images, accum_texts = [], []
        
        for step, (images, texts) in enumerate(train_dl):
            
            # --- Forward Pass and Loss Calculation ---
            if grad_accum_steps == 1:
                optimizer.zero_grad()
                with autocast():
                    vle_output = vle.encode_and_project(images, texts, broadcast=False)
                    losses = criterion(
                        image_features=vle_output.global_image_token,
                        image_tokens=vle_output.local_image_tokens.clone(),
                        text_features=vle_output.global_text_token.squeeze(1),
                        logit_scale=vle.model.logit_scale.exp(),
                        logit_bias=vle.model.logit_bias,
                        visual_proj=vle.model.visual_proj,
                        output_dict=True
                    )
                    total_loss = sum(losses.values())
                
                backward(total_loss, scaler)
            
            else: # --- Gradient Accumulation Logic ---
                # Cache features without tracking gradients
                with torch.no_grad(), autocast():
                    vle_output = vle.encode_and_project(images, texts, broadcast=False)
                    vle_output_dict = {f.name: getattr(vle_output, f.name) for f in dataclasses.fields(vle_output) if f.name in ['global_image_token', 'global_text_token', 'local_image_tokens']}
                    for key, val in vle_output_dict.items():
                        accum_features.setdefault(key, []).append(val)
                    accum_images.append(images)
                    accum_texts.append(texts)

                # Determine if it's time to perform an optimizer step
                is_last_batch = (step + 1) == num_batches_per_epoch
                is_accum_step = (step + 1) % grad_accum_steps == 0
                
                if not is_accum_step and not is_last_batch:
                    continue

                # --- Perform gradient step on accumulated batches ---
                optimizer.zero_grad()
                # Re-run forward pass with gradient tracking for each batch in the accumulation window
                num_accum = len(accum_images)
                for i in range(num_accum):
                    images_i, texts_i = accum_images[i], accum_texts[i]
                    with autocast():
                        vle_output_i = vle.encode_and_project(images_i, texts_i, broadcast=False)
                        vle_output_dict_i = {f.name: getattr(vle_output_i, f.name) for f in dataclasses.fields(vle_output_i) if f.name in ['global_image_token', 'global_text_token', 'local_image_tokens']}

                        # Gather features from all accumulated batches for negatives
                        inputs = {}
                        for key in accum_features:
                            accumulated = accum_features[key]
                            inputs[key] = torch.cat(accumulated[:i] + [vle_output_dict_i[key]] + accumulated[i + 1:])

                        losses = criterion(
                            image_features=inputs['global_image_token'],
                            image_tokens=inputs['local_image_tokens'].clone(),
                            text_features=inputs['global_text_token'].squeeze(1),
                            logit_scale=vle.model.logit_scale.exp(),
                            logit_bias=vle.model.logit_bias,
                            visual_proj=vle.model.visual_proj,
                            output_dict=True
                        )
                        del inputs
                        total_loss = sum(losses.values())
                        
                    # Accumulate gradients. Loss is scaled by num_accum to average gradients.
                    backward(total_loss / num_accum, scaler)

                # Reset accumulators
                accum_images, accum_texts, accum_features = [], [], {}
            
            # --- Optimizer Step and Scheduler Update ---
            if scheduler:
                scheduler(global_step)
            
            grad_norm = 0.0
            if scaler:
                if VLE_TRAIN_CONFIG['grad_clip_norm']:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(vle.model.parameters(), VLE_TRAIN_CONFIG['grad_clip_norm'], norm_type=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                if VLE_TRAIN_CONFIG['grad_clip_norm']:
                    grad_norm = torch.nn.utils.clip_grad_norm_(vle.model.parameters(), VLE_TRAIN_CONFIG['grad_clip_norm'], norm_type=2.0)
                optimizer.step()

            global_step += 1 # Increment global step *only* after an optimizer step

            if not VLE_TRAIN_CONFIG['grad_clip_norm']:
                grad_norm = torch.nn.utils.clip_grad_norm_(vle.model.parameters(), float('inf'), norm_type=2.0)

            with torch.no_grad():
                vle.model.logit_scale.clamp_(0, math.log(100))
            
            # --- Logging ---
            if global_step % VLE_TRAIN_CONFIG['log_every'] == 0:
                current_lr = optimizer.param_groups[0]['lr']
                step_in_epoch = (step // grad_accum_steps) + 1
                log_manager.log_scores(
                    f"epoch: {epoch+1}/{VLE_TRAIN_CONFIG['num_epochs']}, step: {step_in_epoch}/{num_steps_per_epoch} (global_step: {global_step})",
                    total_loss, None, global_step, "train",
                    f", lr: {current_lr:.2e}, grad_norm: {grad_norm:.2f}", "batch_"
                )

            torch.cuda.synchronize() if CONFIG["device"] == "cuda" else None

        # --- End of Epoch Validation and Checkpointing ---
        val_loss = vle.evaluate(val_dl, criterion)
        log_manager.log_scores(f"epoch: {epoch+1}/{VLE_TRAIN_CONFIG['num_epochs']}, VALIDATION", val_loss, None, epoch+1, "val", None, "val_")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            if VLE_TRAIN_CONFIG['save_weights_root_path']:
                # Note: 'epoch' is saved, so on resume we start from 'epoch + 1'
                new_checkpoint_dict = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': vle.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if scaler:
                    new_checkpoint_dict['scaler_state_dict'] = scaler.state_dict()

                save_dir = Path(VLE_TRAIN_CONFIG['save_weights_root_path'])
                save_dir.mkdir(parents=True, exist_ok=True)
                ckp_filename = f"flair-{vle.version}-{VLE_TRAIN_CONFIG['exp_name']}.pth"
                full_ckp_path = save_dir / ckp_filename
                torch.save(new_checkpoint_dict, full_ckp_path)
                log_manager.log_line(f"New best model saved to {full_ckp_path} with validation loss: {best_val_loss:.4f}")

    log_manager.log_title("Training Finished")

def main() -> None:

    mask_color: str = VLE_TRAIN_CONFIG['mask_color']

    train_image_text_ds = ConcatDataset([
        ImageCaptionDataset(
            ImageDataset(Path(f'/home/olivieri/exp/data/data_gen/VOC2012/train/images_{mask_color}')),
            JSONLDataset(Path('/home/olivieri/exp/data/data_gen/VOC2012/train/captions.jsonl')),
        ),
        ImageCaptionDataset(
            ImageDataset(Path(f'/home/olivieri/exp/data/data_gen/COCO2017/train/images_{mask_color}')),
            JSONLDataset(Path('/home/olivieri/exp/data/data_gen/COCO2017/train/captions.jsonl'))
        )
    ])

    val_image_text_ds = ConcatDataset([
        ImageCaptionDataset(
            ImageDataset(Path(f'/home/olivieri/exp/data/data_gen/VOC2012/val/images_{mask_color}')),
            JSONLDataset(Path('/home/olivieri/exp/data/data_gen/VOC2012/val/captions.jsonl'))
        ),
        ImageCaptionDataset(
            ImageDataset(Path(f'/home/olivieri/exp/data/data_gen/COCO2017/val/images_{mask_color}')),
            JSONLDataset(Path('/home/olivieri/exp/data/data_gen/COCO2017/val/captions.jsonl'))
        )
    ])

    # Vision-Language Encoder
    vle: VLEncoder = VLE_REGISTRY.get("flair", version='flair-cc3m-recap.pt', device=CONFIG['device'], vision_adapter=False, text_adapter=True)

    checkpoint_dict = None
    if VLE_TRAIN_CONFIG['resume_path']:
        resume_path = Path(VLE_TRAIN_CONFIG['resume_path'])
        if resume_path.exists():
            checkpoint_dict = torch.load(resume_path, map_location=CONFIG['device'])
            vle.model.load_state_dict(checkpoint_dict['model_state_dict'])
        else:
            raise AttributeError(f"ERROR: Resume path '{resume_path}' not found. ")
    
    # vle.set_vision_trainable_params(['proj', 'visual_proj', 'vision_adapter'])
    vle.set_vision_trainable_params(['text_adapter'])

    # DataLoaders
    train_collate_fn = partial(
        crop_image_preprocess_image_text_batch,
        crop_fn=T.CenterCrop(VLE_CONFIG['image_size']),
        preprocess_images_fn=vle.preprocess_images,
        preprocess_texts_fn=vle.preprocess_texts
    )
    val_collate_fn = partial(
        crop_image_preprocess_image_text_batch,
        crop_fn=T.CenterCrop(VLE_CONFIG['image_size']),
        preprocess_images_fn=vle.preprocess_images,
        preprocess_texts_fn=vle.preprocess_texts
    )

    train_image_text_dl = DataLoader(
        train_image_text_ds,
        batch_size=VLE_TRAIN_CONFIG["batch_size"],
        shuffle=True,
        generator=get_torch_gen(),
        collate_fn=train_collate_fn,
    )
    
    val_image_text_dl = DataLoader(
        val_image_text_ds,
        batch_size=VLE_TRAIN_CONFIG["batch_size"],
        shuffle=False,
        generator=get_torch_gen(),
        collate_fn=val_collate_fn,
    )

    criterion = vle.create_loss(
        add_mps_loss=True,
        num_caps_per_img=1
    )

    log_manager.log_intro(
        config=CONFIG,
        train_ds=train_image_text_ds,
        val_ds=val_image_text_ds,
        train_dl=train_image_text_dl,
        val_dl=val_image_text_dl
    )
    
    log_manager.log_title("Trainable Params")
    [log_manager.log_line(t) for t in get_layer_numel_str(vle.model, print_only_total=False, only_trainable=True).split('\n')]

    try:
        train_loop(
            vle,
            train_image_text_dl,
            val_image_text_dl,
            criterion,
            checkpoint_dict
        )
    except KeyboardInterrupt:
        log_manager.log_title("Training Interrupted", pad_symbol='~')

    log_manager.close_loggers()

if __name__ == '__main__':
    main()
