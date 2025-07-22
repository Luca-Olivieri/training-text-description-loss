from config import *
from data import JSONLDataset, ImageDataset, ImageCaptionDataset, crop_image_preprocess_image_text_batch
from models.vl_encoders import VLE_REGISTRY, VLEncoder
from viz import get_layer_numel_str
from utils import compile_torch_model
from logger import LogManager

from torch import nn
from functools import partial
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, ConcatDataset
import math
from open_clip_train.precision import get_autocast # for AMP
from torch.amp import GradScaler # for AMP
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown

from typing import Callable, Optional
import dataclasses

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
) -> torch.Tensor:
    
    # optimizer
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(vle.model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    # the following optimizer settings are take straight from FLAIR authors.
    optimizer = torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": 0.5}, # authors use 0.5, seems quite high
        ],
        lr=VLE_TRAIN_CONFIG['lr_schedule']['base_lr'],
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict']) if VLE_TRAIN_CONFIG['resume_path'] else None
    optimizer.zero_grad()

    num_batches = len(train_dl) // VLE_TRAIN_CONFIG['grad_accum_steps']

    # scheduler
    total_steps = num_batches * VLE_TRAIN_CONFIG['num_epochs']
    sched_config = VLE_TRAIN_CONFIG['lr_schedule']
    match sched_config['policy']:
        case None:
            scheduler = None
        case 'const':
            scheduler = const_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)
        case 'const-cooldown':
            cooldown_steps = num_batches * sched_config['epochs_cooldown']
            scheduler = const_lr_cooldown(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps, cooldown_steps, sched_config['lr_cooldown_power'], sched_config['lr_cooldown_end'])
        case 'cosine':
            scheduler = cosine_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)
    
    # AMP scaler
    scaler = GradScaler() if VLE_TRAIN_CONFIG['precision'] == "amp" else None
    scaler.load_state_dict(checkpoint_dict["scaler_state_dict"]) if (scaler and VLE_TRAIN_CONFIG['resume_path']) else None

    vle.model = compile_torch_model(vle.model) # effective only with GPUs with compute capability >= 7.0

    log_manager.log_title("Initial Validation")
    val_loss = vle.evaluate(val_dl, criterion)
    log_manager.log_scores(f"Before any weight update, VALIDATION", val_loss, None, 0, "val", None, "val_")
    best_val_loss = val_loss

    log_manager.log_title("Training Start")

    autocast = get_autocast(VLE_TRAIN_CONFIG['precision'])

    if VLE_TRAIN_CONFIG['grad_accum_steps'] > 1:
        accum_images, accum_texts, accum_features = [], [], {}
    
    for epoch in range(VLE_TRAIN_CONFIG["num_epochs"]):

        global_epoch = checkpoint_dict['global_epoch'] + epoch if VLE_TRAIN_CONFIG['resume_path'] else epoch

        for step, (images, texts) in enumerate(train_dl):

            step_accum = step // VLE_TRAIN_CONFIG['grad_accum_steps']
            step_counter = epoch*num_batches + step_accum
            global_step_counter = checkpoint_dict['global_step_counter'] + step_counter if VLE_TRAIN_CONFIG['resume_path'] else step_counter

            vle.model.train()

            scheduler(global_step_counter) if scheduler else None
            
            optimizer.zero_grad()

            if VLE_TRAIN_CONFIG['grad_accum_steps'] == 1:
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
                    total_loss = sum(losses.values()) # in our case, we only have losses has only the 'constrastive loss' key.

                backward(total_loss, scaler)
            else:
                # First, cache the features without any gradient tracking.
                with torch.no_grad():
                    with autocast():
                        vle_output = vle.encode_and_project(images, texts, broadcast=False)
                        vle_output_dict = {f.name: getattr(vle_output, f.name) for f in dataclasses.fields(vle_output) if f.name in ['global_image_token', 'global_text_token', 'local_image_tokens']}

                        for key, val in vle_output_dict.items():
                            if key in accum_features:
                                accum_features[key].append(val)
                            else:
                                accum_features[key] = [val]

                    accum_images.append(images)
                    accum_texts.append(texts)

                # If (i + 1) % grad_accum_steps is not zero, move on to the next batch.
                if ((step + 1) % VLE_TRAIN_CONFIG['grad_accum_steps']) > 0:
                    continue

                # Now, ready to take gradients for the last accum_freq batches.
                # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
                # Call backwards each time, but only step optimizer at the end.
                optimizer.zero_grad()
                for j in range(VLE_TRAIN_CONFIG['grad_accum_steps']):
                    images = accum_images[j]
                    texts = accum_texts[j]
                    with autocast():
                        vle_output = vle.encode_and_project(images, texts, broadcast=False)
                        vle_output_dict = {f.name: getattr(vle_output, f.name) for f in dataclasses.fields(vle_output) if f.name in ['global_image_token', 'global_text_token', 'local_image_tokens']}

                        inputs = {}
                        for key, val in accum_features.items():
                            accumulated = accum_features[key]
                            inputs[key] = torch.cat(accumulated[:j] + [vle_output_dict[key]] + accumulated[j + 1:])

                        losses = criterion(
                                image_features=inputs['global_image_token'],
                                image_tokens=inputs['local_image_tokens'].clone(),
                                text_features=inputs['global_text_token'].squeeze(1),
                                logit_scale=vle.model.logit_scale.exp(),
                                logit_bias=vle.model.logit_bias,
                                visual_proj=vle.model.visual_proj,
                                output_dict=True,
                        )
                        del inputs
                        total_loss = sum(losses.values())
                        losses["loss"] = total_loss

                    backward(total_loss, scaler)

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

            if VLE_TRAIN_CONFIG['grad_clip_norm'] is None:
                grad_norm = torch.nn.utils.clip_grad_norm_(vle.model.parameters(), float('inf'), norm_type=2.0)

            # reset gradient accum, if enabled
            if VLE_TRAIN_CONFIG['grad_accum_steps'] > 1:
                accum_images, accum_texts, accum_features = [], [], {}

            with torch.no_grad():
                vle.model.logit_scale.clamp_(0, math.log(100))
            
            if (step_accum+1) % CONFIG['vle']['train']['log_every'] == 0:
                log_manager.log_scores(f"epoch: {epoch+1}/{VLE_TRAIN_CONFIG['num_epochs']} ({global_epoch+1}), step: {step_accum+1}/{num_batches} ({global_step_counter+1})", total_loss, None, global_step_counter+1, "train", f", lr: {optimizer.param_groups[0]['lr']:.2e}, grad_norm: {grad_norm:.2f}" ,"batch_")

            torch.cuda.synchronize() if CONFIG["device"] == "cuda" else None
            
        val_loss = vle.evaluate(val_dl, criterion)

        log_manager.log_scores(f"epoch: {epoch+1}/{VLE_TRAIN_CONFIG['num_epochs']} ({global_epoch+1}), VALIDATION", val_loss, None, global_epoch+1, "val", None,"val_")
        
        if val_loss > best_val_loss:
            best_val_loss = val_loss
            
            # best checkpoint saving
            if VLE_TRAIN_CONFIG['save_weights_path']:
                new_checkpoint_dict = {
                        'global_epoch': global_epoch,
                        'global_step_counter': global_step_counter,
                        'model_state_dict': vle.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                }
                new_checkpoint_dict |= {'scaler_state_dict': scaler.state_dict()} if scaler else {}

                ckp_filename = f"flair-{vle.version}-{VLE_TRAIN_CONFIG['exp_name']}-e{global_epoch}"
                full_ckp_path = Path(VLE_TRAIN_CONFIG['save_weights_path']) / f"{ckp_filename}.pth"
                torch.save(new_checkpoint_dict, full_ckp_path)
                log_manager.log_line(f"Model {full_ckp_path} successfully saved.")

    log_manager.log_title("Training Finished")

def main() -> None:

    mask_color: str = VLE_TRAIN_CONFIG['mask_color'] # can be 'RB' or 'L'

    # Datasets
    train_image_text_ds = ImageCaptionDataset(
        ConcatDataset([
            ImageDataset(Path(f'/home/olivieri/exp/data/data_gen/VOC2012/train/images_{mask_color}')),
            ImageDataset(Path(f'/home/olivieri/exp/data/data_gen/COCO2017/val/images_{mask_color}')),
        ]),
        ConcatDataset([
            JSONLDataset(Path('/home/olivieri/exp/data/data_gen/VOC2012/train/captions.jsonl')),
            JSONLDataset(Path('/home/olivieri/exp/data/data_gen/COCO2017/val/captions.jsonl')),
        ])
    )
    val_image_text_ds = ImageCaptionDataset(
        ImageDataset(Path(f'/home/olivieri/exp/data/data_gen/VOC2012/val/images_{mask_color}')),
        JSONLDataset(Path('/home/olivieri/exp/data/data_gen/VOC2012/val/captions.jsonl'))
    )

    # Vision-Language Encoder
    vle: VLEncoder = VLE_REGISTRY.get("flair", device=CONFIG['device'], vision_adapter=True)

    if VLE_TRAIN_CONFIG['resume_path']:
        checkpoint_dict = torch.load(Path(VLE_TRAIN_CONFIG['resume_path']))
        vle.model.load_state_dict(checkpoint_dict['model_state_dict']) # load my weights
    else:
        checkpoint_dict = None
    vle.set_vision_trainable_params(['proj', 'visual_proj', 'vision_adapter'])

    # TODO try to set the weights of 'vision_adapter' to 0 and see what happens.

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
        add_mps_loss=True, # as suggested by the authors (depending by the trainable params, this might not affect the training)
        num_caps_per_img=1
    )

    log_manager.log_intro(
        config=CONFIG,
        train_ds=train_image_text_ds,
        val_ds=val_image_text_ds,
        train_dl=train_image_text_dl,
        val_dl=val_image_text_dl
    )
    
    # Log trainable parameters
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
