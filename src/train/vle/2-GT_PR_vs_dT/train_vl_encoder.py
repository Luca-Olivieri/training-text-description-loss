from core.config import *
from core.data import crop_image_preprocess_image_text_batch, crop_augment_preprocess_batch
from core.datasets import JSONLDataset, ImageDataset, ImageCaptionDataset, VOC2012SegDataset, COCO2017SegDataset
from core.data_utils import dispatch_and_process_in_batches

from core.viz import get_layer_numel_str
from core.logger import LogManager
from core.torch_utils import compile_torch_model, map_tensors
from core.pipeline import format_pos_ovr_masks

from models.vle import VLE_REGISTRY, VLEncoder, NewLayer, OldFLAIRLayer
from models.seg import SEGMODELS_REGISTRY, SegModelWrapper

from pathlib import Path
from collections import OrderedDict

import torch
from functools import partial
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, ConcatDataset
import math
from open_clip_train.precision import get_autocast # for AMP
from torch.amp import GradScaler # for AMP
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from torchvision.transforms._presets import SemanticSegmentation

from core._types import Callable, Optional

from vendors.flair.src.flair.train import backward

config = setup_config(BASE_CONFIG, Path('/home/olivieri/exp/src/train/vle/2-GT_PR_vs_dT/config.yml'))

vle_config = config['vle']
vle_train_config = vle_config['train']

seg_config = config['seg']

config['var_name'] += f'-{config["timestamp"]}'

exp_path: Path = config['root_exp_path'] / config['exp_name'] / config['var_name']

logs_path: Path = exp_path/'logs'

save_weights_root_path: Path = exp_path/'weights'

if vle_train_config['log_only_to_stdout']:
    log_manager = LogManager(
        exp_name=f'{config["exp_name"]}-{config["var_name"]}',
        exp_desc=config['exp_desc'],
    )
else:
    log_manager = LogManager(
        exp_name=f'{config["exp_name"]}-{config["var_name"]}',
        exp_desc=config['exp_desc'],
        file_logs_dir_path=logs_path,
        tb_logs_dir_path=logs_path
    )

def train_loop(
        vle: VLEncoder,
        train_dl: DataLoader,
        val_dl: DataLoader,
        criterion: Callable,
        voc2012_train_dataset: VOC2012SegDataset,
        coco2017_train_dataset: COCO2017SegDataset,
        voc2012_val_dataset: VOC2012SegDataset,
        coco2017_val_dataset: COCO2017SegDataset,
        segmodel: SegModelWrapper,
        checkpoint_dict: Optional[dict] = None
) -> None:
    
    # --- 1. Initialization and State Restoration ---
    start_epoch = 0
    global_step = 0
    if checkpoint_dict:
        start_epoch = checkpoint_dict['epoch'] + 1
        global_step = checkpoint_dict['global_step']
        log_manager.log_line(f"Resuming training from epoch {start_epoch}, global step {global_step}.")

    num_batches_per_epoch = len(train_dl)
    # The number of optimizer steps per epoch
    num_steps_per_epoch = math.ceil(num_batches_per_epoch)
    
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
        lr=vle_train_config['lr_schedule']['base_lr'],
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    if checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

    # --- 3. Scheduler Setup ---
    total_steps = num_steps_per_epoch * vle_train_config['num_epochs']
    sched_config = vle_train_config['lr_schedule']
    
    scheduler = None
    if sched_config['policy'] == 'const':
        scheduler = const_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)
    elif sched_config['policy'] == 'const-cooldown':
        cooldown_steps = num_steps_per_epoch * sched_config['epochs_cooldown']
        scheduler = const_lr_cooldown(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps, cooldown_steps, sched_config['lr_cooldown_power'], sched_config['lr_cooldown_end'])
    elif sched_config['policy'] == 'cosine':
        scheduler = cosine_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)
    
    # --- 4. AMP and Model Compilation Setup ---
    autocast = get_autocast(vle_train_config['precision'])
    scaler = GradScaler() if vle_train_config['precision'] == "amp" else None
    if scaler and checkpoint_dict:
        scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])
    
    vle.model = compile_torch_model(vle.model)

    # --- SEG ---

    seg_collate_fn = partial(
        crop_augment_preprocess_batch,
        crop_fn=T.CenterCrop(seg_config['image_size']),
        augment_fn=None,
        preprocess_fn=None
    )
    seg_preprocess_fn = partial(SemanticSegmentation, resize_size=seg_config['image_size'])()

    # --- 5. Initial Validation ---
    log_manager.log_title("Initial Validation")
    val_loss = evaluate(vle, val_dl, criterion, voc2012_val_dataset, coco2017_val_dataset, segmodel, seg_collate_fn, seg_preprocess_fn)
    log_manager.log_scores(
        title=f"Before any weight update, VALIDATION",
        loss=val_loss,
        metrics_score=None,
        tb_log_counter=start_epoch,
        tb_phase="val",
        suffix=None,
        metrics_prefix="val_"
    )
    best_val_loss = val_loss

    log_manager.log_title("Training Start")
    
    # --- 6. Main Training Loop ---
    for epoch in range(start_epoch, vle_train_config["num_epochs"]):
        
        vle.model.train()

        for step, (images, texts, texts_metadata) in enumerate(train_dl):
            
            imgs_uids = [d['img_uid'] for d in texts_metadata]
            pos_classes = [d['pos_class'] for d in texts_metadata]

            is_from_coco = [not ('_' in uid) for uid in imgs_uids]

            seg_batch = [retrieve_img_by_uid(voc2012_train_dataset, coco2017_train_dataset, uid) for uid in imgs_uids]

            scs_img, gts = seg_collate_fn(seg_batch)

            with torch.inference_mode(), autocast():
                scs: torch.Tensor = seg_preprocess_fn(scs_img)
                
                scs = scs.to(config["device"])
                gts = gts.to(config["device"]) # shape [N, H, W]

                logits = segmodel.model(scs)
            logits: torch.Tensor = logits["out"] if isinstance(logits, OrderedDict) else logits # shape [N, C, H, W]
            prs = logits.argmax(dim=1, keepdim=True)

            # NOTE only for COCO, maps pr masks indices to COCO ones.
            for i in range(len(imgs_uids)):
                if is_from_coco[i]:
                    prs[i] = map_tensors(prs[i], coco2017_train_dataset.voc_idx_to_coco_idx)
                    prs[i] = map_tensors(prs[i], coco2017_train_dataset.get_class_map())
            
            scs_img = (scs_img*255).to(torch.uint8)

            gts = gts.unsqueeze(1)

            concat_masks = torch.cat([gts, prs], dim=0)

            ovr_concat_masks = format_pos_ovr_masks(scs_img, concat_masks, pos_classes)

            ovr_concat_masks = vle.preprocess_images(ovr_concat_masks, device=config['device'])

            # --- Forward Pass and Loss Calculation ---
            optimizer.zero_grad()
            with autocast():
                b = len(texts)

                vle_output = vle.encode_and_project(ovr_concat_masks, texts, broadcast=False, pool=False)
                
                gt_global_image_token = vle_output.global_image_token[:b]
                pr_global_image_token = vle_output.global_image_token[b:]

                gt_local_image_tokens = vle_output.local_image_tokens[:b]
                pr_local_image_tokens = vle_output.local_image_tokens[b:]

                lhs_global: torch.Tensor = vle.model.concat_adapter(torch.cat([gt_global_image_token, pr_global_image_token], dim=-1))
                lhs_local: torch.Tensor = vle.model.concat_adapter(torch.cat([gt_local_image_tokens, pr_local_image_tokens], dim=-1))

                rhs = vle_output.global_text_token.squeeze(1)
                
                losses = criterion(
                    image_features=lhs_global,
                    image_tokens=lhs_local.clone(),
                    text_features=rhs,
                    logit_scale=vle.model.logit_scale.exp(),
                    logit_bias=vle.model.logit_bias,
                    visual_proj=vle.model.visual_proj,
                    output_dict=True
                )
                total_loss = sum(losses.values())
            
            backward(total_loss, scaler)
            
            # --- Optimizer Step and Scheduler Update ---
            if scheduler:
                scheduler(global_step)
            
            grad_norm = 0.0
            if scaler:
                if vle_train_config['grad_clip_norm']:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(vle.model.parameters(), vle_train_config['grad_clip_norm'], norm_type=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                if vle_train_config['grad_clip_norm']:
                    grad_norm = torch.nn.utils.clip_grad_norm_(vle.model.parameters(), vle_train_config['grad_clip_norm'], norm_type=2.0)
                optimizer.step()

            global_step += 1 # Increment global step *only* after an optimizer step

            if not vle_train_config['grad_clip_norm']:
                grad_norm = torch.nn.utils.clip_grad_norm_(vle.model.parameters(), float('inf'), norm_type=2.0)

            with torch.no_grad():
                vle.model.logit_scale.clamp_(0, math.log(100))
            
            # --- Logging ---
            if global_step % vle_train_config['log_every'] == 0:
                current_lr = optimizer.param_groups[0]['lr']
                step_in_epoch = step + 1
                log_manager.log_scores(
                    title=f"epoch: {epoch+1}/{vle_train_config['num_epochs']}, step: {step_in_epoch}/{num_steps_per_epoch} (global_step: {global_step})",
                    loss=total_loss,
                    metrics_score=None,
                    tb_log_counter=global_step,
                    tb_phase="train",
                    suffix=f", lr: {current_lr:.2e}, grad_norm: {grad_norm:.2f}",
                    metrics_prefix="batch_"
                )

        val_loss = evaluate(vle, val_dl, criterion, voc2012_val_dataset, coco2017_val_dataset, segmodel, seg_collate_fn, seg_preprocess_fn)
        log_manager.log_scores(
            title=f"epoch: {epoch+1}/{vle_train_config['num_epochs']}, VALIDATION",
            loss=val_loss, 
            metrics_score=None,
            tb_log_counter=epoch+1,
            tb_phase="val",
            suffix=None,
            metrics_prefix="val_"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            if save_weights_root_path:
                # NOTE on resume we start from 'epoch + 1'
                new_checkpoint_dict = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': vle.model._orig_mod.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if scaler:
                    new_checkpoint_dict['scaler_state_dict'] = scaler.state_dict()

                save_weights_root_path.mkdir(parents=False, exist_ok=True)
                ckp_filename = f'{vle.version}-{config["exp_name"]}-{config["var_name"]}.pth'
                full_ckp_path = save_weights_root_path / ckp_filename
                torch.save(new_checkpoint_dict, full_ckp_path)
                log_manager.log_line(f"New best model saved to {full_ckp_path} with validation loss: {best_val_loss:.4f}")

    log_manager.log_title("Training Finished")

def evaluate(
        vle: VLEncoder,
        dl: DataLoader,
        criterion: Callable,
        voc2012_val_dataset: VOC2012SegDataset,
        coco2017_val_dataset: COCO2017SegDataset,
        segmodel: SegModelWrapper,
        seg_collate_fn: Callable,
        seg_preprocess_fn: Callable
) -> torch.Tensor:
        running_loss = 0.0
        running_supcount = 0

        vle.model.eval()

        with torch.no_grad():

            for step, (images, texts, texts_metadata) in enumerate(dl):
                imgs_uids = [d['img_uid'] for d in texts_metadata]
                pos_classes = [d['pos_class'] for d in texts_metadata]

                is_from_coco = [not ('_' in uid) for uid in imgs_uids]

                seg_batch = [retrieve_img_by_uid(voc2012_val_dataset, coco2017_val_dataset, uid) for uid in imgs_uids]

                scs_img, gts = seg_collate_fn(seg_batch)

                with torch.inference_mode():
                    scs: torch.Tensor = seg_preprocess_fn(scs_img)
                    
                    scs = scs.to(config["device"])
                    gts = gts.to(config["device"]) # shape [N, H, W]

                    logits = segmodel.model(scs)
                logits: torch.Tensor = logits["out"] if isinstance(logits, OrderedDict) else logits # shape [N, C, H, W]
                prs = logits.argmax(dim=1, keepdim=True)

                # NOTE only for COCO, maps pr masks indices to COCO ones.
                for i in range(len(imgs_uids)):
                    if is_from_coco[i]:
                        prs[i] = map_tensors(prs[i], coco2017_val_dataset.voc_idx_to_coco_idx)
                        prs[i] = map_tensors(prs[i], coco2017_val_dataset.get_class_map())

                scs_img = (scs_img*255).to(torch.uint8)

                gts = gts.unsqueeze(1)

                concat_masks = torch.cat([gts, prs], dim=0)

                ovr_concat_masks = format_pos_ovr_masks(scs_img, concat_masks, pos_classes)

                ovr_concat_masks = vle.preprocess_images(ovr_concat_masks, device=config['device'])
                
                b = len(gts)
                vle_output = vle.encode_and_project(ovr_concat_masks, texts, broadcast=False, pool=False)
                
                gt_global_image_token = vle_output.global_image_token[:b]
                pr_global_image_token = vle_output.global_image_token[b:]

                gt_local_image_tokens = vle_output.local_image_tokens[:b]
                pr_local_image_tokens = vle_output.local_image_tokens[b:]

                lhs_global: torch.Tensor = vle.model.concat_adapter(torch.cat([gt_global_image_token, pr_global_image_token], dim=-1))
                lhs_local: torch.Tensor = vle.model.concat_adapter(torch.cat([gt_local_image_tokens, pr_local_image_tokens], dim=-1))

                rhs = vle_output.global_text_token.squeeze(1)
                
                losses = criterion(
                    image_features=lhs_global,
                    image_tokens=lhs_local.clone(),
                    text_features=rhs,
                    logit_scale=vle.model.logit_scale.exp(),
                    logit_bias=vle.model.logit_bias,
                    visual_proj=vle.model.visual_proj,
                    output_dict=True
                )
                total_batch_loss = sum(losses.values())

                with torch.no_grad():
                    vle.model.logit_scale.clamp_(0, math.log(100))

                running_loss += total_batch_loss.item() * images.size(0)
                running_supcount += images.size(0)
        
        loss = running_loss / running_supcount
        
        return loss

def retrieve_img_by_uid(
        voc2012_dataset: VOC2012SegDataset,
        coco2017_dataset: COCO2017SegDataset,
        uid: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if '_' in uid:
        return voc2012_dataset.get_imgs_by_uid(uid)
    else:
        return coco2017_dataset.get_imgs_by_uid(uid)

def main() -> None:
    # idxs = slice(0, 32, 1)
    idxs=None

    train_image_text_ds = ConcatDataset([
        ImageCaptionDataset(
            ImageDataset(config['datasets_paths']['voc2012']['train']['imgs'], device=config['device'], idxs=idxs),
            JSONLDataset(config['datasets_paths']['voc2012']['train']['jsonl'], line_idxs=idxs),
        ),
        ImageCaptionDataset( # 1st part
            ImageDataset(config['datasets_paths']['coco2017']['train']['imgs']['l0'], device=config['device'], idxs=idxs),
            JSONLDataset(config['datasets_paths']['coco2017']['train']['jsonl']['l0'], line_idxs=idxs),
        ),
        ImageCaptionDataset( # 2st part
            ImageDataset(config['datasets_paths']['coco2017']['train']['imgs']['l1'], device=config['device'], idxs=idxs),
            JSONLDataset(config['datasets_paths']['coco2017']['train']['jsonl']['l1'], line_idxs=idxs),
        ),
    ])

    val_image_text_ds = ConcatDataset([
        ImageCaptionDataset(
            ImageDataset(config['datasets_paths']['voc2012']['val']['imgs'], device=config['device'], idxs=idxs),
            JSONLDataset(config['datasets_paths']['voc2012']['val']['jsonl'], line_idxs=idxs),
        ),
        ImageCaptionDataset(
            ImageDataset(config['datasets_paths']['coco2017']['val']['imgs'], device=config['device'], idxs=idxs),
            JSONLDataset(config['datasets_paths']['coco2017']['val']['jsonl'], line_idxs=idxs),
        ),
    ])

    voc2012_train_dataset = VOC2012SegDataset(
        root_path=config['datasets']['VOC2012_root_path'],
        split='train',
        device=config['device'],
        resize_size=config['seg']['image_size'],
        center_crop=True,
        with_unlabelled=False,
    )
    voc2012_val_dataset = VOC2012SegDataset(
        root_path=config['datasets']['VOC2012_root_path'],
        split='val',
        device=config['device'],
        resize_size=config['seg']['image_size'],
        center_crop=True,
        with_unlabelled=False,
    )
    coco2017_train_dataset = COCO2017SegDataset(
        root_path=Path(config['datasets']['COCO2017_root_path']),
        split='train',
        device=config['device'],
        resize_size=vle_config['image_size'],
        center_crop=True,
        only_VOC_labels=True
    )
    coco2017_val_dataset = COCO2017SegDataset(
        root_path=Path(config['datasets']['COCO2017_root_path']),
        split='val',
        device=config['device'],
        resize_size=vle_config['image_size'],
        center_crop=True,
        only_VOC_labels=True
    )

    segmodel = SEGMODELS_REGISTRY.get(
        'lraspp_mobilenet_v3_large',
        pretrained_weights_path=seg_config['pretrained_weights_path'],
        device=config['device'],
        adaptation=None
    )

    # Vision-Language Encoder

    old_layers_to_train = [old_layer for old_layer in OldFLAIRLayer if old_layer.value in vle_train_config['old_layers_to_train']]
    new_layers = [new_layer for new_layer in NewLayer if new_layer.value in vle_config['new_layers']]
    
    vle: VLEncoder = VLE_REGISTRY.get(
        "flair",
        version='flair-cc3m-recap.pt',
        pretrained_weights_root_path=vle_config['pretrained_weights_root_path'],
        new_layers=new_layers,
        device=config['device']
    )

    checkpoint_dict = None
    if vle_train_config['resume_path']:
        resume_path = Path(vle_train_config['resume_path'])
        if resume_path.exists():
            checkpoint_dict = torch.load(resume_path, map_location=config['device'])
            vle.model.load_state_dict(checkpoint_dict['model_state_dict'])
        else:
            raise AttributeError(f"ERROR: Resume path '{resume_path}' not found.")
        
    vle.set_trainable_params([*old_layers_to_train, *new_layers])

    # DataLoaders
    train_collate_fn = partial(
        crop_image_preprocess_image_text_batch,
        crop_fn=T.CenterCrop(vle_config['image_size']),
        preprocess_images_fn=vle.preprocess_images,
        preprocess_texts_fn=vle.preprocess_texts,
        output_text_metadata=True
    )
    val_collate_fn = partial(
        crop_image_preprocess_image_text_batch,
        crop_fn=T.CenterCrop(vle_config['image_size']),
        preprocess_images_fn=vle.preprocess_images,
        preprocess_texts_fn=vle.preprocess_texts,
        output_text_metadata=True
    )

    train_image_text_dl = DataLoader(
        train_image_text_ds,
        batch_size=vle_train_config["batch_size"],
        shuffle=True,
        generator=get_torch_gen(),
        collate_fn=train_collate_fn,
    )
    
    val_image_text_dl = DataLoader(
        val_image_text_ds,
        batch_size=vle_train_config["batch_size"],
        shuffle=False,
        generator=get_torch_gen(),
        collate_fn=val_collate_fn,
    )

    criterion = vle.create_loss(
        add_mps_loss=True,
        num_caps_per_img=1
    )

    log_manager.log_intro(
        config=config,
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
            voc2012_train_dataset,
            coco2017_train_dataset,
            voc2012_val_dataset,
            coco2017_val_dataset,
            segmodel,
            checkpoint_dict
        )
    except KeyboardInterrupt:
        log_manager.log_title("Training Interrupted", pad_symbol='~')

    log_manager.close_loggers()

if __name__ == '__main__':
    main()
