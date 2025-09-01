from config import *
from data import VOC2012SegDataset, crop_augment_preprocess_batch, apply_classmap
from models.seg_models import SegModelWrapper, SEGMODELS_REGISTRY
from models.vl_models import GenParams, OllamaMLLM
from models.vl_encoders import VLE_REGISTRY, VLEncoder, MapComputeMode
from loss import xen_rescaler, lin_rescale_fn, log_rescale_fn, exp_rescale_fn, pow_rescale_fn
from prompter import FastPromptBuilder
from logger import LogManager
from path import get_mask_prs_path
from viz import get_layer_numel_str, create_diff_mask, overlay_map
from utils import clear_memory, compile_torch_model, subsample_sign_classes, blend_tensors

from functools import partial
from collections import OrderedDict
from torch import nn
from torch.utils.data import DataLoader
from open_clip_train.precision import get_autocast # for AMP
from torch.amp import GradScaler # for AMP
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from torchvision.transforms._presets import SemanticSegmentation
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
import torchmetrics as tm
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from vendors.flair.src.flair.train import backward
import math

import asyncio

from typing import Optional, Callable
from torch.nn.modules.loss import _Loss

import torchvision

SEG_CONFIG = CONFIG['seg']
SEG_TRAIN_CONFIG = SEG_CONFIG['train']
SEG_WITH_TEXT_CONFIG = SEG_TRAIN_CONFIG['with_text']
SEG_XEN_RESCALE_CONFIG = SEG_WITH_TEXT_CONFIG['xen_rescale']

VLE_CONFIG = CONFIG['vle']
VLE_TRAIN_CONFIG = VLE_CONFIG['train']

if SEG_TRAIN_CONFIG['log_only_to_stdout']:
    log_manager = LogManager(
        exp_name=SEG_TRAIN_CONFIG['exp_name'],
        exp_desc=SEG_TRAIN_CONFIG['exp_desc'],
    )
else:
    log_manager = LogManager(
        exp_name=SEG_TRAIN_CONFIG['exp_name'],
        exp_desc=SEG_TRAIN_CONFIG['exp_desc'],
        file_logs_dir_path=SEG_TRAIN_CONFIG['file_logs_dir_path'],
        tb_logs_dir_path=SEG_TRAIN_CONFIG['tb_logs_dir_path']
    )

async def train_loop(
        segmodel: SegModelWrapper,
        vlm: OllamaMLLM,
        vle: VLEncoder,
        train_dl: DataLoader,
        val_dl: DataLoader,
        fast_prompt_builder: FastPromptBuilder,
        seg_preprocess_fn: nn.Module,
        gen_params: GenParams,
        train_criterion: _Loss,
        val_criterion: _Loss,
        weights_trend_fn: Callable[[torch.Tensor], torch.Tensor],
        metrics_dict: dict[dict, tm.Metric],
        checkpoint_dict: Optional[dict] = None,
        sign_classes_filter: Optional[Callable[[list[int]], list[int]]] = None,
) -> None:
    
    # --- 1. Initialization and State Restoration ---
    start_epoch = 0
    global_step = 0
    if checkpoint_dict:
        start_epoch = checkpoint_dict['epoch'] + 1
        global_step = checkpoint_dict['global_step']
        log_manager.log_line(f"Resuming training from epoch {start_epoch}, global step {global_step}.")

    grad_accum_steps = SEG_TRAIN_CONFIG['grad_accum_steps']
    num_batches_per_epoch = len(train_dl)
    # The number of optimizer steps per epoch
    num_steps_per_epoch = math.ceil(num_batches_per_epoch / grad_accum_steps)

    # --- 2. Optimizer Setup ---
    lr=SEG_TRAIN_CONFIG['lr_schedule']['base_lr']
    optimizer = torch.optim.AdamW(segmodel.model.parameters(), lr=lr)
    if checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

    # --- 3. Scheduler Setup ---
    total_steps = num_steps_per_epoch * SEG_TRAIN_CONFIG['num_epochs']
    sched_config = SEG_TRAIN_CONFIG['lr_schedule']
    
    scheduler = None
    if sched_config['policy'] == 'const':
        scheduler = const_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)
    elif sched_config['policy'] == 'const-cooldown':
        cooldown_steps = num_steps_per_epoch * sched_config['epochs_cooldown']
        scheduler = const_lr_cooldown(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps, cooldown_steps, sched_config['lr_cooldown_power'], sched_config['lr_cooldown_end'])
    elif sched_config['policy'] == 'cosine':
        scheduler = cosine_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)

    # --- 4. AMP and Model Compilation Setup ---
    autocast = get_autocast(SEG_TRAIN_CONFIG['precision'])
    scaler = GradScaler() if SEG_TRAIN_CONFIG['precision'] == "amp" else None
    if scaler and checkpoint_dict:
        scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])

    # --- 5. Initial Validation ---
    log_manager.log_title("Initial Validation")
    val_loss, val_metrics_score = segmodel.evaluate(val_dl, val_criterion, metrics_dict)
    log_manager.log_scores(f"Before any weight update, VALIDATION", val_loss, val_metrics_score, start_epoch, "val", None, "val_")
    best_val_mIoU = val_metrics_score['mIoU']

    log_manager.log_title("Training Start")
    
    # --- 6. Main Training Loop ---
    train_metrics = tm.MetricCollection(metrics_dict)
    for epoch in range(start_epoch, SEG_TRAIN_CONFIG["num_epochs"]):

        train_metrics.reset() # in theory, this can be removed
        segmodel.model.train()

        cs_counter = 0
        accum_rescale_mult = []

        for step, (scs_img, gts) in enumerate(train_dl):

            # --- Seg --- #

            scs = seg_preprocess_fn(scs_img)

            scs = scs.to(CONFIG["device"])
            gts = gts.to(CONFIG["device"]) # shape [B, H, W]
            
            with autocast():
                logits = segmodel.model(scs)
                logits: torch.Tensor = logits["out"] if isinstance(logits, OrderedDict) else logits # shape [N, C, H, W]

            batch_loss: torch.Tensor = train_criterion(logits, gts) / grad_accum_steps # [B, H, W]

            train_metrics.update(logits.detach().argmax(dim=1), gts)

            if SEG_WITH_TEXT_CONFIG['with_text']:

                # --- VLM --- #

                scs_img = (scs_img*255).to(torch.uint8)
                gts = gts.unsqueeze(1)
                prs = logits.argmax(dim=1, keepdim=True)
                # Both VLM and VLE receive the images in the same downsampled size.
                gts_down = TF.resize(gts, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
                prs_down = TF.resize(prs, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
                scs_down = TF.resize(scs_img, fast_prompt_builder.image_size, TF.InterpolationMode.BILINEAR)
                cs_prompts = fast_prompt_builder.build_cs_inference_prompts(apply_classmap(gts_down, fast_prompt_builder.class_map), apply_classmap(prs_down, fast_prompt_builder.class_map), scs_down, sign_classes_filter)

                batch_idxs = [train_dl.batch_size*step + i for i in range(len(scs_down))]
                
                cs_answer_list = await vlm.predict_many_class_splitted(
                    cs_prompts,
                    batch_idxs,
                    gen_params=gen_params,
                    jsonl_save_path=None,
                    only_text=True,
                    splits_in_parallel=False,
                    batch_size=None,
                    use_tqdm=False
                )

                # --- VLE --- #

                with autocast():

                    batch_maps = []

                    # aggregate the class-splitted global text tokens
                    for i, (img_idx, sc, gt, pr) in enumerate(zip(batch_idxs, scs_down, gts_down, prs_down)):
                        
                        cs_answers = cs_answer_list[i]['content'] # gather the text for each pos. class of this image
                        cs_maps = []

                        for pos_c, ans in cs_answers.items():

                            pos_class_gt = (gt == pos_c)
                            pos_class_pr = (pr == pos_c)

                            diff_mask = create_diff_mask(pos_class_gt, pos_class_pr)

                            # L overlay image
                            ovr_diff_mask_L = blend_tensors(sc, diff_mask*255, CONFIG['data_gen']['alpha'])
                            
                            img_tensor = vle.preprocess_images([ovr_diff_mask_L], device=CONFIG['device'])
                            text_tensor = vle.preprocess_texts([ans], device=CONFIG['device'])
                            map, min_value, max_value = vle.get_maps(
                                img_tensor, text_tensor,
                                map_compute_mode=MapComputeMode.ATTENTION,
                                upsample_size=SEG_CONFIG['image_size'], upsample_mode=TF.InterpolationMode.BILINEAR,
                                attn_heads_idx=[0, 3, 5, 7] # as done by the authors
                            ) # [1, 1, H, W], m, M
                            map = map.squeeze(0).squeeze(0) # [H, W]

                            cs_maps.append(map)

                        reduced_cs_maps = torch.stack(cs_maps, dim=0).mean(dim=0) # [H, W]
                    
                        batch_maps.append(reduced_cs_maps)
                        cs_counter += len(cs_answers.keys())

                    batch_maps = torch.stack(batch_maps) # [B, H, W]
                    batch_maps = weights_trend_fn(batch_maps) # [B, H, W]

                    old_batch_loss = batch_loss.clone()
                    batch_loss = xen_rescaler(batch_loss, batch_maps, SEG_XEN_RESCALE_CONFIG['rescale_alpha'], normalise=False)
                    _rescale_mult = batch_loss.mean()/old_batch_loss.mean()
                    accum_rescale_mult.append(_rescale_mult)
                    del old_batch_loss
            
            backward(batch_loss, scaler)

            # del scs_img, scs, gts, prs, logits, bottleneck_out, global_text_tokens
            
            is_last_batch = (step + 1) == num_batches_per_epoch
            is_accum_step = (step + 1) % grad_accum_steps == 0

            # --- Optimizer Step and Scheduler Update ---
            if is_accum_step or is_last_batch:

                if SEG_WITH_TEXT_CONFIG['with_text']:
                    cs_mult = cs_counter/(train_dl.batch_size*grad_accum_steps)
                    rescale_mult = torch.stack(accum_rescale_mult).mean()
                else:
                    cs_mult = -1.0
                    rescale_mult = -1.0
                
                if scheduler:
                    scheduler(global_step)

                max_grad_norm = SEG_TRAIN_CONFIG['grad_clip_norm']
                if scaler:
                    if SEG_TRAIN_CONFIG['grad_clip_norm']:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            segmodel.model.parameters(),
                            max_grad_norm if max_grad_norm else float('inf'),
                            norm_type=2.0
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if SEG_TRAIN_CONFIG['grad_clip_norm']:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            segmodel.model.parameters(),
                            max_grad_norm if max_grad_norm else float('inf'),
                            norm_type=2.0
                        )
                    optimizer.step()
                
                optimizer.zero_grad()

                global_step += 1 # Increment global step *only* after an optimizer step

                # --- Logging ---
                if global_step % SEG_TRAIN_CONFIG['log_every'] == 0:
                    train_metrics_score = train_metrics.compute()
                    current_lr = optimizer.param_groups[0]['lr']
                    step_in_epoch = (step // grad_accum_steps) + 1
                    log_manager.log_scores(
                        f"epoch: {epoch+1}/{SEG_TRAIN_CONFIG['num_epochs']}, step: {step_in_epoch}/{num_steps_per_epoch} (global_step: {global_step})",
                        batch_loss, train_metrics_score, global_step, "train",
                        f", lr: {current_lr:.2e}, grad_norm: {grad_norm:.2f}, cs_mult: {cs_mult:.2f}, rescale_mult: {rescale_mult:.8f}", "batch_"
                    )
                
                # reset accumulated values
                cs_counter = 0
                accum_rescale_mult = []

                train_metrics.reset() # only the batch metrics are logged

        # --- End of Epoch Validation and Checkpointing ---
        val_loss, val_metrics_score = segmodel.evaluate(val_dl, val_criterion, metrics_dict)
        log_manager.log_scores(f"epoch: {epoch+1}/{SEG_TRAIN_CONFIG['num_epochs']}, VALIDATION", val_loss, val_metrics_score, epoch+1, "val", None, "val_")

        if val_metrics_score['mIoU'] > best_val_mIoU:
            best_val_mIoU = val_metrics_score['mIoU']
            
            if SEG_TRAIN_CONFIG['save_weights_root_path']:
                # Note: 'epoch' is saved, so on resume we start from 'epoch + 1'
                new_checkpoint_dict = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': segmodel.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if scaler:
                    new_checkpoint_dict['scaler_state_dict'] = scaler.state_dict()

                save_dir = Path(SEG_TRAIN_CONFIG['save_weights_root_path'])
                save_dir.mkdir(parents=True, exist_ok=True)
                ckp_filename = f"lraspp_mobilenet_v3_large_{SEG_TRAIN_CONFIG['exp_name']}.pth"
                full_ckp_path = save_dir / ckp_filename
                torch.save(new_checkpoint_dict, full_ckp_path)
                log_manager.log_line(f"New best model saved to {full_ckp_path} with validation mIoU: {best_val_mIoU:.4f}")
    
    log_manager.log_title("Training Finished")

async def main() -> None:
    train_ds = VOC2012SegDataset(
        root_path=Path("/home/olivieri/exp/data/VOCdevkit"),
        split='train',
        resize_size=SEG_CONFIG['image_size'],
        center_crop=False,
        with_unlabelled=True,
    )

    val_ds = VOC2012SegDataset(
        root_path=Path("/home/olivieri/exp/data/VOCdevkit"),
        split='val',
        resize_size=SEG_CONFIG['image_size'],
        center_crop=False,
        with_unlabelled=True,
    )

    # Segmentation Model
    segmodel: SegModelWrapper = SEGMODELS_REGISTRY.get(
        'lraspp_mobilenet_v3_large',
        pretrained_weights_path=Path(SEG_CONFIG['pretrained_weights_path']),
        adaptation='contrastive_global',
        device=CONFIG['device']
    ) # TODO to modify with the actual segnet intermediate checkpoint

    segmodel.adapt()

    segmodel.set_trainable_params(train_decoder_only=False)

    checkpoint_dict = None
    if SEG_TRAIN_CONFIG['resume_path']:
        seg_weights_path = Path(SEG_TRAIN_CONFIG['resume_path'])
        if seg_weights_path.exists():
            checkpoint_dict = torch.load(seg_weights_path, map_location=CONFIG['device'])
            segmodel.model.load_state_dict(checkpoint_dict['model_state_dict'])
        else:
            raise AttributeError(f"ERROR: Resume path '{seg_weights_path}' not found. ")

    # Vision-Language Model
    model_name = "gemma3:12b-it-qat"
    vlm = OllamaMLLM(model_name)

    by_model = "LRASPP_MobileNet_V3"

    gen_params = GenParams(
        seed=CONFIG["seed"],
        temperature=SEG_WITH_TEXT_CONFIG['vlm_temperature'],
    )

    prompt_blueprint={
            "context": "default",
            "color_map": "default",
            "input_format": "sep_ovr_original",
            "task": "default",
            "output_format": "default",
            "support_set_intro": "default",
            "support_set_item": "default",
            "query": "default",
    }

    # NOTE when used in this pipeline, the dataset is useful only to access class maps and color maps, the actual data is not retrieved from here.
    seg_dataset = VOC2012SegDataset(
        root_path=Path(CONFIG['datasets']['VOC2012_root_path']),
        split='train',
        resize_size=CONFIG['seg']['image_size'],
        center_crop=True,
        with_unlabelled=False,
    )

    sup_set_seg_dataset = VOC2012SegDataset(
        root_path=Path(CONFIG['datasets']['VOC2012_root_path']),
        split='prompts_split',
        resize_size=CONFIG['seg']['image_size'],
        center_crop=True,
        with_unlabelled=False,
        mask_prs_path=get_mask_prs_path(by_model=by_model)
    )

    fast_prompt_builder = FastPromptBuilder(
        seg_dataset=seg_dataset,
        seed=CONFIG["seed"],
        prompt_blueprint=prompt_blueprint,
        by_model=by_model,
        alpha=0.6,
        class_map=seg_dataset.get_class_map(with_unlabelled=False),
        color_map=seg_dataset.get_color_map_dict(with_unlabelled=False),
        image_size=CONFIG['vlm']['image_size'],
        sup_set_img_idxs=[16],
        sup_set_seg_dataset=sup_set_seg_dataset,
        str_formats=None,
    )

    # Vision-Language Encoder
    vle: VLEncoder = VLE_REGISTRY.get("flair", version='flair-cc3m-recap.pt', device=CONFIG['device'], vision_adapter=False, text_adapter=False)
    vle_weights_path = Path(SEG_WITH_TEXT_CONFIG['vle_weights_path'])
    if vle_weights_path.exists():
        vle.model.load_state_dict(torch.load(vle_weights_path, map_location=CONFIG['device'])['model_state_dict'])
    else:
        raise AttributeError(f"ERROR: VLE weights path '{vle_weights_path}' not found.")

    vle.set_vision_trainable_params(None)

    clear_memory()
    vle.model = compile_torch_model(vle.model)

    seg_preprocess_fn = partial(SemanticSegmentation, resize_size=SEG_CONFIG['image_size'])() # same as original one, but with custom resizing

    # training cropping functions
    center_crop_fn = T.CenterCrop(SEG_CONFIG['image_size'])
    random_crop_fn = T.RandomCrop(SEG_CONFIG['image_size'])

    # augmentations
    augment_fn = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
    ])

    train_collate_fn = partial(
        crop_augment_preprocess_batch,
        crop_fn=random_crop_fn,
        augment_fn=augment_fn,
        preprocess_fn=None
    )

    val_collate_fn = partial(
        crop_augment_preprocess_batch,
        crop_fn=T.CenterCrop(SEG_CONFIG['image_size']),
        augment_fn=None,
        preprocess_fn=seg_preprocess_fn
    )

    train_criterion = nn.CrossEntropyLoss(ignore_index=21, reduction='none')
    val_criterion = nn.CrossEntropyLoss(ignore_index=21)
    # weights_trend_fn = lin_rescale_fn
    weights_trend_fn = partial(pow_rescale_fn, exp=0.8)

    sign_classes_filter = partial(subsample_sign_classes, k=0)

    train_dl = DataLoader(
        train_ds,
        batch_size=SEG_TRAIN_CONFIG["batch_size"],
        shuffle=True,
        generator=get_torch_gen(),
        collate_fn=train_collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=SEG_TRAIN_CONFIG["batch_size"],
        shuffle=False,
        generator=get_torch_gen(),
        collate_fn=val_collate_fn,
    )

    metrics_dict = {
        "acc": MulticlassAccuracy(num_classes=train_ds.get_num_classes(with_unlabelled=True), top_k=1, average="micro", multidim_average="global", ignore_index=21).to(CONFIG["device"]),
        "mIoU": MulticlassJaccardIndex(num_classes=train_ds.get_num_classes(with_unlabelled=True), average="macro", ignore_index=21).to(CONFIG["device"]),
    }

    log_manager.log_intro(
        config=CONFIG,
        train_ds=train_ds,
        val_ds=val_ds,
        train_dl=train_dl,
        val_dl=val_dl
    )

    # Log trainable parameters
    log_manager.log_title("Trainable Params")
    [log_manager.log_line(t) for t in get_layer_numel_str(segmodel.model, print_only_total=False, only_trainable=True).split('\n')]

    try:
        await train_loop(
            segmodel,
            vlm,
            vle,
            train_dl,
            val_dl,
            fast_prompt_builder,
            seg_preprocess_fn,
            gen_params,
            train_criterion,
            val_criterion,
            weights_trend_fn,
            metrics_dict,
            checkpoint_dict,
            sign_classes_filter
    )
    except KeyboardInterrupt:
        log_manager.log_title("Training Interrupted", pad_symbol='~')

    segmodel.remove_handles()

    log_manager.close_loggers()


if __name__ == '__main__':
    asyncio.run(main())
