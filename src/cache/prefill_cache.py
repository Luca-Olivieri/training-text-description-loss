from core.config import *
from core.datasets import VOC2012SegDataset
from core.data import crop_augment_preprocess_batch
from models.seg import SegModelWrapper, SEGMODELS_REGISTRY
from core.prompter import FastPromptBuilder
from core.logger import LogManager
from core.viz import get_layer_numel_str
from core.torch_utils import nanstd, map_tensors
from core.utils import subsample_sign_classes
from cache.cache import Cache, PercentilePolicy, MaskTextCache, Identity

from functools import partial
from collections import OrderedDict
from torch import nn
from torch.utils.data import DataLoader
from open_clip_train.precision import get_autocast # for AMP
from torch.amp import GradScaler # for AMP
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from torchvision.transforms._presets import SemanticSegmentation
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex, BinaryJaccardIndex
import torchmetrics as tm
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from torch.nn.modules.loss import _Loss
import math

import asyncio

from core._types import Optional, Callable

prefill_config = load_config(Path('/home/olivieri/exp/src/cache/config.yml'))
ckp_config_path = Path(prefill_config['checkpoint_config'])

ckp_config = setup_config(BASE_CONFIG, ckp_config_path)

ckp_seg_config = ckp_config['seg']
ckp_seg_train_config = ckp_seg_config['train']
ckp_seg_train_with_text_config = ckp_seg_train_config['with_text']

ckp_vlm_config = ckp_config['vlm']

ckp_vle_config = ckp_config['vle']

ckp_config['var_name'] += f'-{ckp_config["timestamp"]}'


# NOTE overriding the original checkpoint config
ckp_seg_train_config['lr_schedule']['base_lr'] = 0.0
ckp_seg_train_with_text_config['loss_lambda'] = 0.0

log_manager = LogManager(
    exp_name=f'prefill_cache-{ckp_config["exp_name"]}-{ckp_config["var_name"]}',
    exp_desc=ckp_config['exp_desc'],
)

async def train_loop(
        segmodel: SegModelWrapper,
        train_dl: DataLoader,
        val_dl: DataLoader,
        fast_prompt_builder: FastPromptBuilder,
        seg_preprocess_fn: nn.Module,
        criterion: _Loss,
        metrics_dict: dict[str, tm.Metric],
        mask_text_cache: MaskTextCache,
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

    num_batches_per_epoch = len(train_dl)
    # The number of optimizer steps per epoch
    num_steps_per_epoch = math.ceil(num_batches_per_epoch)

    # --- 2. Optimizer Setup ---
    lr=ckp_seg_train_config['lr_schedule']['base_lr']
    optimizer = torch.optim.AdamW(segmodel.model.parameters(), lr=lr, weight_decay=math.pow(10, -0.5))
    if checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

    # --- 3. Scheduler Setup ---
    total_steps = num_steps_per_epoch * ckp_seg_train_config['num_epochs']
    sched_config = ckp_seg_train_config['lr_schedule']
    
    scheduler = None
    if sched_config['policy'] == 'const':
        scheduler = const_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)
    elif sched_config['policy'] == 'const-cooldown':
        cooldown_steps = num_steps_per_epoch * sched_config['epochs_cooldown']
        scheduler = const_lr_cooldown(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps, cooldown_steps, sched_config['lr_cooldown_power'], sched_config['lr_cooldown_end'])
    elif sched_config['policy'] == 'cosine':
        scheduler = cosine_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)

    # --- 4. AMP and Model Compilation Setup ---
    autocast = get_autocast(ckp_seg_train_config['precision'])
    scaler = GradScaler() if ckp_seg_train_config['precision'] == "amp" else None
    if scaler and checkpoint_dict:
        scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])

    # --- 5. Initial Validation ---
    log_manager.log_title("Initial Validation")
    val_loss, val_metrics_score = segmodel.evaluate(val_dl, criterion, metrics_dict)
    log_manager.log_scores(
        title=f"Before any weight update, VALIDATION",
        loss=val_loss,
        metrics_score=val_metrics_score,
        tb_log_counter=start_epoch,
        tb_phase="val",
        suffix=None,
        metrics_prefix="val_"
    )

    log_manager.log_title("Training Start")
    
    # --- 6. Main Training Loop ---
    train_metrics = tm.MetricCollection(metrics_dict)
    ckp_seg_train_config["num_epochs"] = start_epoch+1 # only one epoch is done
    for epoch in range(start_epoch, ckp_seg_train_config["num_epochs"]):
        
        train_metrics.reset() # in theory, this can be removed
        segmodel.model.train()

        print(mask_text_cache)

        for step, (uids, scs_img, gts) in enumerate(train_dl):

            cs_counter = 0

            # --- Seg --- #

            scs: torch.Tensor = seg_preprocess_fn(scs_img)

            scs: torch.Tensor = scs.to(ckp_config['device'])
            gts: torch.Tensor = gts.to(ckp_config['device']) # shape [B, H, W]
            
            with autocast():
                if not 'dropout' in ckp_seg_train_config['regularizers']:
                    segmodel.model.eval() # in eval mode, no dropout
                logits = segmodel.model(scs)
                segmodel.model.train()
                logits: torch.Tensor = logits["out"] if isinstance(logits, OrderedDict) else logits # shape [N, C, H, W]

                seg_batch_loss: torch.Tensor = criterion(logits, gts)

            train_metrics.update(logits.detach().argmax(dim=1), gts)

            # --- VLM --- #

            scs_img = (scs_img*255).to(torch.uint8)
            gts = gts.unsqueeze(1)
            prs = logits.argmax(dim=1, keepdim=True)
            # Both VLM and VLE receive the images in the same downsampled size.
            gts_down = TF.resize(gts, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
            prs_down = TF.resize(prs, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
            scs_down = TF.resize(scs_img, fast_prompt_builder.image_size, TF.InterpolationMode.BILINEAR)
            cs_prompts = fast_prompt_builder.build_cs_inference_prompts(map_tensors(gts_down, fast_prompt_builder.class_map), map_tensors(prs_down, fast_prompt_builder.class_map), scs_down, sign_classes_filter)

            prs_by_uid = {uid: pr for uid, pr in zip(uids, prs_down)}

            cs_dict: dict[str, list[int]] = {uid: list(cs_p.keys()) for uid, cs_p in zip(uids, cs_prompts)}

            cs_counter += sum([1 for uid, sign_classes in cs_dict.items() for pos_c in sign_classes])
            
            cs_answer_dict_to_update = {uid: {pos_c: "." for pos_c in pos_classes} for uid, pos_classes in cs_dict.items()}

            # images and texts to update
            cs_splitted_bool_masks_to_update = {uid: {pos_c: prs_by_uid[uid] == pos_c for pos_c in sign_classes} for uid, sign_classes in cs_dict.items()}
            batch = {f"{uid}-{pos_c}": (bool_mask, cs_answer_dict_to_update[uid][pos_c]) for uid, cs_bool_mask in cs_splitted_bool_masks_to_update.items() for pos_c, bool_mask in cs_bool_mask.items()}

            global_step += 1 # Increment global step *only* after an optimizer step

            # --- Logging ---
            if global_step % ckp_seg_train_config['log_every'] == 0:
                train_metrics_score = train_metrics.compute()
                metric_diffs_values = torch.stack(list(mask_text_cache.get_metric_diffs(batch).values()))
                train_metrics_score |= {'metric_diff_mean': metric_diffs_values.nanmean(), 'metric_diff_std': nanstd(metric_diffs_values, dim=0)}
                step_in_epoch = (step) + 1
                log_manager.log_scores(
                    title=f"epoch: {epoch+1}/{ckp_seg_train_config['num_epochs']}, step: {step_in_epoch}/{num_steps_per_epoch} (global_step: {global_step})",
                    loss=seg_batch_loss,
                    metrics_score=train_metrics_score,
                    tb_log_counter=global_step,
                    tb_phase="train",
                    suffix=None,
                    metrics_prefix="batch_"
                )
            
            mask_text_cache.update(batch)

            train_metrics.reset() # only the batch metrics are logged

        # --- End of Epoch Validation and Checkpointing ---
        val_loss, val_metrics_score = segmodel.evaluate(val_dl, criterion, metrics_dict)
        log_manager.log_scores(
            title=f"epoch: {epoch+1}/{ckp_seg_train_config['num_epochs']}, VALIDATION",
            loss=val_loss,
            metrics_score=val_metrics_score,
            tb_log_counter=epoch+1,
            tb_phase="val",
            suffix=None,
            metrics_prefix="val_"
        )

        mask_text_cache.cache.save(
            directory_path=Path(f"{prefill_config['target_path']}/cache/{prefill_config['cache_name']}_k={prefill_config['sign_classes_filter_k']}")
        )

        print(mask_text_cache)
    
    log_manager.log_title("Training Finished")

async def main() -> None:
    train_ds = VOC2012SegDataset(
        root_path=ckp_config['datasets']['VOC2012_root_path'],
        split='train',
        device=ckp_config['device'],
        resize_size=ckp_seg_config['image_size'],
        center_crop=False,
        with_unlabelled=True,
        output_uids=True,
    )
    
    val_ds = VOC2012SegDataset(
        root_path=ckp_config['datasets']['VOC2012_root_path'],
        split='val',
        device=ckp_config['device'],
        resize_size=ckp_seg_config['image_size'],
        center_crop=False,
        with_unlabelled=True,
    )

    # Segmentation Model
    segmodel = SEGMODELS_REGISTRY.get(
        'lraspp_mobilenet_v3_large',
        pretrained_weights_path=ckp_seg_config['pretrained_weights_path'],
        device=ckp_config['device'],
        adaptation=ckp_seg_config['adaptation']
    )

    segmodel.adapt()

    if ckp_seg_config['checkpoint_path']:
        state_dict: OrderedDict = torch.load(ckp_seg_config['checkpoint_path'])
        model_state_dict = state_dict.get('model_state_dict', state_dict)
        segmodel.model.load_state_dict(model_state_dict)

    cache = Cache(storage_device='cpu', memory_device='cuda')
    update_policy = PercentilePolicy(
        metric=BinaryJaccardIndex().to(ckp_config['device']),
        percentile=ckp_seg_train_with_text_config['cache_update_policy_percentile'],
        trend=Identity()
    )
    mask_text_cache = MaskTextCache(cache, update_policy)

    segmodel.model.requires_grad_(False) # the network is frozen

    checkpoint_dict = None
    if ckp_seg_train_config['resume_path']:
        seg_weights_path = Path(ckp_seg_train_config['resume_path'])
        if seg_weights_path.exists():
            checkpoint_dict = torch.load(seg_weights_path, map_location=ckp_config['device'])
            segmodel.model.load_state_dict(checkpoint_dict['model_state_dict'])
        else:
            raise AttributeError(f"ERROR: Resume path '{seg_weights_path}' not found. ")

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
        root_path=ckp_config['datasets']['VOC2012_root_path'],
        split='train',
        device=ckp_config['device'],
        resize_size=ckp_seg_config['image_size'],
        center_crop=True,
        with_unlabelled=False,
    )

    sup_set_seg_dataset = VOC2012SegDataset(
        root_path=ckp_config['datasets']['VOC2012_root_path'],
        split='prompts_split',
        device=ckp_config['device'],
        resize_size=ckp_seg_config['image_size'],
        center_crop=True,
        with_unlabelled=False,
        mask_prs_path=ckp_config['mask_prs_path']
    )

    fast_prompt_builder = FastPromptBuilder(
        seg_dataset=seg_dataset,
        prompts_file_path=ckp_config['prompts_path'] / 'fast_cs_prompt.json',
        prompt_blueprint=prompt_blueprint,
        by_model=ckp_config['by_model'],
        alpha=ckp_vlm_config['alpha'],
        class_map=seg_dataset.get_class_map(with_unlabelled=False),
        color_map=seg_dataset.get_color_map_dict(with_unlabelled=False),
        image_size=ckp_config['vlm']['image_size'],
        sup_set_img_idxs=ckp_vlm_config['sup_set_img_idxs'],
        sup_set_gt_path=ckp_config['sup_set_gt_path'],
        sup_set_seg_dataset=sup_set_seg_dataset,
        str_formats=None,
        seed=ckp_config["seed"],
    )

    seg_preprocess_fn = partial(SemanticSegmentation, resize_size=ckp_seg_config['image_size'])() # same as original one, but with custom resizing

    # training cropping functions
    if 'random_crop' in ckp_seg_train_config['regularizers']:
        crop_fn = T.RandomCrop(ckp_seg_config['image_size'])
    else:
        crop_fn = T.CenterCrop(ckp_seg_config['image_size'])

    # augmentations
    augment_fn = T.Compose([
        T.Identity(),
    ])
    if 'random_horizontal_flip' in ckp_seg_train_config['regularizers']:
        augment_fn.transforms.append(T.RandomHorizontalFlip(p=0.5))

    train_collate_fn = partial(
        partial(crop_augment_preprocess_batch, output_uids=True),
        crop_fn=crop_fn,
        augment_fn=augment_fn,
        preprocess_fn=None
    )

    val_collate_fn = partial(
        crop_augment_preprocess_batch,
        crop_fn=T.CenterCrop(ckp_seg_config['image_size']),
        augment_fn=None,
        preprocess_fn=seg_preprocess_fn
    )

    criterion = nn.CrossEntropyLoss(ignore_index=21)

    if ckp_seg_train_with_text_config['sign_classes_filter_k'] is not None:
        sign_classes_filter = partial(
            subsample_sign_classes,
            k=ckp_seg_train_with_text_config['sign_classes_filter_k']
        ) # only take K PR classes
    else:
        sign_classes_filter = None # take PR all classes

    train_dl = DataLoader(
        train_ds,
        batch_size=ckp_seg_train_config['batch_size'],
        shuffle=True,
        generator=get_torch_gen(),
        collate_fn=train_collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=ckp_seg_train_config['batch_size'],
        shuffle=False,
        generator=get_torch_gen(),
        collate_fn=val_collate_fn,
    )

    metrics_dict = {
        "acc": MulticlassAccuracy(num_classes=train_ds.get_num_classes(with_unlabelled=True), top_k=1, average='micro', multidim_average='global', ignore_index=21).to(ckp_config['device']),
        "mIoU": MulticlassJaccardIndex(num_classes=train_ds.get_num_classes(with_unlabelled=True), average='macro', ignore_index=21).to(ckp_config['device']),
    }

    log_manager.log_intro(
        config=ckp_config,
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
            train_dl,
            val_dl,
            fast_prompt_builder,
            seg_preprocess_fn,
            criterion,
            metrics_dict,
            mask_text_cache,
            checkpoint_dict,
            sign_classes_filter
    )
    except KeyboardInterrupt:
        log_manager.log_title("Training Interrupted", pad_symbol='~')

    segmodel.remove_handles()

    log_manager.close_loggers()

if __name__ == '__main__':
    asyncio.run(main())
