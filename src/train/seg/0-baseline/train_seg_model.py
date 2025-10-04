from core.config import *
from core.data import VOC2012SegDataset, crop_augment_preprocess_batch
from models.seg import SegModelWrapper, SEGMODELS_REGISTRY
from core.logger import LogManager
from core.viz import get_layer_numel_str

from functools import partial
from collections import OrderedDict
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from torchvision.transforms._presets import SemanticSegmentation
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
import torchmetrics as tm
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
import math

from typing import Optional
from torch.nn.modules.loss import _Loss

SEG_CONFIG = CONFIG['seg']
SEG_TRAIN_CONFIG = SEG_CONFIG['train']

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

def train_loop(
        segmodel: SegModelWrapper,
        train_dl: DataLoader,
        val_dl: DataLoader,
        criterion: _Loss,
        metrics_dict: dict[dict, tm.Metric],
        checkpoint_dict: Optional[dict] = None
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
    ...

    # --- 5. Initial Validation ---
    log_manager.log_title("Initial Validation")
    val_loss, val_metrics_score = segmodel.evaluate(val_dl, criterion, metrics_dict)
    log_manager.log_scores(f"Before any weight update, VALIDATION", val_loss, val_metrics_score, start_epoch, "val", None, "val_")
    best_val_mIoU = val_metrics_score['mIoU']

    log_manager.log_title("Training Start")
    
    # --- 6. Main Training Loop ---
    train_metrics = tm.MetricCollection(metrics_dict)
    for epoch in range(start_epoch, SEG_TRAIN_CONFIG["num_epochs"]):

        train_metrics.reset() # in theory, this can be removed

        for step, (scs, gts) in enumerate(train_dl):

            segmodel.model.train()

            scs = scs.to(CONFIG["device"])
            gts = gts.to(CONFIG["device"]) # shape [B, H, W]

            logits = segmodel.model(scs)
            logits: torch.Tensor = logits["out"] if isinstance(logits, OrderedDict) else logits # shape [N, C, H, W]

            batch_loss: torch.Tensor = criterion(logits, gts) / grad_accum_steps
            batch_loss.backward()

            train_metrics.update(logits.detach().argmax(dim=1), gts)

            is_last_batch = (step + 1) == num_batches_per_epoch
            is_accum_step = (step + 1) % grad_accum_steps == 0

            # --- Optimizer Step and Scheduler Update ---
            if is_accum_step or is_last_batch:
                
                if scheduler:
                    scheduler(global_step)

                max_grad_norm = SEG_TRAIN_CONFIG['grad_clip_norm']
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
                        batch_loss * grad_accum_steps, train_metrics_score, global_step, "train",
                        f", lr: {current_lr:.2e}, grad_norm: {grad_norm:.2f}", "batch_"
                    )

                train_metrics.reset() # only the batch metrics are logged

            # torch.cuda.synchronize() if CONFIG['device'] == 'cuda' else None

        # --- End of Epoch Validation and Checkpointing ---
        val_loss, val_metrics_score = segmodel.evaluate(val_dl, criterion, metrics_dict)
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

                save_dir = Path(SEG_TRAIN_CONFIG['save_weights_root_path'])
                save_dir.mkdir(parents=True, exist_ok=True)
                ckp_filename = f"lraspp_mobilenet_v3_large_{SEG_TRAIN_CONFIG['exp_name']}.pth"
                full_ckp_path = save_dir / ckp_filename
                torch.save(new_checkpoint_dict, full_ckp_path)
                log_manager.log_line(f"New best model saved to {full_ckp_path} with validation mIoU: {best_val_mIoU:.4f}")

        # if best_val_mIoU > 0.3:
        # if epoch == 19:
            # break
    
    log_manager.log_title("Training Finished")

def main() -> None:

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

    segmodel: SegModelWrapper = SEGMODELS_REGISTRY.get(
        'lraspp_mobilenet_v3_large',
        pretrained_weights_path=Path(SEG_CONFIG['pretrained_weights_path']),
        device=CONFIG['device']
    )

    checkpoint_dict = None
    if SEG_TRAIN_CONFIG['resume_path']:
        resume_path = Path(SEG_TRAIN_CONFIG['resume_path'])
        if resume_path.exists():
            checkpoint_dict = torch.load(resume_path, map_location=CONFIG['device'])
            segmodel.model.load_state_dict(checkpoint_dict['model_state_dict'])
        else:
            raise AttributeError(f"ERROR: Resume path '{resume_path}' not found.")

    segmodel.set_trainable_params(train_decoder_only=SEG_TRAIN_CONFIG['train_decoder_only'])
    
    preprocess_fn = partial(SemanticSegmentation, resize_size=SEG_CONFIG['image_size'])() # same as original one, but with custom resizing

    # training cropping functions
    center_crop_fn = T.CenterCrop(SEG_CONFIG['image_size'])
    random_crop_fn = T.RandomCrop(SEG_CONFIG['image_size'])
    
    # augmentations
    augment_fn = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        # T.RandomAffine(degrees=0, scale=(0.5, 2)), # Zooms in and out of the image.
        # T.RandomAffine(degrees=[-30, 30], translate=[0.2, 0.2], scale=(0.5, 2), shear=15), # Full affine transform.
        # T.RandomPerspective(p=0.5, distortion_scale=0.2) # Shears the image
    ])

    train_collate_fn = partial(
        crop_augment_preprocess_batch,
        crop_fn=random_crop_fn,
        augment_fn=augment_fn,
        preprocess_fn=preprocess_fn
    )
    
    val_collate_fn = partial(
        crop_augment_preprocess_batch,
        crop_fn=T.CenterCrop(SEG_CONFIG['image_size']),
        augment_fn=None,
        preprocess_fn=preprocess_fn
    )

    criterion = nn.CrossEntropyLoss(ignore_index=21)

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
        train_loop(
            segmodel,
            train_dl,
            val_dl,
            criterion,
            metrics_dict,
            checkpoint_dict
    )
    except KeyboardInterrupt:
        log_manager.log_title("Training Interrupted", pad_symbol='~')

    log_manager.close_loggers()

if __name__ == '__main__':
    main()
