from core.config import *
from core.datasets import VOC2012SegDataset
from core.data import crop_augment_preprocess_batch
from models.seg import SegModelWrapper, SEGMODELS_REGISTRY
from core.logger import LogManager
from core.viz import get_layer_numel_str

from functools import partial
from collections import OrderedDict
from torch import nn
from torch.utils.data import DataLoader
from open_clip_train.precision import get_autocast # for AMP
from torch.amp import GradScaler # for AMP
import torchvision.transforms.v2 as T
from torchvision.transforms._presets import SemanticSegmentation
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
import torchmetrics as tm
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from vendors.flair.src.flair.train import backward
from torch.nn.modules.loss import _Loss
import math

import asyncio

from core._types import Optional

config = setup_config(BASE_CONFIG, Path('/home/olivieri/exp/src/train/seg/0-baseline/config.yml'))

seg_config = config['seg']
seg_train_config = seg_config['train']

config['var_name'] += f'-{config["timestamp"]}'

exp_path: Path = config['root_exp_path'] / config['exp_name'] / config['var_name']

logs_path: Path = exp_path/'logs'

save_weights_root_path: Path = exp_path/'weights'

if seg_train_config['log_only_to_stdout']:
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

async def train_loop(
        segmodel: SegModelWrapper,
        train_dl: DataLoader,
        val_dl: DataLoader,
        criterion: _Loss,
        metrics_dict: dict[str, tm.Metric],
        checkpoint_dict: Optional[dict] = None,
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
    lr=seg_train_config['lr_schedule']['base_lr']
    optimizer = torch.optim.AdamW(segmodel.model.parameters(), lr=lr, weight_decay=math.pow(10, -0.5))
    if checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

    # --- 3. Scheduler Setup ---
    total_steps = num_steps_per_epoch * seg_train_config['num_epochs']
    sched_config = seg_train_config['lr_schedule']
    
    scheduler = None
    if sched_config['policy'] == 'const':
        scheduler = const_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)
    elif sched_config['policy'] == 'const-cooldown':
        cooldown_steps = num_steps_per_epoch * sched_config['epochs_cooldown']
        scheduler = const_lr_cooldown(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps, cooldown_steps, sched_config['lr_cooldown_power'], sched_config['lr_cooldown_end'])
    elif sched_config['policy'] == 'cosine':
        scheduler = cosine_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)

    # --- 4. AMP and Model Compilation Setup ---
    autocast = get_autocast(seg_train_config['precision'])
    scaler = GradScaler() if seg_train_config['precision'] == "amp" else None
    if scaler and checkpoint_dict:
        scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])

    # --- 5. Initial Validation ---
    log_manager.log_title("Initial Validation")
    val_loss, val_metrics_score = segmodel.evaluate(val_dl, criterion, metrics_dict, autocast)
    log_manager.log_scores(
        title=f"Before any weight update, VALIDATION",
        loss=val_loss,
        metrics_score=val_metrics_score,
        tb_log_counter=start_epoch,
        tb_phase="val",
        suffix=None,
        metrics_prefix="val_"
    )
    best_val_mIoU = val_metrics_score['mIoU']

    log_manager.log_title("Training Start")
    
    # --- 6. Main Training Loop ---
    train_metrics = tm.MetricCollection(metrics_dict)
    for epoch in range(start_epoch, seg_train_config["num_epochs"]):
        
        train_metrics.reset() # in theory, this can be removed
        segmodel.model.train()

        for step, (uids, scs_img, gts) in enumerate(train_dl):

            # --- Seg --- #

            scs: torch.Tensor = segmodel.preprocess_images(scs_img)

            scs: torch.Tensor = scs.to(config['device'])
            gts: torch.Tensor = gts.to(config['device']) # shape [B, H, W]
            
            with autocast():
                if not 'dropout' in seg_train_config['regularizers']:
                    segmodel.model.eval() # in eval mode, no dropout
                logits = segmodel.model(scs)
                segmodel.model.train()
                logits: torch.Tensor = logits["out"] if isinstance(logits, OrderedDict) else logits # shape [N, C, H, W]

                batch_loss: torch.Tensor = criterion(logits, gts)

            train_metrics.update(logits.detach().argmax(dim=1), gts)

            backward(batch_loss, scaler)

            if scheduler:
                scheduler(global_step)

            max_grad_norm = seg_train_config['grad_clip_norm']
            if scaler:
                if seg_train_config['grad_clip_norm']:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        segmodel.model.parameters(),
                        max_grad_norm if max_grad_norm else float('inf'),
                        norm_type=2.0
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                if seg_train_config['grad_clip_norm']:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        segmodel.model.parameters(),
                        max_grad_norm if max_grad_norm else float('inf'),
                        norm_type=2.0
                    )
                optimizer.step()

            optimizer.zero_grad()

            global_step += 1 # Increment global step *only* after an optimizer step

            # --- Logging ---
            if global_step % seg_train_config['log_every'] == 0:
                train_metrics_score = train_metrics.compute()
                train_metrics_score['lr'] = torch.tensor(optimizer.param_groups[0]['lr'])
                train_metrics_score['grad_norm'] = grad_norm
                step_in_epoch = (step) + 1
                log_manager.log_scores(
                    title=f"epoch: {epoch+1}/{seg_train_config['num_epochs']}, step: {step_in_epoch}/{num_steps_per_epoch} (global_step: {global_step})",
                    loss=batch_loss,
                    metrics_score=train_metrics_score,
                    tb_log_counter=global_step,
                    tb_phase="train",
                    suffix=None,
                    metrics_prefix="batch_"
                )

            train_metrics.reset() # only the batch metrics are logged

        # --- End of Epoch Validation and Checkpointing ---
        val_loss, val_metrics_score = segmodel.evaluate(val_dl, criterion, metrics_dict, autocast)
        log_manager.log_scores(
            title=f"epoch: {epoch+1}/{seg_train_config['num_epochs']}, VALIDATION",
            loss=val_loss,
            metrics_score=val_metrics_score,
            tb_log_counter=epoch+1,
            tb_phase="val",
            suffix=None,
            metrics_prefix="val_"
        )

        if val_metrics_score['mIoU'] > best_val_mIoU:
            best_val_mIoU = val_metrics_score['mIoU']
            
            if save_weights_root_path:
                # NOTE on resume we start from 'epoch + 1'
                new_checkpoint_dict = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': segmodel.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if scaler:
                    new_checkpoint_dict['scaler_state_dict'] = scaler.state_dict()

                save_weights_root_path.mkdir(parents=False, exist_ok=True)
                ckp_filename = f'{seg_config["model_name"]}-{config["exp_name"]}-{config["var_name"]}.pth'
                full_ckp_path = save_weights_root_path / ckp_filename
                torch.save(new_checkpoint_dict, full_ckp_path)
                log_manager.log_line(f"New best model saved to {full_ckp_path} with validation mIoU: {best_val_mIoU:.4f}")
    
    log_manager.log_title("Training Finished")

async def main() -> None:
    img_idxs = None

    # Segmentation Model
    segmodel = SEGMODELS_REGISTRY.get(
        name=seg_config['model_name'],
        pretrained_weights_path=seg_config['pretrained_weights_path'],
        device=config['device'],
        adaptation=seg_config['adaptation']
    )

    if seg_config['checkpoint_path']:
        state_dict: OrderedDict = torch.load(seg_config['checkpoint_path'])
        model_state_dict = state_dict.get('model_state_dict', state_dict)
        segmodel.model.load_state_dict(model_state_dict)

    segmodel.set_trainable_params(train_decoder_only=seg_train_config['train_decoder_only'])

    checkpoint_dict = None
    if seg_train_config['resume_path']:
        seg_weights_path = Path(seg_train_config['resume_path'])
        if seg_weights_path.exists():
            checkpoint_dict = torch.load(seg_weights_path, map_location=config['device'])
            segmodel.model.load_state_dict(checkpoint_dict['model_state_dict'])
        else:
            raise AttributeError(f"ERROR: Resume path '{seg_weights_path}' not found. ")

    train_ds = VOC2012SegDataset(
        root_path=config['datasets']['VOC2012_root_path'],
        split='train',
        device=config['device'],
        resize_size=segmodel.image_size,
        center_crop=False,
        with_unlabelled=True,
        output_uids=True,
        img_idxs=img_idxs
    )
    
    val_ds = VOC2012SegDataset(
        root_path=config['datasets']['VOC2012_root_path'],
        split='val',
        device=config['device'],
        resize_size=segmodel.image_size,
        center_crop=False,
        with_unlabelled=True,
        img_idxs=img_idxs
    )

    # training cropping functions
    if 'random_crop' in seg_train_config['regularizers']:
        crop_fn = T.RandomCrop(segmodel.image_size)
    else:
        crop_fn = T.CenterCrop(segmodel.image_size)

    # augmentations
    augment_fn = T.Compose([
        T.Identity(),
    ])
    if 'random_horizontal_flip' in seg_train_config['regularizers']:
        augment_fn.transforms.append(T.RandomHorizontalFlip(p=0.5))

    train_collate_fn = partial(
        partial(crop_augment_preprocess_batch, output_uids=True),
        crop_fn=crop_fn,
        augment_fn=augment_fn,
        preprocess_fn=None
    )

    val_collate_fn = partial(
        crop_augment_preprocess_batch,
        crop_fn=T.CenterCrop(segmodel.image_size),
        augment_fn=None,
        preprocess_fn=segmodel.preprocess_images
    )

    criterion = nn.CrossEntropyLoss(ignore_index=21)

    train_dl = DataLoader(
        train_ds,
        batch_size=seg_train_config['batch_size'],
        shuffle=True,
        generator=get_torch_gen(),
        collate_fn=train_collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=seg_train_config['batch_size'],
        shuffle=False,
        generator=get_torch_gen(),
        collate_fn=val_collate_fn,
    )

    metrics_dict = {
        "acc": MulticlassAccuracy(num_classes=train_ds.get_num_classes(with_unlabelled=True), top_k=1, average='micro', multidim_average='global', ignore_index=21).to(config['device']),
        "mIoU": MulticlassJaccardIndex(num_classes=train_ds.get_num_classes(with_unlabelled=True), average='macro', ignore_index=21).to(config['device']),
    }

    log_manager.log_intro(
        config=config,
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
            criterion,
            metrics_dict,
            checkpoint_dict,
    )
    except KeyboardInterrupt:
        log_manager.log_title("Training Interrupted", pad_symbol='~')

    segmodel.remove_handles()

    log_manager.close_loggers()


if __name__ == '__main__':
    asyncio.run(main())
