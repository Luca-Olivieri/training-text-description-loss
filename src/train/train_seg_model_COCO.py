from config import *
from data import COCO2017SegDataset, crop_augment_preprocess_batch
from models.seg_models import evaluate, set_trainable_params
from logger import LogManager
from viz import get_layer_numel_str

from functools import partial
from collections import OrderedDict
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import segmentation as segmodels
import torchvision
import torchvision.transforms.v2 as T
from torchvision.transforms._presets import SemanticSegmentation
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
import torchmetrics as tm

from typing import Callable

SEG_CONFIG = CONFIG['seg']
SEG_TRAIN_CONFIG = SEG_CONFIG['train']

if SEG_TRAIN_CONFIG['log_only_to_stdout']:
    log_manager = LogManager(
        exp_name=SEG_TRAIN_CONFIG['exp_name']
    )
else:
    log_manager = LogManager(
        exp_name=SEG_TRAIN_CONFIG['exp_name'],
        file_logs_dir_path=SEG_TRAIN_CONFIG['file_logs_dir_path'],
        tb_logs_dir_path=SEG_TRAIN_CONFIG['tb_logs_dir_path']
    )

def train_loop(
        model: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        criterion: Callable,
        metrics_dict: dict[dict, tm.Metric],
) -> None:
    
    train_metrics = tm.MetricCollection(metrics_dict)

    lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    log_manager.log_title("Initial Validation")
    
    val_loss, val_metrics_score = evaluate(model, val_dl, criterion, metrics_dict)
    log_manager.log_scores(f"Before any weight update, VALIDATION", val_loss, val_metrics_score, 0, "val", None, "val_")

    log_manager.log_title("Training Start")
    
    for epoch in range(SEG_TRAIN_CONFIG["num_epochs"]):

        train_metrics.reset()

        for step, (scs, gts) in enumerate(train_dl):

            model.train()

            scs = scs.to(CONFIG["device"])
            gts = gts.to(CONFIG["device"]) # shape [N, H, W]

            optimizer.zero_grad()

            logits = model(scs)
            logits: torch.Tensor = logits["out"] if isinstance(logits, OrderedDict) else logits # shape [N, C, H, W]

            batch_loss = criterion(logits, gts)
            batch_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) # clip gradients
            optimizer.step()

            train_metrics_score = train_metrics(logits.argmax(dim=1), gts)

            step_counter = epoch*len(train_dl) + step + 1

            # train. logs to file and TensorBoard
            if (step+1) % SEG_TRAIN_CONFIG['log_every'] == 0:
                log_manager.log_scores(f"epoch: {epoch+1}/{SEG_TRAIN_CONFIG['num_epochs']}, step: {step+1}/{len(train_dl)}", batch_loss, train_metrics_score, step_counter, "train", f", lr: {lr:.2e}, grad_norm: {grad_norm:.2f}" ,"batch_")

            torch.cuda.synchronize() if CONFIG['device'] == 'cuda' else None

        val_loss, val_metrics_score = evaluate(model, val_dl, criterion, metrics_dict)

        log_manager.log_scores(f"epoch: {epoch+1}/{SEG_TRAIN_CONFIG['num_epochs']}, VALIDATION", val_loss, val_metrics_score, epoch+1, "val", None,"val_")
    
    log_manager.log_title("Training Finished")

    log_manager.log_title("Final Evaluation")
    
    # final train. logs to file
    train_loss, train_metrics_score = evaluate(model, train_dl, criterion, metrics_dict)
    log_manager.log_scores(f"After {SEG_TRAIN_CONFIG['num_epochs']} epochs of training, TRAINING", train_loss, train_metrics_score, step_counter, "train", None, "train_")
    log_manager.log_scores(f"After {SEG_TRAIN_CONFIG['num_epochs']} epochs of training, VALIDATION", val_loss, val_metrics_score, None, None, None, "val_")

def main() -> None:

    train_ds = COCO2017SegDataset(
        root_path=Path(CONFIG['datasets']['COCO2017_root_path']),
        split='train',
        resize_size=SEG_CONFIG['image_size'],
        center_crop=True,
        only_VOC_labels=False
    )
    
    val_ds = COCO2017SegDataset(
        root_path=Path(CONFIG['datasets']['COCO2017_root_path']),
        split='val',
        resize_size=SEG_CONFIG['image_size'],
        center_crop=True,
        only_VOC_labels=False
    )

    model = segmodels.lraspp_mobilenet_v3_large(
        weights=None,
        weights_backbone=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        num_classes=train_ds.get_num_classes()).to(CONFIG["device"])
    # model.load_state_dict(torch.load(TORCH_WEIGHTS_CHECKPOINTS / ("lraspp_mobilenet_v3_large-enc-pt" + ".pth")))
    model = model.eval()

    set_trainable_params(model, train_decoder_only=SEG_TRAIN_CONFIG['train_decoder_only'])
    
    preprocess_fn = partial(SemanticSegmentation, resize_size=SEG_CONFIG['image_size'])() # same as original one, but with custom resizing

    # cropping functions
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

    criterion = nn.CrossEntropyLoss()

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
        "acc": MulticlassAccuracy(num_classes=train_ds.get_num_classes(), top_k=1, average="micro", multidim_average="global").to(CONFIG["device"]),
        "mIoU": MulticlassJaccardIndex(num_classes=train_ds.get_num_classes(), average="macro").to(CONFIG["device"]),
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
    [log_manager.log_line(t) for t in get_layer_numel_str(model, print_only_total=False, only_trainable=True).split('\n')]

    try:
        train_loop(
        model,
        train_dl,
        val_dl,
        criterion,
        metrics_dict
    )
    except KeyboardInterrupt:
        log_manager.log_title("Training Interrupted")
    
    if SEG_TRAIN_CONFIG['save_weights']:
        torch.save(model.state_dict(), TORCH_WEIGHTS_CHECKPOINTS / 'seg' / f"lraspp_mobilenet_v3_large-{SEG_TRAIN_CONFIG['exp_name']}.pth")
        log_manager.log_line(f"Model 'lraspp_mobilenet_v3_large-{SEG_TRAIN_CONFIG['exp_name']}.pth' successfully saved.")

    log_manager.close_loggers()

if __name__ == '__main__':
    main()
