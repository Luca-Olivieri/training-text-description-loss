from config import *
from data import SegDataset, get_image_UIDs, CLASS_MAP_VOID, crop_augment_preprocess_batch, NUM_CLASSES_VOID, SPLITS_PATH
from utils import get_compute_capability
from viz import format_to_title
from models.seg_models import evaluate, set_trainable_params
from logger import log_scores, logger, tb_writer

from functools import partial
from collections import OrderedDict
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import segmentation as segmodels
import torchvision.transforms as T
from torchvision.transforms._presets import SemanticSegmentation
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
import torchmetrics as tm

from typing import Callable

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

    if get_compute_capability() >= 7.0:
        model = torch.compile(model)
    
    # initial val. logs log to file and TensorBoard
    val_loss, val_metrics_score = evaluate(model, val_dl, criterion, metrics_dict)
    log_scores(f"Before any weight update, VALIDATION", val_loss, val_metrics_score, 0, "val", None, "val_")
    
    for epoch in range(CONFIG["seg"]["num_epochs"]):

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

            tb_log_counter = epoch*len(train_dl) + step + 1

            # train. logs to file and TensorBoard
            if (step+1) % CONFIG["log_every"] == 0:
                log_scores(f"epoch: {epoch+1}/{CONFIG['seg']['num_epochs']}, step: {step+1}/{len(train_dl)}", batch_loss, train_metrics_score, tb_log_counter, "train", f", lr: {lr:.2e}, grad_norm: {grad_norm:.2f}" ,"batch_")

            torch.cuda.synchronize() if CONFIG["device"] == "cuda" else None

        val_loss, val_metrics_score = evaluate(model, val_dl, criterion, metrics_dict)

        log_scores(f"epoch: {epoch+1}/{CONFIG['seg']['num_epochs']}, VALIDATION", val_loss, val_metrics_score, epoch+1, "val", None,"val_")
    
    logger.info(format_to_title("Training Finished"))
    
    # final train. logs to file
    train_loss, train_metrics_score = evaluate(model, train_dl, criterion, metrics_dict)
    log_scores(f"After {CONFIG['seg']['num_epochs']} epochs of training, TRAINING", train_loss, train_metrics_score, tb_log_counter, "train", None, "train_")
    log_scores(f"After {CONFIG['seg']['num_epochs']} epochs of training, VALIDATION", val_loss, val_metrics_score, None, None, None, "val_")

def main() -> None:

    train_ds = SegDataset(get_image_UIDs(SPLITS_PATH, split="train", shuffle=False), CONFIG['seg']['image_size'], CLASS_MAP_VOID)
    val_ds = SegDataset(get_image_UIDs(SPLITS_PATH, split="val", shuffle=False), CONFIG['seg']['image_size'], CLASS_MAP_VOID)

    model = segmodels.lraspp_mobilenet_v3_large(weights=None, weights_backbone=None).to(CONFIG["device"])
    # model.load_state_dict(torch.load(MODEL_WEIGHTS_CHECKPOINTS / ("lraspp_mobilenet_v3_large-enc-pt" + ".pth")))
    model.load_state_dict(torch.load(TORCH_WEIGHTS_CHECKPOINTS / ("lraspp_mobilenet_v3_large-1st_half_250630_0910" + ".pth")))
    model.eval()

    set_trainable_params(model, train_decoder_only=CONFIG['seg']['train_decoder_only'])
    
    preprocess_fn = partial(SemanticSegmentation, resize_size=CONFIG['seg']['image_size'])() # same as original one, but with custom resizing

    # cropping functions
    center_crop_fn = T.CenterCrop(CONFIG['seg']['image_size'])
    random_crop_fn = T.RandomCrop(CONFIG['seg']['image_size'])
    
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
        crop_fn=T.CenterCrop(CONFIG['seg']['image_size']),
        augment_fn=None, preprocess_fn=preprocess_fn)

    criterion = nn.CrossEntropyLoss(ignore_index=21)

    train_dl = DataLoader(
        train_ds,
        batch_size=CONFIG["seg"]["batch_size"],
        shuffle=True,
        generator=get_torch_gen(),
        collate_fn=train_collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=CONFIG["seg"]["batch_size"],
        shuffle=False,
        generator=get_torch_gen(),
        collate_fn=val_collate_fn,
    )

    metrics_dict = {
        "acc": MulticlassAccuracy(num_classes=NUM_CLASSES_VOID, top_k=1, average="micro", multidim_average="global", ignore_index=21).to(CONFIG["device"]),
        "mIoU": MulticlassJaccardIndex(NUM_CLASSES_VOID, average="macro", ignore_index=21).to(CONFIG["device"]),
    }

    # TODO speed up things
    # TODO check if the pre-processing can in be coded better.
    # TODO integrate callbacks such as save the best model, etc.
    # TODO Excluding the VOID during training worsen the segmentation visual quality since the borders can be fucked up, it it correct to exclude it?.

    logger.info(format_to_title(CONFIG['exp_name'], pad_symbol='='))
    logger.info(CONFIG["exp_desc"]) if CONFIG["exp_desc"] is not None else None
    logger.info(format_to_title("Config"))
    logger.info(CONFIG)
    logger.info(format_to_title("Data"))
    logger.info(f"- Training data: {len(train_ds)} samples, in {len(train_dl)} mini-batches of size {CONFIG['seg']['batch_size']}")
    logger.info(f"- Validation data: {len(val_ds)} samples, in {len(val_dl)} mini-batches of size {CONFIG['seg']['batch_size']}")
    logger.info(format_to_title("Training Start"))

    try:
        train_loop(
        model,
        train_dl,
        val_dl,
        criterion,
        metrics_dict
    )
    except KeyboardInterrupt:
        logger.info(format_to_title("Training Interrupted", pad_symbol='-'))

    tb_writer.close()
    
    torch.save(model.state_dict(), TORCH_WEIGHTS_CHECKPOINTS / f"lraspp_mobilenet_v3_large-{CONFIG['exp_name']}.pth")
    logger.info(f"Model 'lraspp_mobilenet_v3_large-{CONFIG['exp_name']}.pth' successfully saved.")

if __name__ == '__main__':
    main()
