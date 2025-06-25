from config import *
from data import *
from utils import *
from color_map import apply_colormap, COLOR_MAP_VOID_DICT
from model import *
from logger import get_logger, get_tb_logger

from torchvision.models import segmentation as segmodels
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
from functools import partial
from torchvision.transforms._presets import SemanticSegmentation
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import MulticlassAccuracy
import torchmetrics as tm
from datetime import datetime

def set_trainable_params(
        model: nn.Module,
        train_decoder_only: bool
) -> None:
    model.backbone.requires_grad_(False) if train_decoder_only else model.backbone.requires_grad_(True)
    model.classifier.requires_grad_(True)

def train_loop(
        model: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        criterion: Callable,
        metrics_dict: dict[dict, tm.Metric],
) -> None:
    if get_compute_capability() >= 7.0:
        model = torch.compile(model)

    print(f"- Train. data-loader of {len(train_dl)} mini-batches of size {CONFIG['seg']['batch_size']}")
    print(f"- Val. data-loader of {len(val_dl)} mini-batches of size {CONFIG['seg']['batch_size']}")
    display("\n----------[Training Start]----------\n")

    train_metrics = tm.MetricCollection(metrics_dict)

    lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    CONFIG["exp_name"] += f'_{datetime.now().strftime("%y%m%d_%H%M")}'

    logger = get_logger(CONFIG["log_dir"], CONFIG["exp_name"])
    tb_writer = get_tb_logger(CONFIG["tb_dir"], CONFIG["exp_name"])

    logger.info(CONFIG["exp_name"])

    for epoch in range(CONFIG["seg"]["num_epochs"]):

        train_metrics.reset()

        for step, (scs, gts) in enumerate(train_dl):

            model.train()

            scs = scs.to(CONFIG["device"])
            gts = gts.to(CONFIG["device"])

            optimizer.zero_grad()

            logits: Tensor = model(scs)
            logits = logits["out"] if isinstance(logits, OrderedDict) else logits

            batch_loss = criterion(logits, gts)
            batch_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) # clip gradients
            optimizer.step()

            train_metrics_score = train_metrics(logits.argmax(dim=1), gts)

            log_counter = epoch*len(train_dl) + step

            # logging to file and TensorBoard
            if (log_counter+1) % CONFIG["log_every"] == 0:
                log_str = f"[{epoch+1}/{CONFIG['seg']['num_epochs']}:{step}] loss: {batch_loss:.4f}"
                tb_writer.add_scalar('train/loss', batch_loss, log_counter)
                for m, s in pretty_metrics(train_metrics_score).items():
                    log_str += f", {m}: {s}"
                    tb_writer.add_scalar(f'train/{m}', s, log_counter)
                log_str += f", lr: {lr:.2e}, grad_norm: {grad_norm:.2f}"
                logger.info(log_str)

            torch.cuda.synchronize() if CONFIG["device"] == "cuda" else None

        loss, val_metrics_score = evaluate(model, val_dl, criterion, metrics_dict, "val_dl", tb_writer, "val", epoch)
        print({"loss": f"{loss:.4f}"} | pretty_metrics(val_metrics_score))

def main() -> None:
    
    train_ds = SegDataset(image_train_UIDs, CONFIG["seg"]["image_size"], CLASS_MAP_VOID)
    val_ds = SegDataset(image_val_UIDs, CONFIG["seg"]["image_size"], CLASS_MAP_VOID)
    print(f"{len(train_ds)=}, {len(val_ds)=}")

    model = segmodels.lraspp_mobilenet_v3_large(
        # weights=None, # initialised decoder
        weights=segmodels.LRASPP_MobileNet_V3_Large_Weights.DEFAULT, # pre-trained decoder
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2 # pre-trained backbone
    ).to(CONFIG["device"])
    model.eval()
    
    preprocess = partial(SemanticSegmentation, resize_size=CONFIG["seg"]["image_size"])() # same as default transforms, but resize to 224 (instead of 520), as original backbone is trained on 224x224 ImageNet pictures.

    collate_fn = partial(extract_augment_preprocess_batch, augment_fn=None, preprocess_fn=preprocess)

    criterion = nn.CrossEntropyLoss(ignore_index=21)

    train_dl = DataLoader(
        train_ds,
        batch_size=CONFIG["seg"]["batch_size"],
        shuffle=True,
        generator=torch_gen,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=CONFIG["seg"]["batch_size"],
        generator=torch_gen,
        collate_fn=collate_fn,
    )

    metrics_dict = {
        "acc": MulticlassAccuracy(num_classes=NUM_CLASSES_VOID, top_k=1, average="micro", multidim_average="global", ignore_index=21).to(CONFIG["device"]),
        "mIoU": MeanIoU(num_classes=NUM_CLASSES, include_background=True, per_class=False, input_format="index").to(CONFIG["device"])
    }

    # TODO assert complete data reproducility
    # TODO speed up things
    # TODO refactor everything
    # TODO check if the pre-processing can in be coded better.
    # TODO try SegmentationModels library

    train_loop(
        model,
        train_dl,
        val_dl,
        criterion,
        metrics_dict
    )

if __name__ == '__main__':
    main()
