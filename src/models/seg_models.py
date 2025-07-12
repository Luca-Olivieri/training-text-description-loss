from collections import OrderedDict

import torch
import torchmetrics as tm
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric

from config import *

def set_trainable_params(
        model: nn.Module,
        train_decoder_only: bool
) -> None:
    model.backbone.requires_grad_(False) if train_decoder_only else model.backbone.requires_grad_(True)
    model.classifier.requires_grad_(True)

# used only to validate the correctness of the TorchMetrics metrics
def my_accuracy(
        logits: torch.Tensor,
        gts: torch.Tensor
) -> float:
    preds = logits.argmax(dim=1, keepdim=False).flatten()
    gts = gts.flatten()
    total = gts.size(0)
    acc = (preds == gts).sum().item() / total
    return acc

def evaluate(
        model: nn.Module,
        dl: DataLoader,
        criterion: nn.modules.loss._Loss,
        metrics_dict: dict[str, Metric],
) -> ...:
    running_loss = 0.0
    running_supcount = 0

    metrics = tm.MetricCollection(metrics_dict)
    metrics.reset()

    model.eval()

    with torch.no_grad():
        for step, (scs, gts) in enumerate(dl):
            scs = scs.to(CONFIG["device"])
            gts = gts.to(CONFIG["device"]) # [B, H, W]
            logits = model(scs)
            logits = logits["out"] if isinstance(logits, OrderedDict) else logits # [B, C, H, W]

            batch_loss = criterion(logits, gts)
            running_loss += batch_loss.item() * gts.size(0)
            running_supcount += gts.size(0)

            metrics.update(logits.argmax(dim=1), gts)

            torch.cuda.synchronize() if CONFIG["device"] == "cuda" else None
    
    loss = running_loss / running_supcount
    metrics_score = metrics.compute()
    
    return loss, metrics_score
