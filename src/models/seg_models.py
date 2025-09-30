from config import *
from utils import Registry, get_activation

import torch
import torchmetrics as tm
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric
from torchvision.models import segmentation as segmodels
from collections import OrderedDict
from abc import ABC

from typing import Optional
from torch.utils.hooks import RemovableHandle

SEGMODELS_REGISTRY = Registry()

class SegModelWrapper(ABC):

    def __init__(self) -> None:
        raise NotImplementedError
    
    def evaluate(
            self,
            **kwargs
    ) -> ...:
        raise NotImplementedError
    
    def adapt(
            self,
            **kwargs
    ) -> ...:
        raise NotImplementedError
    
    def adapt_tensor(
            self,
            **kwargs
    ) -> ...:
        raise NotImplementedError
    
    def set_trainable_params(
            self,
            **kwargs
    ) -> ...:
        raise NotImplementedError
    
    def remove_handles(
            self,
            **kwargs
    ) -> ...:
        raise NotImplementedError

@SEGMODELS_REGISTRY.register("lraspp_mobilenet_v3_large")
class LRASPP_MobileNetV3_LargeWrapper(SegModelWrapper):
    
    def __init__(
            self,
            pretrained_weights_path: Path,
            adaptation: Optional[str] = None,
            device: str = 'cuda',
    ) -> None:
        self.device = device
        self.adaptation = adaptation
        model = segmodels.lraspp_mobilenet_v3_large(weights=None, weights_backbone=None).to(self.device)
        state_dict: OrderedDict = torch.load(pretrained_weights_path)
        model_state_dict = state_dict.get('model_state_dict', state_dict)
        model.load_state_dict(model_state_dict)
        model.eval()
        self.model = model

        self.handles: list[RemovableHandle] = list()
        self.activations: dict[str, torch.Tensor] = dict()
    
    def adapt(
            self,
    ) -> None:
        match self.adaptation:
            case 'contrastive_global':
                self.model.add_module('bottleneck_adapter', nn.ModuleDict()) # Module containing all the adaptations
                # GAP layer
                bottleneck_gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
                self.model.bottleneck_adapter.add_module('gap', bottleneck_gap)
                # Dense layer
                bottleneck_mlp = nn.Linear(512, 960, bias=False, device=self.device)
                self.model.bottleneck_adapter.add_module('mlp', bottleneck_mlp)
                # Use Xavier Uniform initialisation for the weight matrix
                nn.init.xavier_uniform_(self.model.bottleneck_adapter.mlp.weight)
                if self.model.bottleneck_adapter.mlp.bias is not None:
                    nn.init.zeros_(self.model.bottleneck_adapter.mlp.bias)
                
                # NOTE should I clone the fw hook output?
                # register the forward hook to store the bottleneck output.
                target_layer: nn.Module = self.model.backbone['16'] # [960, 32, 32] bottleneck output
                handle = target_layer.register_forward_hook(get_activation('bottleneck', self.activations))
                self.handles.append(handle)

            case 'contrastive_diff':
                self.model.add_module('bottleneck_adapter', nn.ModuleDict()) # Module containing all the adaptations
                # GAP layer
                bottleneck_gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
                self.model.bottleneck_adapter.add_module('gap', bottleneck_gap)
                # Dense layer
                bottleneck_mlp = nn.Linear(960, 512, bias=False, device=self.device)
                self.model.bottleneck_adapter.add_module('mlp', bottleneck_mlp)
                # Use Xavier Uniform initialisation for the weight matrix
                nn.init.xavier_uniform_(self.model.bottleneck_adapter.mlp.weight)
                if self.model.bottleneck_adapter.mlp.bias is not None:
                    nn.init.zeros_(self.model.bottleneck_adapter.mlp.bias)
                
                # NOTE should I clone the fw hook output?
                # register the forward hook to store the bottleneck output.
                target_layer: nn.Module = self.model.backbone['16'] # [960, 32, 32] bottleneck output
                handle = target_layer.register_forward_hook(get_activation('bottleneck', self.activations))
                self.handles.append(handle)
            
            case 'contrastive_local':
                ...

    def adapt_tensor(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        match self.adaptation:
            case 'contrastive_global':
                x = self.model.bottleneck_adapter.gap(input).squeeze()
                # x = self.model.bottleneck_adapter.mlp(x)
                return x
            case 'contrastive_diff':
                x = self.model.bottleneck_adapter.gap(input).squeeze()
                # x = self.model.bottleneck_adapter.mlp(x)
                return x
    
    def set_trainable_params(
            self,
            train_decoder_only: bool
    ) -> None:
        if train_decoder_only:
            self.model.backbone.requires_grad_(False)
        else:
            self.model.backbone.requires_grad_(True)
        self.model.classifier.requires_grad_(True)
        
        if hasattr(self.model, 'bottleneck_adapter'):
            self.model.bottleneck_adapter.requires_grad_(True)
    
    def evaluate(
            self,
            dl: DataLoader,
            criterion: nn.modules.loss._Loss,
            metrics_dict: dict[str, Metric],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        running_loss = 0.0
        running_supcount = 0

        metrics = tm.MetricCollection(metrics_dict)
        metrics.reset()

        self.model.eval()

        with torch.no_grad():
            for step, (scs, gts) in enumerate(dl):
                scs = scs.to(CONFIG["device"])
                gts = gts.to(CONFIG["device"]) # [B, H, W]
                logits = self.model(scs)
                logits = logits["out"] if isinstance(logits, OrderedDict) else logits # [B, C, H, W]

                batch_loss = criterion(logits, gts)
                running_loss += batch_loss.item() * gts.size(0)
                running_supcount += gts.size(0)

                metrics.update(logits.argmax(dim=1), gts)

                del scs, gts, logits

                # torch.cuda.synchronize() if CONFIG["device"] == "cuda" else None
        
        loss = running_loss / running_supcount
        metrics_score = metrics.compute()
        
        return loss, metrics_score
    
    def remove_handles(
            self,
    ) -> None:
        [h.remove() for h in self.handles]

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

