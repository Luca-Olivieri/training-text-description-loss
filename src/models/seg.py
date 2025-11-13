"""Segmentation model wrappers and utilities.

This module provides abstract base classes and concrete implementations for wrapping
segmentation models with additional functionality such as adaptation layers, 
contrastive learning capabilities, and evaluation utilities.
"""

from __future__ import annotations

from core.config import *
from core.registry import Registry
from core.torch_utils import get_activation

import torch
import torchmetrics as tm
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric
from torchvision.models import segmentation as segmodels
from torchvision.transforms._presets import SemanticSegmentation
from torch.utils.hooks import RemovableHandle

from collections import OrderedDict
from enum import Enum
from functools import partial

from abc import ABC, abstractmethod
from typing import Optional

class SegSaliencyMapMode(Enum):
    """
    Enumeration for different segmentation saliency map modes.
    """
    SEG_GRAD_CAM = 'seg_grad_cam'

class SegModelWrapper(ABC):
    """Abstract base class for segmentation model wrappers.
    
    This class defines the interface that all segmentation model wrappers must implement.
    It provides methods for model evaluation, adaptation with additional layers,
    parameter management, and forward hook management.
    """

    def __init__(self) -> None:
        """Initialize the segmentation model wrapper.
        
        Raises:
            NotImplementedError: This is an abstract base class and must be subclassed.
        """
        raise NotImplementedError
    
    @abstractmethod
    def preprocess_images(
            self,
            images: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def evaluate(
            self,
            **kwargs
    ) -> ...:
        """Evaluate the model on a dataset.
        
        Args:
            **kwargs: Implementation-specific evaluation parameters.
            
        Returns:
            Evaluation results (implementation-specific).
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def adapt(
            self,
            **kwargs
    ) -> ...:
        """Add adaptation layers to the model for specific learning objectives.
        
        Args:
            **kwargs: Implementation-specific adaptation parameters.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def adapt_tensor(
            self,
            **kwargs
    ) -> ...:
        """Apply adaptation transformations to a tensor.
        
        Args:
            **kwargs: Implementation-specific tensor transformation parameters.
            
        Returns:
            Adapted tensor (implementation-specific).
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def set_trainable_params(
            self,
            **kwargs
    ) -> ...:
        """Configure which model parameters should be trainable.
        
        Args:
            **kwargs: Implementation-specific parameter configuration options.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def remove_handles(
            self,
            **kwargs
    ) -> ...:
        """Remove all registered forward hooks from the model.
        
        Args:
            **kwargs: Implementation-specific handle removal parameters.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

SEGMODELS_REGISTRY = Registry[SegModelWrapper]()

@SEGMODELS_REGISTRY.register("lraspp_mobilenet_v3_large")
class LRASPP_MobileNetV3_LargeWrapper(SegModelWrapper):
    """Wrapper for LRASPP MobileNetV3-Large segmentation model with adaptation support.
    
    This wrapper provides a unified interface for the LRASPP MobileNetV3-Large model
    with optional adaptation layers for contrastive learning. It supports loading
    pretrained weights, managing forward hooks, and configuring trainable parameters.
    
    Attributes:
        device: The torch device to run the model on.
        adaptation: Type of adaptation to apply ('contrastive_global', 'contrastive_diff', 
                   'contrastive_local', or None).
        model: The wrapped LRASPP MobileNetV3-Large segmentation model.
        handles: List of registered forward hook handles for cleanup.
        activations: Dictionary storing intermediate activations captured by forward hooks.
    """
    
    def __init__(
            self,
            pretrained_weights_path: Path,
            device: torch.device,
            adaptation: Optional[str] = None
    ) -> None:
        """Initialize the LRASPP MobileNetV3-Large wrapper.
        
        Args:
            pretrained_weights_path: Path to the pretrained model weights file.
            device: The torch device to load the model on.
            adaptation: Type of adaptation layer to add. Options are:
                       - 'contrastive_global': Global average pooling + MLP (512->960, text-side) 
                       - 'contrastive_diff': Global average pooling + MLP (960->512, bottleneck-side)
                       - 'contrastive_local': Local contrastive adaptation (not implemented)
                       - None: No adaptation layer
        """
        self.device = device
        self.adaptation = adaptation
        model = segmodels.lraspp_mobilenet_v3_large(weights=None, weights_backbone=None).to(self.device)
        state_dict: OrderedDict = torch.load(pretrained_weights_path)
        model_state_dict = state_dict.get('model_state_dict', state_dict)
        model.load_state_dict(model_state_dict)
        model.eval()
        self.model = model

        self.preprocess_fn = partial(SemanticSegmentation, resize_size=520)()

        self.handles: list[RemovableHandle] = list()
        self.activations: dict[str, torch.Tensor] = dict()
    
    def preprocess_images(
            self,
            images: torch.Tensor,
    ) -> torch.Tensor:
        return self.preprocess_fn(images)
    
    def adapt(
            self,
    ) -> None:
        """Add adaptation layers to the model based on the specified adaptation type.
        
        This method dynamically adds adaptation modules to the model and registers
        forward hooks to capture intermediate activations from the backbone.
        
        Supported adaptations:
        - 'contrastive_global': Adds a global average pooling layer and an MLP (512->960)
          to transform text features for contrastive learning.
        - 'contrastive_diff': Adds a global average pooling layer and an MLP (960->512)
          to reduce dimensionality of backbone features.
        - 'contrastive_local': Placeholder for local contrastive adaptation (not implemented).
        
        The adaptation layers are initialized with Xavier uniform initialization for weights
        and zeros for biases (if present). Forward hooks are registered on the backbone's
        layer 16 to capture bottleneck activations [B, 960, 32, 32].
        """
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
        """Apply adaptation transformation to an input tensor.
        
        This method processes the input tensor through the adaptation layers
        (global average pooling) based on the adaptation type specified during
        initialization. The MLP transformation is commented out in the current
        implementation.
        
        Args:
            input: Input tensor from the backbone, typically of shape [B, C, H, W].
                  For 'contrastive_global': expects [B, 960, H, W]
                  For 'contrastive_diff': expects [B, 960, H, W]
        
        Returns:
            Adapted tensor after global average pooling and squeezing.
            For 'contrastive_global' and 'contrastive_diff': returns [B, C] where C
            is the number of channels in the input.
        """
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
        """Configure which parameters of the model should be trainable.
        
        This method controls gradient computation for different parts of the model:
        the backbone (encoder), the classifier (decoder), and the bottleneck adapter
        (if present).
        
        Args:
            train_decoder_only: If True, freezes the backbone and only trains the classifier
                              and adapter. If False, trains both backbone and classifier
                              (and adapter if present).
        """
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
        """Evaluate the model on a given dataset.
        
        This method runs the model in evaluation mode (no gradient computation) on the
        provided dataloader, computes the loss and metrics for each batch, and returns
        the aggregated results.
        
        Args:
            dl: DataLoader providing batches of (images, ground_truth_masks).
            criterion: Loss function to compute the loss between predictions and ground truth.
            metrics_dict: Dictionary of metric names to torchmetrics Metric objects for
                         evaluation (e.g., accuracy, IoU, etc.).
        
        Returns:
            A tuple containing:
                - Average loss over the entire dataset (torch.Tensor)
                - Dictionary of computed metric values (dict[str, Any])
        """
        running_loss = 0.0
        running_supcount = 0

        metrics = tm.MetricCollection(metrics_dict)
        metrics.reset()

        self.model.eval()

        with torch.no_grad():
            for step, (scs, gts) in enumerate(dl):
                scs = scs.to(self.device)
                gts = gts.to(self.device) # [B, H, W]
                logits = self.model(scs)
                logits = logits["out"] if isinstance(logits, OrderedDict) else logits # [B, C, H, W]

                batch_loss = criterion(logits, gts)
                running_loss += batch_loss.item() * gts.size(0)
                running_supcount += gts.size(0)

                metrics.update(logits.argmax(dim=1), gts)

                del scs, gts, logits
        
        loss = running_loss / running_supcount
        metrics_score = metrics.compute()
        
        return loss, metrics_score
    
    def compute_seg_grad_cam_map(
            self,
            feature_volume: torch.Tensor, # (B, C_f, H_f, W_f)
            logits: torch.Tensor, # (B, C, H, W)
            target_classes: list # (B)
    ) -> torch.Tensor:
        # bottleneck_out: torch.Tensor = activations['bottleneck'] # (B, 960, 33, 33)
        # bottleneck_out.shape

        # self.remove_handles()

        B = len(feature_volume)

        if any([B != len(t) for t in [feature_volume, logits, target_classes]]):
            raise ValueError(f"'feature_volume', 'logits' and 'target_classes' should have the same length. Got {len(feature_volume)=}, {len(logits)=} and {len(target_classes)=} instead.")
        
        target_scores = logits[torch.arange(B), target_classes, :, :] # (B, H, W)
        target_scores = target_scores.sum(dim=(1, 2)) # (B), summing the logits across the spatial dimensions to obtain a scalar

        grads = torch.autograd.grad(
            target_scores, # (B)
            feature_volume, # (B, ...)
            grad_outputs=torch.ones_like(target_scores) # handles the batch-size through the separation of the computational graph
        )[0]
        
        # Step 1: GAP of gradients to get weights (alpha)
        # Shape: 'grads' is (B, C_f, H_f, W_f) -> 'weights' is (B, C_f)
        weights = grads.mean(dim=(2, 3))  # average over spatial dimensions

        # Step 2: Weighted combination of feature maps
        # Shape: 'bottleneck_out' is (B, C_f, H_f, W_f), 'weights' is (B, C_f)
        cam = torch.einsum('bchw,bc->bhw', feature_volume, weights) # batched dot product between feature maps and weights

        # Step 3: Apply ReLU (only keep positive influences)
        cam = torch.relu(cam)

        return cam

    def add_handle(
            self,
            handle: RemovableHandle
    ) -> None:
        self.handles.append(handle)
    
    def remove_handles(
            self,
    ) -> None:
        """Remove all registered forward hooks from the model.
        
        This method cleans up all forward hooks that were registered during the
        adapt() method call. It should be called when the hooks are no longer needed
        to prevent memory leaks and unnecessary computation.
        """
        [h.remove() for h in self.handles]

# used only to validate the correctness of the TorchMetrics metrics
def my_accuracy(
        logits: torch.Tensor,
        gts: torch.Tensor
) -> float:
    """Calculate pixel-wise accuracy for segmentation predictions.
    
    This is a simple accuracy implementation used to validate the correctness
    of the TorchMetrics metrics. It computes the percentage of correctly
    classified pixels across all images in the batch.
    
    Args:
        logits: Model output logits of shape [B, C, H, W] where B is batch size,
               C is number of classes, H is height, and W is width.
        gts: Ground truth segmentation masks of shape [B, H, W] with class indices.
    
    Returns:
        Pixel-wise accuracy as a float between 0 and 1.
    """
    preds = logits.argmax(dim=1, keepdim=False).flatten()
    gts = gts.flatten()
    total = gts.size(0)
    acc = (preds == gts).sum().item() / total
    return acc

