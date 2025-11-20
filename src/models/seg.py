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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric
from torchvision.models import segmentation as segmodels
from torchvision.transforms._presets import SemanticSegmentation
from torch.utils.hooks import RemovableHandle
import segmentation_models_pytorch as smp

from collections import OrderedDict
from functools import partial

from abc import ABC, abstractmethod
from typing import Optional


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
        state_dict: OrderedDict = torch.load(pretrained_weights_path, map_location='cpu')
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
        layer 16 to capture bottleneck activations (B, 960, 33, 33).
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
                target_layer: nn.Module = self.model.backbone['16'] # (960, 33, 33) bottleneck output
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
                target_layer: nn.Module = self.model.backbone['16'] # [960, 33, 33] bottleneck output
                handle = target_layer.register_forward_hook(get_activation('bottleneck', self.activations))
                self.handles.append(handle)
            
            case 'contrastive_global_bside_1':
                self.model.bottleneck_adapter = nn.ModuleDict()
                self.model.bottleneck_adapter.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # GAP layer
                self.model.bottleneck_adapter.mlp = nn.Linear(960, 512, bias=False, device=self.device) # linear layer
                
                # Use Xavier Uniform initialisation for the weight matrix
                nn.init.xavier_uniform_(self.model.bottleneck_adapter.mlp.weight)
                if self.model.bottleneck_adapter.mlp.bias is not None:
                    nn.init.zeros_(self.model.bottleneck_adapter.mlp.bias)
                
                # NOTE should I clone the fw hook output?
                # register the forward hook to store the bottleneck output.
                target_layer: nn.Module = self.model.backbone['16'] # (960, 33, 33) bottleneck output
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
            case 'contrastive_global_bside_1':
                x: torch.Tensor = self.model.bottleneck_adapter.gap(input).squeeze()
                x: torch.Tensor = self.model.bottleneck_adapter.mlp(x)
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


@SEGMODELS_REGISTRY.register("deeplabv3_mobilenet_v3_large")
class DeepLabV3_MobileNet_V3_LargeWrapper(SegModelWrapper):
    """
    TODO
    """
    
    def __init__(
            self,
            pretrained_weights_path: Path,
            device: torch.device,
            adaptation: Optional[str] = None
    ) -> None:
        self.device = device
        self.adaptation = adaptation
        model = segmodels.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True).to(self.device)
        state_dict: OrderedDict = torch.load(pretrained_weights_path, map_location='cpu')
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
        self.model.aux_classifier.requires_grad_(False)
        
        if hasattr(self.model, 'bottleneck_adapter'):
            self.model.bottleneck_adapter.requires_grad_(True)


@SEGMODELS_REGISTRY.register("deeplabv3_resnet18")
class DeepLabV3_ResNet18Wrapper(SegModelWrapper):
    """
    TODO
    """

    def __init__(
            self,
            pretrained_weights_path: Path,
            device: torch.device,
            adaptation: Optional[str] = None
    ) -> None:
        self.device = device
        self.adaptation = adaptation
        model = smp.DeepLabV3(
            encoder_name="resnet18", # the paper uses ResNet-101 as the backbone network.
            encoder_weights=None, # the encoder is pre-trained on ImageNet (encoder_weights='imagenet'), but by default we leave it uninit.
            encoder_output_stride=8, # the paper advocates for an output stride of 8 for denser feature maps and better performance.
            decoder_channels=256, # the ASPP module uses 256 filters for its convolutions.
            decoder_atrous_rates=(12, 24, 36), # for an output stride of 8, the atrous rates are doubled to (12, 24, 36).
            classes=21, # VOC2012 classes
            # The upsampling factor should match the output stride. Setting it to None allows the
            # model to infer this automatically, which is consistent with the paper's method of
            # upsampling the final logits by a factor of 8.
            upsampling=None
        ).to(self.device)
        state_dict: OrderedDict = torch.load(pretrained_weights_path, map_location='cpu')
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
            self.model.encoder.requires_grad_(False)
        else:
            self.model.encoder.requires_grad_(True)
        self.model.decoder.requires_grad_(True)
        self.model.segmentation_head.requires_grad_(True)
        
        if hasattr(self.model, 'bottleneck_adapter'):
            self.model.bottleneck_adapter.requires_grad_(True)


@SEGMODELS_REGISTRY.register("deeplabv3_resnet50")
class DeepLabV3_ResNet50Wrapper(SegModelWrapper):
    """
    TODO
    """

    def __init__(
            self,
            pretrained_weights_path: Path,
            device: torch.device,
            adaptation: Optional[str] = None
    ) -> None:
        self.device = device
        self.adaptation = adaptation
        model = segmodels.deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=True).to(self.device)
        state_dict: OrderedDict = torch.load(pretrained_weights_path, map_location='cpu')
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
        self.model.aux_classifier.requires_grad_(False)
        
        if hasattr(self.model, 'bottleneck_adapter'):
            self.model.bottleneck_adapter.requires_grad_(True)


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

def compute_seg_grad_cam(
        feature_volume: torch.Tensor, # (B, C_f, H_f, W_f)
        logits: torch.Tensor, # (B, C, H, W)
        target_classes: list # (B)
) -> torch.Tensor:
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

def compute_seg_grad_cam_pp(
        feature_volume: torch.Tensor, # (B, C_f, H_f, W_f)
        logits: torch.Tensor,         # (B, C, H, W)
        target_classes: list          # (B)
) -> torch.Tensor:
    """
    Computes the Seg-Grad-CAM++ map.
    
    This method follows the logic of Grad-CAM++ but is adapted for segmentation tasks.
    The core idea is to create a class-discriminative localization map by using a
    weighted combination of the feature maps from a specific convolutional layer.
    
    Reference: https://arxiv.org/abs/1710.11063
    """
    B, C_f, H_f, W_f = feature_volume.shape
    
    if any([B != len(t) for t in [feature_volume, logits, target_classes]]):
        raise ValueError(f"'feature_volume', 'logits' and 'target_classes' should have the same length. Got {len(feature_volume)=}, {len(logits)=} and {len(target_classes)=} instead.")

    # Select the logits for the target class for each image in the batch
    target_scores = logits[torch.arange(B), target_classes, :, :] # (B, H, W)
    # Sum the logits across spatial dimensions to get a single score per image
    target_scores = target_scores.sum(dim=(1, 2)) # (B)

    # Compute gradients of the target score with respect to the feature volume
    grads = torch.autograd.grad(
        target_scores, 
        feature_volume,
        grad_outputs=torch.ones_like(target_scores)
    )[0]

    # --- Start of Grad-CAM++ specific logic ---

    # Step 1: Compute the weighting coefficients (alpha)
    # This is the core difference from Grad-CAM.
    
    # Gradients to the power of 2 and 3
    grads_power_2 = grads.pow(2) # (B, C_f, H_f, W_f)
    grads_power_3 = grads.pow(3) # (B, C_f, H_f, W_f)

    # Sum of activations across spatial dimensions
    # Shape: (B, C_f)
    sum_activations = feature_volume.sum(dim=(2, 3))

    # Add small epsilon for numerical stability
    eps = 1e-6

    # Equation 19 from the paper: a_ij^c = (dY/dA_ij)^2 / (2*(dY/dA_ij)^2 + sum(A)* (dY/dA_ij)^3)
    # We name it 'alpha' here for clarity.
    alpha_numerator = grads_power_2
    alpha_denominator = (2 * grads_power_2) + (sum_activations.view(B, C_f, 1, 1) * grads_power_3) + eps
    
    # Element-wise division to get alpha, setting it to 0 where the denominator is 0
    alpha = alpha_numerator / alpha_denominator
    
    # Step 2: Compute the final weights (w_c)
    # Equation 18: w_c = sum_ij(alpha_ij^c * relu(dY/dA_ij^c))
    
    # Apply ReLU to the gradients
    positive_grads = torch.relu(grads) # Only consider positive gradients
    
    # Get the final weights by multiplying alpha with positive gradients and summing over spatial dimensions
    # Shape: (B, C_f)
    weights = (alpha * positive_grads).sum(dim=(2, 3))
    
    # --- End of Grad-CAM++ specific logic ---

    # Step 3: Weighted combination of feature maps (same as Grad-CAM)
    # Shape: (B, H_f, W_f)
    cam = torch.einsum('bchw,bc->bhw', feature_volume, weights)

    # Step 4: Apply ReLU (only keep positive influences)
    cam = torch.relu(cam)

    return cam

def compute_seg_xgrad_cam(
        feature_volume: torch.Tensor, # (B, C_f, H_f, W_f)
        logits: torch.Tensor,         # (B, C, H, W)
        target_classes: list          # (B)
) -> torch.Tensor:
    """
    Computes the Seg-XGrad-CAM map.
    
    This method adapts XGrad-CAM for segmentation. It calculates channel weights
    by scaling the gradients by the normalized feature map activations. This ensures
    that gradients from more "important" feature maps (those with higher activation
    sums) have a larger influence.
    
    Reference: https://arxiv.org/abs/2008.02312
    """
    B, C_f, H_f, W_f = feature_volume.shape
    
    if any([B != len(t) for t in [feature_volume, logits, target_classes]]):
        raise ValueError(f"'feature_volume', 'logits' and 'target_classes' should have the same length. Got {len(feature_volume)=}, {len(logits)=} and {len(target_classes)=} instead.")

    # Select the logits for the target class for each image in the batch
    target_scores = logits[torch.arange(B), target_classes, :, :] # (B, H, W)
    # Sum the logits across spatial dimensions to get a single score per image
    target_scores = target_scores.sum(dim=(1, 2)) # (B)

    # Compute gradients of the target score with respect to the feature volume
    grads = torch.autograd.grad(
        target_scores, 
        feature_volume,
        grad_outputs=torch.ones_like(target_scores)
    )[0]

    # --- Start of XGrad-CAM specific logic ---

    # Step 1: Compute the final weights (w_c)
    # The core idea is to weight the gradients by the activations.
    # w_c = sum_ij (dY/dA_ij^c * A_ij^c) / sum_ij (A_ij^c)

    # Sum of activations across spatial dimensions for normalization
    # Shape: (B, C_f)
    sum_activations = feature_volume.sum(dim=(2, 3))
    
    # Add a small epsilon for numerical stability
    eps = 1e-7

    # Element-wise multiplication of gradients and activations
    # Shape: (B, C_f, H_f, W_f)
    scaled_grads = grads * feature_volume
    
    # Normalize by the sum of activations and then sum to get the final weights.
    # The view() call reshapes sum_activations for broadcasting.
    # Shape: (B, C_f)
    weights = scaled_grads.sum(dim=(2, 3)) / (sum_activations + eps)
    
    # --- End of XGrad-CAM specific logic ---

    # Step 2: Weighted combination of feature maps (same as other CAM methods)
    # Shape: (B, H_f, W_f)
    cam = torch.einsum('bchw,bc->bhw', feature_volume, weights)

    # Step 3: Apply ReLU (only keep positive influences)
    # Although not explicit in the snippet, this is standard practice for CAM visualization.
    cam = torch.relu(cam)

    return cam

def compute_seg_xres_cam(
        feature_volume: torch.Tensor,    # (B, C_f, H_f, W_f)
        logits: torch.Tensor,            # (B, C, H, W)
        target_classes: list,            # (B)
        pool_window: int = 1,
        pool_type: str = 'mean'
) -> torch.Tensor:
    """
    Computes the Seg-XRes-CAM map for a batch of images.

    This method adapts the HiResCAM/XRes-CAM logic for segmentation tasks. Instead of
    globally average pooling the gradients to get a single weight per channel (like Seg-Grad-CAM),
    it performs an element-wise multiplication between the feature maps and their gradients.
    This preserves spatial information, leading to more localized explanations.

    The implementation includes the generalization from the paper (Eq. 5), allowing for
    optional pooling of the gradients to control the explanation's coarseness.

    Args:
        feature_volume (torch.Tensor): The feature maps from the target layer of the model.
                                    Shape: (B, C_f, H_f, W_f).
        logits (torch.Tensor): The final output (segmentation map) from the model.
                            Shape: (B, Num_Classes, H_out, W_out).
        target_classes (list): A list of target class indices for each item in the batch.
                            Length: B.
        pool_window (int): The kernel size for pooling the gradients. A value of 1 corresponds
                        to the original HiResCAM (no pooling). Default is 1.
        pool_type (str): The type of pooling to apply to gradients ('mean' or 'max').
                        Only used if pool_window > 1. Default is 'mean'.

    Returns:
        torch.Tensor: The computed Seg-XRes-CAM maps, one for each image in the batch.
                    Shape: (B, H_f, W_f).
    """
    B = len(feature_volume)

    if not (B == len(logits) == len(target_classes)):
        raise ValueError(
            f"'feature_volume', 'logits' and 'target_classes' should have the same batch size. "
            f"Got {len(feature_volume)=}, {len(logits)=} and {len(target_classes)=} instead."
        )

    # --- This part is the same as Seg-Grad-CAM ---
    # 1. Calculate the score for the target class in the segmentation map.
    #    The paper proposes summing the scores within a desired region (mask M).
    #    Here, we sum across the entire spatial map for the target class.
    target_scores = logits[torch.arange(B), target_classes, :, :]  # (B, H, W)
    target_scores = target_scores.sum(dim=(1, 2))  # (B), scalar score per image

    # 2. Backpropagate to get gradients of the score w.r.t. the feature maps.
    grads = torch.autograd.grad(
        target_scores,
        feature_volume,
        grad_outputs=torch.ones_like(target_scores)
    )[0]  # Shape: (B, C_f, H_f, W_f)

    # --- This is the core difference for Seg-XRes-CAM ---
    element_wise_weights = grads # Start with the original gradients

    # 3. (Optional) Generalization: Pool and Upsample gradients as per Eq. 5
    if pool_window > 1:
        if pool_type == 'mean':
            # NOTE: Using AdaptiveAvgPool2d is more robust if feature map sizes vary.
            # For a fixed kernel, AvgPool2d is what the paper implies.
            pool_layer = torch.nn.AvgPool2d(kernel_size=pool_window, stride=pool_window)
        elif pool_type == 'max':
            pool_layer = torch.nn.MaxPool2d(kernel_size=pool_window, stride=pool_window)
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}. Choose 'mean' or 'max'.")
        
        pooled_grads = pool_layer(grads)
        
        # Upsample the pooled gradients back to the original feature map size
        element_wise_weights = F.interpolate(
            pooled_grads,
            size=feature_volume.shape[2:],
            mode='bilinear',
            align_corners=False
        )

    # 4. Element-wise multiply feature maps with their (potentially pooled) gradients
    # This is the key operation of HiResCAM/XRes-CAM
    weighted_features = feature_volume * element_wise_weights  # Hadamard product

    # 5. Sum the weighted features along the channel dimension
    cam = weighted_features.sum(dim=1)  # (B, H_f, W_f)

    # 6. Apply ReLU to keep only positive contributions
    cam = torch.relu(cam)

    return cam

def compute_seg_hires_grad_cam(
        feature_volume: torch.Tensor, # (B, C_f, H_f, W_f), Activations A^k
        logits: torch.Tensor,         # (B, C, H, W), Model output y
        target_classes: list          # (B), Target class indices c
) -> torch.Tensor:
    """
    Computes the Seg-HiRes-Grad CAM map.

    This method adapts the classification-based HiRes CAM for segmentation tasks.
    Unlike Seg-Grad CAM, it does not average the gradients to create weights.
    Instead, it performs an element-wise multiplication between the feature maps
    and their corresponding gradients, then sums the result across the channel dimension.
    This preserves spatial information and avoids inaccuracies from averaging.ù

    # NOTE: this method works exactly like 'seg_xres_cam' with pool_window=1.

    Args:
        feature_volume (torch.Tensor): The feature maps from the target convolutional layer,
                                    with shape (B, C_f, H_f, W_f).
        logits (torch.Tensor): The final output logits from the model,
                            with shape (B, C, H, W).
        target_classes (list): A list of target class indices for each item in the batch.

    Returns:
        torch.Tensor: The computed Seg-HiRes-Grad CAM heatmap with shape (B, H_f, W_f).
    """
    B = len(feature_volume)

    if not (B == len(logits) == len(target_classes)):
        raise ValueError(
            f"'feature_volume', 'logits' and 'target_classes' must have the same batch size. "
            f"Got {len(feature_volume)=}, {len(logits)=} and {len(target_classes)=} instead."
        )

    # Step 1: Calculate the score for the target class (y_c,new).
    # This corresponds to Eq. (5) in the paper, where the score is the sum of
    # logits for the target class over all spatial locations (pixel set M = all pixels).
    # Shape: (B, H, W) -> (B)
    target_scores = logits[torch.arange(B), target_classes, :, :].sum(dim=(1, 2))

    # Step 2: Compute the gradients of the score with respect to the feature maps.
    # This calculates ∂(y_c,new) / ∂(A^k), as shown in Eq. (6).
    grads = torch.autograd.grad(
        outputs=target_scores,
        inputs=feature_volume,
        grad_outputs=torch.ones_like(target_scores)
    )[0]  # Shape: (B, C_f, H_f, W_f)

    # Step 3: Compute the weighted combination of feature maps.
    # THIS IS THE KEY DIFFERENCE from Seg-Grad CAM.
    # We perform element-wise multiplication of features and gradients,
    # then sum along the channel dimension.
    # Formula: L_Seg-HiRes-GradCAM = sum_k (A^k ⊙ ∂(y_c,new)/∂(A^k))
    cam = (feature_volume * grads).sum(dim=1)  # Shape: (B, H_f, W_f)
    
    # An equivalent and explicit way using einsum:
    # cam = torch.einsum('bchw,bchw->bhw', feature_volume, grads)

    # Step 4: Apply ReLU to keep only the features that have a positive
    # influence on the class of interest.
    cam = torch.relu(cam)

    return cam
