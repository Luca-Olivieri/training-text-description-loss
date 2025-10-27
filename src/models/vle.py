from core.config import *
from core.registry import Registry
from core.torch_utils import unprefix_state_dict

import torch
from torch import nn
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from abc import abstractmethod, ABC
import math
from PIL import Image
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from functools import partial

# FLAIR
from vendors.flair.src import flair
from open_clip.tokenizer import SimpleTokenizer
from vendors.flair.src.flair.loss import FlairLoss

# FG-CLIP
from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM

from typing import Optional, Literal
from vendors.flair.src.flair.model import FLAIR
from enum import Enum

from core._types import deprecated, override, Callable

class MapComputeMode(Enum):
    """
    Enumeration for different map computation modes.
    
    Attributes:
        SIMILARITY: Compute similarity maps using cosine similarity between tokens
        ATTENTION: Compute attention maps from the visual projection layer
    """
    SIMILARITY = 'similarity'
    ATTENTION = 'attention'
    MAX_TEXT_TOKEN_SIM = 'max_text_token_sim'
    AVG_TEXT_TOKEN_SIM = 'avg_text_token_sim'
    MAX_TEXT_TOKEN_ATTN = 'max_text_token_attn'
    AVG_TEXT_TOKEN_ATTN = 'avg_text_token_attn'

@deprecated("Image and text about is not split.")
@dataclass
class VLEncoderOutput:
    """
    Output of VL encoder containing global and local tokens for both image and text.
    
    Args:
        global_image_token: Global image token [B_i, D]
        global_text_token: Global text token [B_t, 1, D] or [B_i, B_t, D] when broadcasted
        local_image_tokens: Local image tokens [B_i, n_i, D]
        local_text_tokens: Local text tokens [B_t, n_t, D]
        
    """
    global_image_token: Optional[torch.Tensor] = None   # [B_i, D]
    global_text_token: Optional[torch.Tensor] = None    # [B_i, B_t, D]
    local_image_tokens: Optional[torch.Tensor] = None   # [B_i, n_i, D]
    local_text_tokens: Optional[torch.Tensor] = None    # [B_t, n_t, D]

@dataclass
class VLEImageOutput:
    global_image_token: torch.Tensor # (B_i, D)
    local_image_tokens: torch.Tensor # (B_i, n_i, D)
    

@dataclass
class VLETextOutput:
    global_text_token: torch.Tensor # (B_i, D)
    local_text_tokens: torch.Tensor # (B_t, n_t, D)


@dataclass
class VLEPoolOutput:
    pooled_image_token: torch.Tensor # (B_i, D) or (B_i, B_t, D) if broadcast
    attn_maps_flat: torch.Tensor # (B_i, H, 1, n_i+1) or (B_i, H, B_t, n_i+1) if broadcast


@deprecated("Pool output is return separately from the main VLE output.")
@dataclass
class VLEncoderOutputWithAttn(VLEncoderOutput):
    """
    Extended VL encoder output that includes attention maps.
    
    Attributes:
        attn_maps_flat: Flattened attention maps with shape [B_i, B_t, n_i+1] where the +1
            accounts for the CLS token. Inherits all attributes from VLEncoderOutput.
    """
    attn_maps_flat: Optional[torch.Tensor] = None # [B_i, B_t, n_i+1]


class VLEncoder(ABC):
    """
    Abstract base class for Vision-Language Encoders.
    
    This class defines the interface that all VLE implementations must follow,
    including methods for preprocessing, encoding, computing similarities and maps,
    and training utilities.
    """
    
    def __init__(self) -> None:
        """Initialize the vision-language encoder."""
        self.viz_attn_heads_idx: Optional[list[int]] = None

    def load_model_state_dict(
            self,
            checkpoint_path: Path
    ) -> None:
        # TODO check if map_location='cpu' can improve the vRAM usage somewhere else in the codebase.
        if checkpoint_path.exists():
            self.model.load_state_dict(unprefix_state_dict(torch.load(checkpoint_path, map_location='cpu')['model_state_dict'], prefix='_orig_mod.'))
        else:
            raise ValueError(f"VLE weights path '{checkpoint_path}' not found.")
    
    @abstractmethod
    @torch.inference_mode()
    def preprocess_images(
            self,
            images: torch.Tensor | list[torch.Tensor],
            **kwargs
    ) -> torch.Tensor:
        """
        Preprocess images for the vision encoder.
        
        Args:
            images: Input images as a tensor or list of tensors
            **kwargs: Additional preprocessing arguments
            
        Returns:
            Preprocessed image tensor ready for encoding
        """
        raise NotImplementedError
    
    @abstractmethod
    @torch.inference_mode()
    def preprocess_texts(
            self,
            texts: list[str],
            **kwargs
    ) -> torch.Tensor:
        """
        Tokenize and preprocess text for the text encoder.
        
        Args:
            texts: List of text strings to preprocess
            **kwargs: Additional preprocessing arguments
            
        Returns:
            Tokenized text tensor ready for encoding
        """
        raise NotImplementedError

    @abstractmethod
    def encode_and_project_images(
            self,
            images: torch.Tensor,
            **kwargs
    ) -> VLEImageOutput:
        """
        Encode and project images into a shared embedding space.
        
        Args:
            images: Preprocessed image tensor
            **kwargs: Additional encoding arguments
            
        Returns:
            VLEncoderOutput containing global and local tokens for images.
        """
        raise NotImplementedError
    
    @abstractmethod
    def encode_and_project_texts(
            self,
            texts: torch.Tensor,
            **kwargs
    ) -> VLETextOutput:
        """
        Encode and project texts into a shared embedding space.
        
        Args:
            texts: Preprocessed text tensor
            **kwargs: Additional encoding arguments
            
        Returns:
            VLEncoderOutput containing global and local tokens for texts.
        """
        raise NotImplementedError
    
    # TODO separate pool from this and see if it makes sense to split this method into image and text versions.
    @deprecated("Separated into multiple functions.")
    @abstractmethod
    def encode_and_project(
            self,
            images: Optional[torch.Tensor],
            texts: Optional[torch.Tensor],
            broadcast: bool = False,
            **kwargs
    ) -> VLEncoderOutput:
        """
        Encode and project images and/or texts into a shared embedding space.
        
        Args:
            images: Preprocessed image tensor or None
            texts: Preprocessed text tensor or None
            broadcast: If True, compute all pairwise image-text combinations
            **kwargs: Additional encoding arguments
            
        Returns:
            VLEncoderOutput containing global and local tokens for images and texts
        """
        raise NotImplementedError

    @abstractmethod
    @torch.inference_mode()
    def get_similarity(
            self,
            image_output: VLEImageOutput,
            text_output: VLETextOutput,
            broadcast: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError
    
    @torch.inference_mode()
    def get_maps(
            self,
            image_output: VLEImageOutput,
            text_output: VLETextOutput,
            map_compute_mode: MapComputeMode = MapComputeMode.SIMILARITY,
            upsample_size: Optional[int | tuple[int]] = None,
            upsample_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
            broadcast: bool = False,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute spatial maps (similarity or attention) between images and texts.
        
        Args:
            images: Preprocessed image tensor
            texts: Preprocessed text tensor
            map_compute_mode: Mode for computing maps (SIMILARITY or ATTENTION)
            upsample_size: Target size for upsampling maps, or None to keep original size
            upsample_mode: Interpolation mode for upsampling
            broadcast: If True, compute maps for all image-text pairs
            **kwargs: Additional arguments passed to specific map computation methods
            
        Returns:
            Tuple of (maps, min_value, max_value) where maps has shape [B_i, B_t, H, W]
            
        Raises:
            ValueError: If map_compute_mode is not recognized
        """
        match map_compute_mode:
            case MapComputeMode.SIMILARITY:
                return self.get_sim_maps(image_output, text_output, upsample_size=upsample_size, upsample_mode=upsample_mode, broadcast=broadcast, **kwargs)
            case MapComputeMode.ATTENTION:
                return self.get_attn_maps(image_output, text_output, upsample_size=upsample_size, upsample_mode=upsample_mode, broadcast=broadcast, **kwargs)
            case MapComputeMode.MAX_TEXT_TOKEN_SIM:
                return self.get_aggr_text_token_sim_maps(image_output, text_output, lambda x: ((r:=torch.max(x, dim=-1)).values, r.indices), upsample_size=upsample_size, upsample_mode=upsample_mode, broadcast=broadcast)
            case MapComputeMode.AVG_TEXT_TOKEN_SIM:
                return self.get_aggr_text_token_sim_maps(image_output, text_output, lambda x: (torch.mean(x, dim=-1), None), upsample_size=upsample_size, upsample_mode=upsample_mode, broadcast=broadcast)
            case MapComputeMode.MAX_TEXT_TOKEN_ATTN:
                return self.get_aggr_text_token_attn_maps(image_output, text_output, lambda x: ((r:=torch.max(x, dim=-2)).values, r.indices), upsample_size=upsample_size, upsample_mode=upsample_mode, broadcast=broadcast)
            case MapComputeMode.AVG_TEXT_TOKEN_ATTN:
                return self.get_aggr_text_token_attn_maps(image_output, text_output, lambda x: (torch.mean(x, dim=-2), None), upsample_size=upsample_size, upsample_mode=upsample_mode, broadcast=broadcast)
            case _:
                raise ValueError(f"Unknown mode: {map_compute_mode}. The only supported types are {[c.name for c in MapComputeMode]}.")
    
    @torch.inference_mode()
    def get_sim_maps(
            self,
            image_output: VLEImageOutput,
            text_output: VLETextOutput,
            upsample_size: Optional[int | tuple[int]] = None,
            upsample_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
            broadcast: bool = False,
    ) -> tuple[torch.Tensor, None]:
        # normalize token embds for cosine similarity
        local_image_features = F.normalize(image_output.local_image_tokens, dim=-1)  # (B_i, n_i, D)
        global_text_features = F.normalize(text_output.global_text_token, dim=-1) # (B_t, D)

        n_i = local_image_features.shape[1]
        l_i = int(math.sqrt(n_i)) # number of patches per axis

        if broadcast:
            sim_maps_flat = torch.einsum('ind,td->itn', local_image_features, global_text_features) # (B_i, B_t, n_i)
        else:
            sim_maps_flat = torch.einsum('ind,id->in', local_image_features, global_text_features) # (B_i, n_i)

        # arrange the flattened patches into a grid
        sim_maps = sim_maps_flat.view(*sim_maps_flat.shape[:-1], l_i, l_i) # [B_i, l_i, l_i] or [B_i, B_t, l_i, l_i] if broadcast
        
        # upsampling
        if upsample_size:
            sim_maps = TF.resize(sim_maps, size=upsample_size, interpolation=upsample_mode) # [B_i, H, W] or [B_i, B_t, H, W] if broadcast
        
        return sim_maps, None
    
    @torch.inference_mode()
    def get_attn_maps(
            self,
            image_output: VLEImageOutput,
            text_output: VLETextOutput,
            upsample_size: Optional[int | tuple[int]] = None,
            upsample_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
            broadcast: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError
    
    @torch.inference_mode()
    def get_aggr_text_token_sim_maps(
            self,
            image_output: VLEImageOutput,
            text_output: VLETextOutput,
            aggr_fn: Callable[[torch.Tensor], tuple[torch.Tensor, Optional[torch.Tensor]]],
            upsample_size: Optional[int | tuple[int]] = None,
            upsample_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
            broadcast: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # normalize token embds for cosine similarity
        local_image_features = F.normalize(image_output.local_image_tokens, dim=-1)  # (B_i, n_i, D)
        local_text_tokens = F.normalize(text_output.local_text_tokens, dim=-1) # (B_t, n_t, D)

        n_i = local_image_features.shape[1]
        l_i = int(math.sqrt(n_i)) # number of patches per axis

        if broadcast:
            sim_maps_flat_per_txt_token = torch.einsum('inf,tmf->itnm', local_image_features, local_text_tokens) # (B_i, B_t, n_i, n_t)
        else:
            sim_maps_flat_per_txt_token = torch.einsum('inf,imf->inm', local_image_features, local_text_tokens) # (B_i, n_i, n_t)
        sim_maps_flat, indices = aggr_fn(sim_maps_flat_per_txt_token) # aggregate the maximum along the text tokens
        
        # NOTE 'n_i' should always be kept as final dimension for the subsequent operations

        # arrange the flattened patches into a grid
        sim_maps = sim_maps_flat.view(*sim_maps_flat.shape[:-1], l_i, l_i) # [B_i, l_i, l_i] or [B_i, B_t, l_i, l_i] if broadcast
        
        # upsampling
        if upsample_size:
            sim_maps = TF.resize(sim_maps, size=upsample_size, interpolation=upsample_mode) # [B_i, H, W] or [B_i, B_t, H, W] if broadcast
        
        return sim_maps, indices
    
    @torch.inference_mode()
    def get_aggr_text_token_attn_maps(
            self,
            image_output: VLEImageOutput,
            text_output: VLETextOutput,
            aggr_fn: Callable[[torch.Tensor], tuple[torch.Tensor, Optional[torch.Tensor]]],
            upsample_size: Optional[int | tuple[int]] = None,
            upsample_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
            broadcast: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError
    
    def set_trainable_params(self, **kwargs) -> None:
        """
        Configure which parameters of the model should be trainable.
        
        Args:
            *arg: Positional arguments for parameter selection
            **kwargs: Keyword arguments for parameter selection
        """
        raise NotImplementedError
    
    def create_loss(self, **kwargs) -> nn.Module:
        """
        Create and return the loss function for training this encoder.
        
        Args:
            *args: Positional arguments for loss creation
            **kwargs: Keyword arguments for loss creation
            
        Returns:
            Loss module appropriate for this encoder
        """
        raise NotImplementedError

    # TODO this method should not belong here, since the loss can vary. This should be in the train-script-specific
    def evaluate(
            self,
            dl: DataLoader,
            criterion: nn.modules.loss._Loss,
    ) -> torch.Tensor:
        """
        Evaluate the model on a dataset.
        
        Args:
            dl: DataLoader providing (images, texts) batches
            criterion: Loss function to compute evaluation loss
            
        Returns:
            Average loss across the entire dataset
        """
        running_loss = 0.0
        running_supcount = 0

        self.model.eval()

        with torch.no_grad():
            for step, (images, texts) in enumerate(dl):

                # vle_output = self.encode_and_project(images, texts, broadcast=False)
                img_output = self.encode_and_project_images(images)
                txt_output = self.encode_and_project_texts(texts)

                batch_losses = criterion(
                        image_features=img_output.global_image_token,
                        image_tokens=img_output.local_image_tokens.clone(),
                        text_features=txt_output.global_text_token.squeeze(1),
                        logit_scale=self.model.logit_scale.exp(),
                        visual_proj=self.model.visual_proj,
                        logit_bias=self.model.logit_bias,
                        output_dict=True
                )
                total_batch_loss = sum(batch_losses.values()) # in our case, we only have losses has only the 'constrastive loss' key.

                with torch.no_grad():
                    self.model.logit_scale.clamp_(0, math.log(100))

                running_loss += total_batch_loss.item() * images.size(0)
                running_supcount += images.size(0)
        
        loss = running_loss / running_supcount
        
        return loss
    
    @abstractmethod
    def count_tokens(self, *args, **kwargs) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            *args: Positional arguments (typically the text string)
            **kwargs: Keyword arguments
            
        Returns:
            Number of tokens after tokenization
        """
        raise NotImplementedError

    @torch.inference_mode()
    def decode_tokens(
            self,
            token_ids: torch.Tensor
    ) -> np.ndarray[str]:
        raise NotImplementedError

VLE_REGISTRY = Registry[VLEncoder]()

class _SupportsPooling(ABC):
    @abstractmethod
    def pool(
        self,
        image_output: VLEImageOutput,
        text_output: VLETextOutput,
        broadcast: bool = False,
        **kwargs
    ) -> VLEPoolOutput:
        raise NotImplementedError
    
    @abstractmethod
    def pool_by_local_text_tokens(
            self,
            image_output: VLEImageOutput,
            text_output: VLETextOutput,
            broadcast: bool = False
    ) -> VLEPoolOutput:
        raise NotImplementedError

class NewLayer(Enum):
    """
    Enumeration of new adapter layers that can be added to FLAIR.
    
    Attributes:
        VISION_ADAPTER: Adapter layer applied after vision encoding
        TEXT_ADAPTER: Adapter layer applied after text encoding
        CONCAT_ADAPTER: Adapter layer applied to concatenated features
    """
    VISION_ADAPTER = 'vision_adapter'
    TEXT_ADAPTER = 'text_adapter'
    CONCAT_ADAPTER = 'concat_adapter'

class OldFLAIRLayer(Enum):
    """
    Enumeration of existing FLAIR layers that can be fine-tuned.
    
    Attributes:
        IMAGE_POST: Post-processing layer for image embeddings
        TEXT_POST: Post-processing layer for text embeddings
        VISUAL_PROJ: Visual projection layer for cross-attention pooling
    """
    IMAGE_POST = 'image_post'
    TEXT_POST = 'text_post'
    VISUAL_PROJ = 'visual_proj'

@deprecated("No longer used since pooling is implemented in a separate method.")
@dataclass
class FLAIROutput(VLEncoderOutputWithAttn):
    """
    Extended output specific to FLAIR encoder.
    
    Attributes:
        local_image_features: Image features after attention pooling with text queries,
            shape [B_i, B_t, D]. Inherits all attributes from VLEncoderOutputWithAttn.
    """
    local_image_features: Optional[torch.Tensor] = None # [B_i, B_t, D]

@VLE_REGISTRY.register("flair")
class FLAIRAdapter(VLEncoder, _SupportsPooling):
    """
    Adapter for the FLAIR (Fine-grained Late-interaction Representation) model.
    
    FLAIR is a vision-language model that uses cross-attention pooling to condition
    text representations on image patches, enabling fine-grained spatial understanding.
    This adapter supports adding custom adapter layers and selective fine-tuning.
    
    Attributes:
        device: Device where the model runs
        version: FLAIR checkpoint version
        model: The underlying FLAIR model
        new_layers: List of new adapter layers added to the model
        preprocess_fn: Image preprocessing transform
        tokenizer: Text tokenizer
        context_length: Maximum token context length
    """
    
    def __init__(
            self,
            version: Literal['flair-cc3m-recap.pt',
                             'flair-cc12m-recap.pt',
                             'flair-yfcc15m-recap.pt',
                             'flair-merged30m.pt'],
            pretrained_weights_root_path: Path,
            new_layers: list[NewLayer],
            device: torch.device,
            viz_attn_heads_idx: Optional[list[int]] = [0, 3, 5, 7] # recommended by the authors
    ) -> None:
        """
        Initialize FLAIR adapter.
        
        Args:
            version: FLAIR checkpoint to load
            pretrained_weights_root_path: Path to cache pretrained weights
            new_layers: List of new adapter layers to add to the model
            device: Device to load the model on
        """
        super().__init__()

        self.device = device
        self.version = version

        # Model
        model, _, preprocess_fn = flair.create_model_and_transforms(
            model_name='ViT-B-16-FLAIR',
            pretrained=hf_hub_download(repo_id='xiaorui638/flair', filename=version, cache_dir=pretrained_weights_root_path),
            device=self.device
        )
        
        self.model: FLAIR = model
        self.new_layers = new_layers
        self.add_new_layers()
        self.init_new_layers_weights()
        self.model.requires_grad_(False) # freeze parameters
        self.model.eval()

        # Preprocess
        # NOTE the following instruction remove some preprocessing functions that are for PIL images, but we only work with tensors.
        preprocess_fn.transforms.pop(2) # remove _convert_to_rgb()
        preprocess_fn.transforms.pop(2) # remove ToTensor()
        self.preprocess_fn: T.Compose = preprocess_fn

        # Tokenizer
        self.tokenizer: SimpleTokenizer = flair.get_tokenizer('ViT-B-16-FLAIR')
        self.context_length = self.tokenizer.context_length

        self.viz_attn_heads_idx = viz_attn_heads_idx

        self.patch_size = self.model.visual.patch_size

    def add_new_layers(self) -> None:
        """
        Add new adapter layers to the FLAIR model.
        
        Creates linear adapter layers (without bias or activation) and registers them
        as submodules of the model.
        """
        # NOTE the authors do not use a bias nor an activation function, I won't use them either.
        if NewLayer.VISION_ADAPTER in self.new_layers:
            self.model.vision_adapter = nn.Linear(512, 512, bias=False, device=self.device)
        if NewLayer.TEXT_ADAPTER in self.new_layers:
            self.model.text_adapter = nn.Linear(512, 512, bias=False, device=self.device)
        if NewLayer.CONCAT_ADAPTER in self.new_layers:
            self.model.concat_adapter = nn.Linear(1024, 512, bias=False, device=self.device)
    
    def init_new_layers_weights(self) -> None:
        """
        Initialize weights of newly added adapter layers.
        
        Uses Xavier uniform initialization for weights and zeros for biases.
        """
        # NOTE for all linear layers, I use Xavier Uniform weight initialisation and bias to 0
        for new_l in self.new_layers:
            module = self.model.get_submodule(new_l.value)
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.inference_mode()
    def preprocess_images(
            self, 
            images: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(images, list):
            images = torch.stack(images, dim=0)
        images = images.to(self.device)
        if images.dtype == torch.uint8:
            images = images/255.
        imgs_tensor = self.preprocess_fn(images) # [B, 3, H_vle, W_vle]
        return imgs_tensor
    
    @torch.inference_mode()
    def preprocess_texts(
            self, 
            texts: list[str],
    ) -> torch.Tensor:
        """
        Tokenize texts for FLAIR encoder.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tokenized text tensor of shape [B, context_length]
        """
        texts_tensor = self.tokenizer(texts).to(self.device) # [B, context_length]
        return texts_tensor

    @override
    def encode_and_project_images(
            self,
            images: torch.Tensor,
    ) -> VLEImageOutput:
        """
        Encode and project images into a shared embedding space.
        
        Args:
            images: Preprocessed image tensor
            
        Returns:
            VLEImageOutput containing global and local tokens for images.
        """
        # get the raw embeddings from the encoders
        global_image_token: torch.Tensor
        local_image_tokens: torch.Tensor
        global_image_token, local_image_tokens = self.model.encode_image(images) # (B_i, d_i), (B_i, n_i, d_i)
        # concatenate tokens for parallelism
        concat_image_tokens = torch.cat([global_image_token.unsqueeze(1), local_image_tokens], dim=1) # (B_i, n_i+1, d_i)
        # project the raw embeddings into the same space
        concat_image_tokens: torch.Tensor = self.model.image_post(concat_image_tokens) # (B_i, n_i+1, D)
        # apply vision adapter
        if NewLayer.VISION_ADAPTER in self.new_layers:
            concat_image_tokens: torch.Tensor = self.model.vision_adapter(concat_image_tokens) # (B_i, n_i+1, D)
        # split the concatenated tokens into the global and text tokens
        global_image_token, local_image_tokens = concat_image_tokens[:, 0], concat_image_tokens[:, 1:] # (B_i, D), (B_i, n_i, D)

        return VLEImageOutput(global_image_token, local_image_tokens)
    
    @override
    def encode_and_project_texts(
            self,
            texts: torch.Tensor,
    ) -> VLETextOutput:
        """
        Encode and project texts into a shared embedding space.
        
        Args:
            texts: Preprocessed text tensor
            
        Returns:
            VLETextOutput containing global and local tokens for texts.
        """
        # get the raw embeddings from the encoders
        global_text_token: torch.Tensor
        local_text_tokens: torch.Tensor
        global_text_token, local_text_tokens = self.model.encode_text(texts) # (B_t, d_t), (B_t, n_t, d_t)
        # concatenate tokens for parallelism
        concat_text_tokens = torch.cat([global_text_token.unsqueeze(1), local_text_tokens], dim=1) # (B_t, n_t+1, d_t)
        # project the raw embeddings into the same space
        concat_text_tokens: torch.Tensor = self.model.text_post(concat_text_tokens) # (B_t, n_t+1, D)
        # apply text adapter
        if NewLayer.TEXT_ADAPTER in self.new_layers:
            concat_text_tokens = self.model.text_adapter(concat_text_tokens) # (B_t, n_t+1, D)
        # split the concatenated tokens into the global and text tokens
        global_text_token, local_text_tokens = concat_text_tokens[:, 0], concat_text_tokens[:, 1:] # (B_t, D), (B_t, n_t, D)

        return VLETextOutput(global_text_token, local_text_tokens)
    
    @override
    def pool(
            self,
            image_output: VLEImageOutput,
            text_output: VLETextOutput,
            broadcast: bool = False
    ) -> VLEPoolOutput:
        local_image_tokens = image_output.local_image_tokens # (B_i, n_i, D)
        global_text_token = text_output.global_text_token # (B_t, D)

        global_text_token = self._arrange_global_text_token_for_pooling(
            global_text_token,
            local_image_tokens,
            broadcast=broadcast
        ) # (B_i, 1, D) or (B_i, B_t, D) if broadcast

        # perform the attention pooling: condition the 'global_text_token' (Q) on the 'local_image_tokens' (K and V)
        # NOTE: the +1 is there for the added 'cls' token
        pooled_image_token, attn_maps_flat = self.model.visual_proj(
            global_text_token, # (B_i, 1, D) or (B_i, B_t, D) if broadcast
            local_image_tokens, # (B_i, n_i, D)
            local_image_tokens, # (B_i, n_i, D)
            output_attn_weights=True,
            average_attn_weights=False
        )
        pooled_image_token: torch.Tensor # (B_i, 1, D) or (B_i, B_t, D) if broadcast
        attn_maps_flat: torch.Tensor # (B_i, H, 1, n_i+1) or (B_i, H, B_t, n_i+1) if broadcast
        
        if broadcast:
            pooled_image_token: torch.Tensor # (B_i, B_t, D)
            attn_maps_flat: torch.Tensor # (B_i, H, B_t, n_i+1)
        else:
            pooled_image_token = pooled_image_token.squeeze(-2) # (B_i, D)
            attn_maps_flat = attn_maps_flat.squeeze(-2) # (B_i, H, n_i+1)

        return VLEPoolOutput(pooled_image_token, attn_maps_flat)

    def _arrange_global_text_token_for_pooling(
            self,
            global_text_token: torch.Tensor,
            local_image_tokens: torch.Tensor,
            broadcast: bool = False
    ) -> torch.Tensor:
        # global_text_token is (B_t, D)
        # local_image_tokens is (B_i, n_i, D)
        if len(global_text_token.shape) != 2:
            raise ValueError(f"'global_text_token' should be of (B_t, D), got {global_text_token.shape}.")
        if len(local_image_tokens.shape) != 3:
            raise ValueError(f"'local_image_tokens' should be of (B_i, n_i, D), got {local_image_tokens.shape}.")
        
        B_i = local_image_tokens.shape[0]
        B_t = global_text_token.shape[0]

        if broadcast:
            global_text_token = global_text_token.unsqueeze(0).expand(B_i, -1, -1) # (B_i, B_t, D)
        else:
            if B_i != B_t:
                raise ValueError(f"If not broadcast, image and text tokens should be in the same number, got {B_i=} and {B_t=}.")
            global_text_token = global_text_token.unsqueeze(1) # (B_i, 1, D)

        return global_text_token
    
    def pool_by_local_text_tokens(
            self,
            image_output: VLEImageOutput,
            text_output: VLETextOutput,
            broadcast: bool = False
    ) -> VLEPoolOutput:

        if broadcast:
            raise NotImplementedError("Broadcasting has not been implemented for this method.")
        local_image_tokens = image_output.local_image_tokens # (B_i, n_i, D)
        local_text_tokens = text_output.local_text_tokens # (B_t, n_t, D)

        local_text_tokens = self._arrange_local_text_tokens_for_pooling(
            local_text_tokens,
            local_image_tokens,
            broadcast=broadcast
        ) # (B_i, n_t, D)

        # perform the attention pooling: condition the 'local_text_tokens' (Q) on the 'local_image_tokens' (K and V)
        # NOTE: the +1 is there for the added 'cls' token
        pooled_image_token, attn_maps_flat = self.model.visual_proj(
            local_text_tokens, # (B_i, n_t, D)
            local_image_tokens, # (B_i, n_i, D)
            local_image_tokens, # (B_i, n_i, D)
            output_attn_weights=True,
            average_attn_weights=False
        )
        pooled_image_token: torch.Tensor # (B_i, n_t, D)
        attn_maps_flat: torch.Tensor # (B_i, H, n_t, n_i+1)

        return VLEPoolOutput(pooled_image_token, attn_maps_flat)
    
    def _arrange_local_text_tokens_for_pooling(
            self,
            local_text_tokens: torch.Tensor,
            local_image_tokens: torch.Tensor,
            broadcast: bool = False
    ) -> torch.Tensor:
        # local_text_tokens is (B_t, n_t, D)
        # local_image_tokens is (B_i, n_i, D)
        if len(local_text_tokens.shape) != 3:
            raise ValueError(f"'local_text_tokens' should be of (B_t, n_t, D), got {local_text_tokens.shape}.")
        if len(local_image_tokens.shape) != 3:
            raise ValueError(f"'local_image_tokens' should be of (B_i, n_i, D), got {local_image_tokens.shape}.")
        
        B_i = local_image_tokens.shape[0]
        B_t = local_text_tokens.shape[0]

        if broadcast:
            raise NotImplementedError("Broadcasting has not been implemented for this method.")
            # local_text_tokens = local_text_tokens.unsqueeze(0).expand(B_i, -1, -1, -1) # (B_i, B_t, n_t, D)
        else:
            if B_i != B_t:
                raise ValueError(f"If not broadcast, image and text tokens should be in the same number, got {B_i=} and {B_t=}.")

        return local_text_tokens
    
    @deprecated("Encode and project of images and texts and pooling is done separately by ad-hoc methods.")
    def encode_and_project(
            self,
            images: Optional[torch.Tensor],
            texts: Optional[torch.Tensor],
            broadcast: bool = False,
            pool: bool = True
    ) -> FLAIROutput:
        """
        Encode and project images and texts through FLAIR.
        
        Encodes inputs through vision/text encoders, applies projection layers and
        optional adapter layers, then performs cross-attention pooling if requested.
        
        Args:
            images: Preprocessed image tensor or None
            texts: Preprocessed text tensor or None
            broadcast: If True, compute all pairwise image-text combinations
            pool: If True, perform attention pooling to get local_image_features
            
        Returns:
            FLAIROutput containing all encoded tokens and optionally attention maps
        """
        
        flair_output = FLAIROutput()
        
        if images is not None:
            # get the raw embeddings from the encoders
            global_image_token, local_image_tokens = self.model.encode_image(images)    # [B_i, d_i], [B_i, n_i, d_i]
            # project the raw embeddings into the same space
            global_image_token: torch.Tensor = self.model.image_post(global_image_token)    # [B_i, D]
            local_image_tokens: torch.Tensor = self.model.image_post(local_image_tokens)    # [B_i, n_i, D]
            # apply vision adapter
            if NewLayer.VISION_ADAPTER in self.new_layers:
                global_image_token = self.model.vision_adapter(global_image_token) # [B_i, D]
                local_image_tokens = self.model.vision_adapter(local_image_tokens) # [B_i, n_i, D]
            
            flair_output.global_image_token = global_image_token        # [B_i, D]
            flair_output.local_image_tokens = local_image_tokens  # [B_i, n_i, D]

        if texts is not None:
            # get the raw embeddings from the encoders
            global_text_token, local_text_tokens = self.model.encode_text(texts)       # [B_t, d_t], [B_t, n_t, d_t]
            # project the raw embeddings into the same space
            global_text_token: torch.Tensor = self.model.text_post(global_text_token)       # [B_t, D]
            local_text_tokens: torch.Tensor = self.model.text_post(local_text_tokens)       # [B_t, n_t, D]
            # apply text adapter
            if NewLayer.TEXT_ADAPTER in self.new_layers:
                global_text_token = self.model.text_adapter(global_text_token) # [B_t, D]
                local_text_tokens = self.model.text_adapter(local_text_tokens) # [B_t, n_t, D]

            flair_output.global_text_token = global_text_token # [B_t, D]
            flair_output.local_text_tokens = local_text_tokens # [B_t, n_t, D]

        if (images is not None) and (texts is not None):

            image_batch_size = local_image_tokens.shape[0] # B_i, n_i

            if broadcast:
                # adapt text token for broadcasting
                global_text_token = global_text_token.unsqueeze(0).expand(image_batch_size, -1, -1) # [B_i, B_t, D]
            else:
                # if images.shape[0] != texts.shape[0]:
                    # raise AttributeError(f"When not broadcasting, 'images' and 'texts' should contain the same number of elements, but got {images.shape[0]=} and {texts.shape[0]=} instead.")
                # from now on, B_t = 1
                global_text_token = global_text_token.unsqueeze(1) # [B_i, 1, D]
            
            if pool:
                # perform the attention pooling: condition the 'global_text_token' (Q) on the 'local_image_tokens' (K and V)
                local_image_features, attn_maps_flat = self.model.visual_proj(
                    global_text_token,      # [B_i, B_t, D]
                    local_image_tokens,     # [B_i, n_i, D]
                    local_image_tokens,     # [B_i, n_i, D]
                    output_attn_weights=True,
                    average_attn_weights=False
                ) # [B_i, B_t, D], [B_i, B_t, n_i+1] (the +1 is there for the added 'cls' token)
                
                flair_output.local_image_features = local_image_features    # [B_i, B_t, D]
                flair_output.attn_maps_flat = attn_maps_flat                # [B_i, B_t, n_i+1]

            flair_output.global_text_token = global_text_token          # [B_i, B_t, D]
        
        return flair_output
    
    @torch.inference_mode()
    def get_similarity(
            self,
            image_output: VLEImageOutput,
            text_output: VLETextOutput,
            broadcast: bool = False
    ) -> torch.Tensor:
        """
        Compute similarity scores between images and texts using FLAIR.
        
        Uses normalized pooled_image_token (after attention pooling) and global_text_token
        to compute cosine similarity.
        
        Args:
            images: Preprocessed image tensor
            texts: Preprocessed text tensor
            broadcast: If True, compute similarity for all image-text pairs
            
        Returns:
            Similarity scores. Shape [B_i, B_t] if broadcast=True, else [B]
        """
        pool_output = self.pool(image_output, text_output, broadcast)

        # normalise the features
        image_features = F.normalize(pool_output.pooled_image_token, dim=-1) # (B_i, D) or (B_i, B_t, D) if broadcast
        text_features = F.normalize(text_output.global_text_token, dim=-1) # (B_t, D)
        
        if broadcast:
            image_features: torch.Tensor # (B_i, B_t, D)
            text_features: torch.Tensor # (B_t, D)
            sim = torch.einsum('itd,td->it', image_features, text_features) # (B_i, B_t)
        else:
            image_features: torch.Tensor # (B_i, D)
            text_features: torch.Tensor # (B_i, D)
            sim = torch.einsum('id,id->i', image_features, text_features) # (B_i)

        return sim
    
    @torch.inference_mode()
    def get_attn_maps(
            self,
            image_output: VLEImageOutput,
            text_output: VLETextOutput,
            upsample_size: Optional[int | tuple[int]] = None,
            upsample_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
            broadcast: bool = False,
    ) -> tuple[torch.Tensor, None]:
        """
        Compute attention maps from the visual projection layer.
        
        Encodes images and texts, then extracts attention weights from the cross-attention
        mechanism that conditions text queries on image patches.
        
        Args:
            images: Preprocessed image tensor
            texts: Preprocessed text tensor
            upsample_size: Target size for upsampling attention maps
            upsample_mode: Interpolation mode for upsampling
            broadcast: If True, compute attention for all image-text pairs
            attn_heads_idx: Indices of attention heads to average. If None, average all heads.
            
        Returns:
            Tuple of (attention_maps, min_attn, max_attn) where:
                - attention_maps: Tensor of shape [B_i, B_t, H, W] (or [B_i, B_t, sqrt(n_i), sqrt(n_i)] if not upsampled)
                - min_attn: Minimum attention value across all maps
                - max_attn: Maximum attention value across all maps
        """
        pool_output = self.pool(image_output, text_output, broadcast=broadcast) 
        # remove the <cls> token
        attn_maps_flat = pool_output.attn_maps_flat[..., :-1] # (B_i, H, n_i) or (B_i, H, B_t, n_i) if broadcast

        n_i = attn_maps_flat.shape[-1]
        l_i = int(math.sqrt(n_i)) # number of patches per axis

        if self.viz_attn_heads_idx is None:
            viz_attn_heads_idx = slice(None) # consider all attn heads
        else:
            viz_attn_heads_idx = self.viz_attn_heads_idx # consider specific attn heads
        
        # average the considered attn heads
        attn_maps_flat = attn_maps_flat[:, viz_attn_heads_idx, ...].mean(dim=1) # (B_i, n_i) or (B_i, B_t, n_i) if broadcast

        # arrange the flattened patches into a grid
        attn_maps = attn_maps_flat.view(*attn_maps_flat.shape[:-1], l_i, l_i) # (B_i, l_i, l_i) or (B_i, B_t, l_i, l_i) if broadcast
        
        # upsampling
        if upsample_size:
            attn_maps = TF.resize(attn_maps, size=upsample_size, interpolation=upsample_mode) # (B_i, H, W) or (B_i, B_t, H, W) if broadcast

        return attn_maps, None
    
    @torch.inference_mode()
    def get_aggr_text_token_attn_maps(
            self,
            image_output: VLEImageOutput,
            text_output: VLETextOutput,
            aggr_fn: Callable[[torch.Tensor], tuple[torch.Tensor, Optional[torch.Tensor]]],
            upsample_size: Optional[int | tuple[int]] = None,
            upsample_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
            broadcast: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention maps from the visual projection layer.
        
        Encodes images and texts, then extracts attention weights from the cross-attention
        mechanism that conditions text queries on image patches.
        
        Args:
            images: Preprocessed image tensor
            texts: Preprocessed text tensor
            upsample_size: Target size for upsampling attention maps
            upsample_mode: Interpolation mode for upsampling
            broadcast: If True, compute attention for all image-text pairs
            attn_heads_idx: Indices of attention heads to average. If None, average all heads.
            
        Returns:
            Tuple of (attention_maps, min_attn, max_attn) where:
                - attention_maps: Tensor of shape [B_i, B_t, H, W] (or [B_i, B_t, sqrt(n_i), sqrt(n_i)] if not upsampled)
                - min_attn: Minimum attention value across all maps
                - max_attn: Maximum attention value across all maps
        """
        if broadcast:
            raise NotImplementedError("Broadcasting has not been implemented for this method.")

        pool_output = self.pool_by_local_text_tokens(image_output, text_output, broadcast=broadcast) 
        # remove the <cls> token
        attn_maps_flat_per_txt_token = pool_output.attn_maps_flat[..., :-1] # (B_i, H, n_t, n_i)

        n_i = attn_maps_flat_per_txt_token.shape[-1]
        l_i = int(math.sqrt(n_i)) # number of patches per axis

        if self.viz_attn_heads_idx is None:
            viz_attn_heads_idx = slice(None) # consider all attn heads
        else:
            viz_attn_heads_idx = self.viz_attn_heads_idx # consider specific attn heads
        
        # average the considered attn heads
        attn_maps_flat_per_txt_token = attn_maps_flat_per_txt_token[:, viz_attn_heads_idx, ...].mean(dim=1) # (B_i, n_t, n_i)

        # aggregate the maximum along the text tokens
        attn_maps_flat, indices = aggr_fn(attn_maps_flat_per_txt_token) # (B_i, n_i)

        # arrange the flattened patches into a grid
        attn_maps = attn_maps_flat.view(*attn_maps_flat.shape[:-1], l_i, l_i) # (B_i, l_i, l_i)
        
        # upsampling
        if upsample_size:
            attn_maps = TF.resize(attn_maps, size=upsample_size, interpolation=upsample_mode) # (B_i, H, W)

        return attn_maps, indices
    
    def set_trainable_params(
            self,
            trainable_modules: Optional[list[OldFLAIRLayer | NewLayer]],
    ) -> None:
        """
        Configure which FLAIR modules should be trainable.
        
        Freezes all parameters by default, then enables gradients for specified modules.
        
        Args:
            trainable_modules: List of modules to make trainable, or None to freeze all
        """
        # first, gradients are disabled for all modules.
        self.model.requires_grad_(False)

        # enable the gradients in the selected modules (if any)
        if trainable_modules:
            [self.model.get_submodule(m.value).requires_grad_(True) for m in trainable_modules]
    
    def create_loss(
            self,
            add_mps_loss: bool,
            world_size: int = 1,
            rank: int = 0,
            num_caps_per_img: int = 1
    ) -> FlairLoss:
        """
        Create FLAIR-specific contrastive loss.
        
        Args:
            add_mps_loss: Whether to add the maximum patch similarity loss
            world_size: Number of distributed processes
            rank: Rank of current process in distributed setting
            num_caps_per_img: Number of captions per image in the dataset
            
        Returns:
            Configured FlairLoss module
        """
        flair_loss = FlairLoss(
            rank=rank,
            world_size=world_size,
            num_cap_per_img=num_caps_per_img,
            added_mps_loss=add_mps_loss
        )
        return flair_loss

    @torch.inference_mode()
    def count_tokens(
            self,
            text: str
    ) -> int:
        """
        Count tokens in a text string using FLAIR tokenizer.
        
        Args:
            text: Input text string
            
        Returns:
            Number of tokens including SoT and EoT special tokens
        """
        tokenizer = flair.get_tokenizer('ViT-B-16-FLAIR', context_length=1000) # NOTE If in need, increase it to working upper bound.
        return len(tokenizer.encode(text)) + 2 # 'SoT' and 'EoT' are added.

    @torch.inference_mode()
    def decode_tokens(
            self,
            token_ids: torch.Tensor
    ) -> np.ndarray[str]:
        if len(token_ids.shape) == 1:
            return np.array([self.tokenizer.decode([int(id)]) for id in token_ids])
        elif len(token_ids.shape) == 2:
            return np.stack([np.array([self.tokenizer.decode([int(id)]) for id in row_ids]) for row_ids in token_ids])
        else:
            raise ValueError(f"'token_ids' should be of shape (B, L) or (L), got shape {token_ids.shape}")


class SimSegAdapter(VLEncoder):
    """
    Placeholder for SimSeg adapter (not implemented).
    
    Note: Integration deemed too complex for expected benefits.
    """
    ...
    # NOTE too much integration work likely for unimpressive results, I would skip this.


@VLE_REGISTRY.register("fg-clip")
class FG_CLIPAdapter(VLEncoder):
    """
    Adapter for FG-CLIP (Fine-Grained CLIP) model.
    
    FG-CLIP provides both global and dense (patch-level) image features for
    fine-grained vision-language understanding. Unlike FLAIR, it does not use
    cross-attention pooling.
    
    Attributes:
        image_size: Input image size expected by the model
        model: The underlying FG-CLIP model
        walk_short_pos: Whether to use short position embeddings for text
        context_length: Maximum token context length
        tokenizer: Text tokenizer
        image_processor: Image preprocessor
    """
    
    def __init__(
            self,
            version: Literal[
                'fg-clip-base',
                'fg-clip-large'],
            pretrained_weights_root_path: Path,
            device: torch.device,
            long_captions = True
    ) -> None:
        """
        Initialize FG-CLIP adapter.
        
        Args:
            version: FG-CLIP version to load ('fg-clip-base' or 'fg-clip-large')
            device: Device to load the model on
            long_captions: If True, use context length of 248, else 77
        """
        super().__init__()

        self.device = device

        model_root = f'qihoo360/{version}' # NOTE HF repo ID 

        vers_2_imgsize = {
            'fg-clip-base': 224,
            'fg-clip-large': 336
        }
        self.image_size = vers_2_imgsize[version]
        model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True, cache_dir=pretrained_weights_root_path)
        model.to(self.device)
        model.eval()
        self.model = model
        
        # short or long captions
        if long_captions:
            self.walk_short_pos = False
            self.context_length = 248
        else:
            self.walk_short_pos = True
            self.context_length = 77

        self.tokenizer = AutoTokenizer.from_pretrained(model_root, use_fast=True, model_max_length=self.context_length, cache_dir=pretrained_weights_root_path)
        self.image_processor = AutoImageProcessor.from_pretrained(model_root, use_fast=True, cache_dir=pretrained_weights_root_path)

    @torch.inference_mode()
    def preprocess_images(
            self, 
            images: torch.Tensor | list[torch.Tensor] | list[Image.Image],
    ) -> torch.Tensor:
        if isinstance(images, torch.Tensor) or isinstance(images[0], torch.Tensor):
            images = [to_pil_image(img) for img in images]
        
        images = [img.resize((self.image_size, self.image_size)) for img in images]
        # TODO parallelise this by providing tensors directly

        imgs_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values'].to(self.device) # [1, 3, H_vle, W_vle]

        return imgs_tensor
    
    @torch.inference_mode()
    def preprocess_texts(
            self, 
            texts: list[str],
    ) -> torch.Tensor:
        """
        Tokenize texts for FG-CLIP encoder.
        
        Args:
            texts: List of text strings
            device: Device to move tokenized texts to
            
        Returns:
            Tokenized text tensor of shape [B, context_length]
        """
        texts_tensor = torch.tensor(self.tokenizer(texts, max_length=self.context_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=self.device) # [B, context_length]
        return texts_tensor

    @deprecated("Define separate methods for imgs and texts.")
    def encode_and_project(
            self,
            images: torch.Tensor,
            texts: torch.Tensor,
            broadcast: bool = False
    ) -> VLEncoderOutput:
        """
        Encode and project images and texts through FG-CLIP.
        
        Extracts global image features, dense image features, and global text features.
        All features are L2-normalized.
        
        Args:
            images: Preprocessed image tensor
            texts: Preprocessed text tensor
            broadcast: If True, expand text features for all image-text pairs
            
        Returns:
            VLEncoderOutput containing normalized global and local tokens
            
        Raises:
            AttributeError: If broadcast=False but image and text batch sizes don't match
        """

        image_feature = self.model.get_image_features(images)
        dense_image_feature = self.model.get_image_dense_features(images)
        text_feature = self.model.get_text_features(texts,walk_short_pos=self.walk_short_pos)
        image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True) # [B_i, D]
        dense_image_feature = dense_image_feature / dense_image_feature.norm(p=2, dim=-1, keepdim=True) # [B, n_i, D]
        text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True) # [B_i, D]

        image_batch_size = image_feature.shape[0] # B_i

        if broadcast:
            # adapt text token for broadcasting
            text_feature = text_feature.unsqueeze(0).expand(image_batch_size, -1, -1) # [B_i, B_t, D]
        else:
            if images.shape[0] != texts.shape[0]:
                raise AttributeError(f"When not broadcasting, 'images' and 'texts' should contain the same number of elements, but got {images.shape[0]=} and {texts.shape[0]=} instead.")
            text_feature = text_feature.unsqueeze(1)

        vle_output = VLEncoderOutput(
            global_image_token=image_feature,
            global_text_token=text_feature,
            local_image_tokens=dense_image_feature,
            local_text_tokens=None
        )

        return vle_output
    
    @torch.inference_mode()
    def get_similarity_(
            self,
            images: torch.Tensor,
            texts: torch.Tensor,
            broadcast: bool = False
    ) -> torch.Tensor:
        """
        Compute similarity scores between images and texts using FG-CLIP.
        
        Uses global image and text features (already normalized during encoding).
        
        Args:
            images: Preprocessed image tensor
            texts: Preprocessed text tensor
            broadcast: If True, compute similarity for all image-text pairs
            
        Returns:
            Similarity scores. Shape [B_i, B_t] if broadcast=True, else [B]
        """
        vle_output = self.encode_and_project(images, texts, broadcast)
        if broadcast:
            sim = vle_output.global_image_token @ vle_output.global_text_token.mT # [B_i, B_t, D]
        else:
            text_feature = vle_output.global_text_token.squeeze(1)
            sim = torch.einsum('bf,bf->b', vle_output.global_image_token, text_feature) # # [B, D]
        return sim
    
    @torch.inference_mode()
    def count_tokens(
            self,
            text: str
    ) -> int:
        """
        Count tokens in a text string using FG-CLIP tokenizer.
        
        Args:
            text: Input text string
            
        Returns:
            Number of tokens including special tokens
        """
        return len(self.tokenizer.encode(text)) # 'SoT' and 'EoT' are added.
    
    def get_attn_maps(self, *args, **kwargs) -> None:
        """
        Attention maps are not available for FG-CLIP.
        
        Raises:
            AttributeError: Always, as FG-CLIP does not provide attention maps
        """
        raise AttributeError("The VLE 'FG-CLIP' does not provide attention maps.")

def main() -> None:
    """
    Test function to verify VLE registry and FLAIR adapter initialization.
    
    Prints registered VLE models and tests FLAIR loss creation.
    """
    from typing import cast
    print(VLE_REGISTRY.registered_objects())
    model: VLEncoder = VLE_REGISTRY.get('flair')
    model = cast(FLAIRAdapter, model)
    print(model.create_loss(add_mps_loss=True, rank=0, world_size=1, num_caps_per_img=1))
    
if __name__ == '__main__':
    main()
