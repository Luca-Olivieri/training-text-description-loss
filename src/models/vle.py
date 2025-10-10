from core.config import *
from core.registry import Registry

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
from collections import OrderedDict
from huggingface_hub import hf_hub_download

# FLAIR
from vendors.flair.src import flair
from open_clip.tokenizer import HFTokenizer, SimpleTokenizer
from vendors.flair.src.flair.loss import FlairLoss
from vendors.flair.src.flair.train import backward, unwrap_model

# FG-CLIP
from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM

from typing import Optional, Literal
from vendors.flair.src.flair.model import FLAIR
from enum import Enum

class MapComputeMode(Enum):
    """
    Enumeration for different map computation modes.
    
    Attributes:
        SIMILARITY: Compute similarity maps using cosine similarity between tokens
        ATTENTION: Compute attention maps from the visual projection layer
    """
    SIMILARITY = 'similarity'
    ATTENTION = 'attention'


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
    
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the vision-language encoder."""
        raise NotImplementedError
    
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
            images: torch.Tensor,
            texts: torch.Tensor,
            broadcast: bool = False
    ) -> torch.Tensor:
        """
        Compute similarity scores between images and texts.
        
        Args:
            images: Preprocessed image tensor
            texts: Preprocessed text tensor
            broadcast: If True, compute similarity for all image-text pairs
            
        Returns:
            Similarity scores tensor. Shape [B_i, B_t] if broadcast=True, else [B] where B=B_i=B_t
        """
        raise NotImplementedError
    
    @torch.inference_mode()
    def get_maps(
            self,
            images: torch.Tensor,
            texts: torch.Tensor,
            map_compute_mode: MapComputeMode = MapComputeMode.SIMILARITY,
            upsample_size: Optional[int | tuple[int]] = None,
            upsample_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
            broadcast: bool = False,
            **kwargs
    ) -> torch.Tensor:
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
            case MapComputeMode.ATTENTION:
                return self.get_attn_maps(images, texts, upsample_size=upsample_size, upsample_mode=upsample_mode, broadcast=broadcast, **kwargs)
            case MapComputeMode.SIMILARITY:
                return self.get_sim_maps(images, texts, upsample_size=upsample_size, upsample_mode=upsample_mode, broadcast=broadcast, **kwargs)
            case _:
                raise ValueError(f"Unknown mode: {map_compute_mode}. The only supported types are {[c.name for c in MapComputeMode]}.")
    
    @torch.inference_mode()
    def get_attn_maps(
            self,
            images: torch.Tensor,
            texts: torch.Tensor,
            upsample_size: Optional[int | tuple[int]] = None,
            upsample_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
            broadcast: bool = False,
            attn_heads_idx: Optional[list[int]] = None # [0, 3, 5, 7] are selected by the authors for FLAIR
    ) -> torch.Tensor:
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
                For FLAIR, [0, 3, 5, 7] are recommended by the authors
            
        Returns:
            Tuple of (attention_maps, min_attn, max_attn) where:
                - attention_maps: Tensor of shape [B_i, B_t, H, W] (or [B_i, B_t, sqrt(n_i), sqrt(n_i)] if not upsampled)
                - min_attn: Minimum attention value across all maps
                - max_attn: Maximum attention value across all maps
        """
        # _, [B_i, B_t, D], [B_i, n_i, D], _, _, [B_i, B_t, n_i+1]
        vle_output = self.encode_and_project(images, texts, broadcast)
        # NOTE 'B_t' = 1 if 'broadcast' = False

        if attn_heads_idx:
            vle_output.attn_maps_flat = vle_output.attn_maps_flat[:, attn_heads_idx, ...].mean(dim=1, keepdim=False)
        else:
            vle_output.attn_maps_flat = vle_output.attn_maps_flat.mean(dim=1, keepdim=False)

        text_batch_size = vle_output.global_text_token.shape[1] # B_t
        image_batch_size, image_num_patches = vle_output.local_image_tokens.shape[:2] # B_i, n_i
        
        # reshape attention maps
        num_patches_per_axis = int(math.sqrt(image_num_patches))
        attn_maps: torch.Tensor = vle_output.attn_maps_flat[:, :, :-1].view(image_batch_size, text_batch_size, num_patches_per_axis, num_patches_per_axis) # [B_i, B_t, sqrt(n_i), sqrt(n_i)]
        min_attn, max_attn = attn_maps.min(), attn_maps.max() # max token attention
        
        # upsampling
        if upsample_size:
            attn_maps: torch.Tensor = TF.resize(attn_maps, upsample_size, interpolation=upsample_mode) # [B_i, B_t, H, W]

        return attn_maps, min_attn, max_attn
    
    @torch.inference_mode()
    def get_sim_maps(
            self,
            images: torch.Tensor,
            texts: torch.Tensor,
            upsample_size: Optional[int | tuple[int]] = None,
            upsample_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
            broadcast: bool = False
    ) -> torch.Tensor:
        """
        Compute similarity maps using cosine similarity between text and image tokens.
        
        Computes cosine similarity between global text tokens and local image patch tokens
        to produce spatial similarity maps.
        
        Args:
            images: Preprocessed image tensor
            texts: Preprocessed text tensor
            upsample_size: Target size for upsampling similarity maps
            upsample_mode: Interpolation mode for upsampling
            broadcast: If True, compute similarity for all image-text pairs
            
        Returns:
            Tuple of (similarity_maps, min_sim, max_sim) where:
                - similarity_maps: Tensor of shape [B_i, B_t, H, W] (or [B_i, B_t, sqrt(n_i), sqrt(n_i)] if not upsampled)
                - min_sim: Minimum similarity value across all maps
                - max_sim: Maximum similarity value across all maps
        """
        # _, [B_i, B_t, D], [B_i, n_i, D], _, _, _
        # _, global_text_token, local_image_tokens, _, _, _ = self.encode_and_project(images, texts, broadcast)
        vle_output = self.encode_and_project(images, texts, broadcast)
        text_batch_size = vle_output.global_text_token.shape[1] # B_t
        image_batch_size, image_num_patches = vle_output.local_image_tokens.shape[:2] # B_i, n_i
        
        # normalize token embds for cosine similarity
        norm_global_text_token = F.normalize(vle_output.global_text_token, p=2, dim=-1)    # [B_i, B_t, D]
        norm_local_image_tokens = F.normalize(vle_output.local_image_tokens, p=2, dim=-1)  # [B_i, n_i, D]

        # [B_t, D] @ [n_i, D] --> [B_t, n_i] (in a B_i batch)
        sim_maps = torch.bmm(norm_global_text_token, norm_local_image_tokens.swapaxes(-1, -2)) # batched matrix multiplication
        
        # reshape similarity maps
        num_patches_per_axis = int(math.sqrt(image_num_patches))
        sim_maps = sim_maps.view(image_batch_size, text_batch_size, num_patches_per_axis, num_patches_per_axis) # [B_i, B_t, n_i, n_i]
        min_sim, max_sim = sim_maps.min(), sim_maps.max() # max token similarity
        
        # upsampling
        if upsample_size:
            sim_maps: torch.Tensor = TF.resize(sim_maps, upsample_size, interpolation=upsample_mode) # [B_i, B_t, H, W]
        
        return sim_maps, min_sim, max_sim
    
    def set_trainable_params(self, *arg, **kwargs) -> None:
        """
        Configure which parameters of the model should be trainable.
        
        Args:
            *arg: Positional arguments for parameter selection
            **kwargs: Keyword arguments for parameter selection
        """
        raise NotImplementedError
    
    def create_loss(self, *args, **kwargs) -> nn.Module:
        """
        Create and return the loss function for training this encoder.
        
        Args:
            *args: Positional arguments for loss creation
            **kwargs: Keyword arguments for loss creation
            
        Returns:
            Loss module appropriate for this encoder
        """
        raise NotImplementedError

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

                vle_output = self.encode_and_project(images, texts, broadcast=False)

                batch_losses = criterion(
                        image_features=vle_output.global_image_token,
                        image_tokens=vle_output.local_image_tokens.clone(),
                        text_features=vle_output.global_text_token.squeeze(1),
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

VLE_REGISTRY = Registry[VLEncoder]()

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
class FLAIRAdapter(VLEncoder):
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
            device: torch.device
    ) -> None:
        """
        Initialize FLAIR adapter.
        
        Args:
            version: FLAIR checkpoint to load
            pretrained_weights_root_path: Path to cache pretrained weights
            new_layers: List of new adapter layers to add to the model
            device: Device to load the model on
        """
        self.device = device
        self.version = version

        # Model
        pretrained = hf_hub_download(repo_id='xiaorui638/flair', filename=version, cache_dir=pretrained_weights_root_path)
        model, _, preprocess_fn = flair.create_model_and_transforms('ViT-B-16-FLAIR', pretrained=pretrained, device=self.device)
        
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
            global_text_token, local_text_tokens = self.model.encode_text(texts)        # [B_t, d_t], [B_t, n_t, d_t]
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
            images: torch.Tensor,
            texts: torch.Tensor,
            broadcast: bool = False
    ) -> torch.Tensor:
        """
        Compute similarity scores between images and texts using FLAIR.
        
        Uses normalized local_image_features (after attention pooling) and global_text_token
        to compute cosine similarity.
        
        Args:
            images: Preprocessed image tensor
            texts: Preprocessed text tensor
            broadcast: If True, compute similarity for all image-text pairs
            
        Returns:
            Similarity scores. Shape [B_i, B_t] if broadcast=True, else [B]
        """
        if broadcast:
            flair_output = self.encode_and_project(images, texts, broadcast=True)
            image_features, text_features = F.normalize(flair_output.local_image_features, dim=-1), F.normalize(flair_output.global_text_token, dim=-1)
            # NOTE to me it seems that these squeezes should not be here
            text_features = text_features.squeeze(1)
            image_features = image_features.squeeze(1)
            sim = torch.einsum('bf,bif->bi', image_features, text_features)
        else:
            flair_output = self.encode_and_project(images, texts, broadcast=False)
            image_features, text_features = F.normalize(flair_output.local_image_features, dim=-1), F.normalize(flair_output.global_text_token, dim=-1)
            image_features = image_features.squeeze(1)
            text_features = text_features.squeeze(1)
            sim = torch.einsum('bf,bf->b', image_features, text_features)
        return sim
    
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
        tokenizer = flair.get_tokenizer('ViT-B-16-FLAIR', context_length=1000) # I do not expect to use text longer than 1000 tokens. If so, increase it to working upper bound.
        return len(tokenizer.encode(text)) + 2 # 'SoT' and 'EoT' are added.

    def add_new_layers(self) -> None:
        """
        Add new adapter layers to the FLAIR model.
        
        Creates linear adapter layers (without bias or activation) and registers them
        as submodules of the model.
        """
        # NOTE the authors do not use a bias nor an activation function, I won't use them either.
        if NewLayer.VISION_ADAPTER in self.new_layers:
            vision_adapter = nn.Linear(512, 512, bias=False, device=self.device)
            self.model.add_module('vision_adapter', vision_adapter)
        if NewLayer.TEXT_ADAPTER in self.new_layers:
            text_adapter = nn.Linear(512, 512, bias=False, device=self.device)
            self.model.add_module('text_adapter', text_adapter)
        if NewLayer.CONCAT_ADAPTER in self.new_layers:
            concat_adapter = nn.Linear(1024, 512, bias=False, device=self.device)
            self.model.add_module('concat_adapter', concat_adapter)


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
            checkpoint: Literal['fg-clip-base', 
                                'fg-clip-large'],
            device: torch.device,
            long_captions = True
    ) -> None:
        """
        Initialize FG-CLIP adapter.
        
        Args:
            checkpoint: FG-CLIP checkpoint to load ('fg-clip-base' or 'fg-clip-large')
            device: Device to load the model on
            long_captions: If True, use context length of 248, else 77
        """
        model_root = f'qihoo360/{checkpoint}'
        ckp_2_imgsize = {
            'fg-clip-base': 224,
            'fg-clip-large': 336
        }
        self.image_size = ckp_2_imgsize[checkpoint]
        model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True)
        model.to(device)
        model.eval()
        self.model = model
        
        # short or long captions
        if long_captions:
            self.walk_short_pos = False
            self.context_length = 248
        else:
            self.walk_short_pos = True
            self.context_length = 77

        self.tokenizer = AutoTokenizer.from_pretrained(model_root, use_fast=True, model_max_length=self.context_length)
        self.image_processor = AutoImageProcessor.from_pretrained(model_root, use_fast=True)

    @torch.inference_mode()
    def preprocess_images(
            self, 
            images: torch.Tensor | list[torch.Tensor] | list[Image.Image],
            device: torch.device
    ) -> torch.Tensor:
        if isinstance(images, torch.Tensor) or isinstance(images[0], torch.Tensor):
            images = [to_pil_image(img) for img in images]
        
        images = [img.resize((self.image_size, self.image_size)) for img in images]
        # image = TF.resize(...)

        imgs_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values'].to(device) # [1, 3, H_vle, W_vle]

        return imgs_tensor
    
    @torch.inference_mode()
    def preprocess_texts(
            self, 
            texts: list[str],
            device: torch.device
    ) -> torch.Tensor:
        """
        Tokenize texts for FG-CLIP encoder.
        
        Args:
            texts: List of text strings
            device: Device to move tokenized texts to
            
        Returns:
            Tokenized text tensor of shape [B, context_length]
        """
        texts_tensor = torch.tensor(self.tokenizer(texts, max_length=self.context_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device) # # [B, context_length]
        return texts_tensor

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
    def get_similarity(
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
