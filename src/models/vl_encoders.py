from config import *
from utils import Registry

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

VLE_REGISTRY = Registry()


class MapComputeMode(Enum):
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
    global_image_token: torch.Tensor    # [B_i, D]
    global_text_token: torch.Tensor     # [B_i, B_t, D]
    local_image_tokens: torch.Tensor    # [B_i, n_i, D]
    local_text_tokens: torch.Tensor     # [B_t, n_t, D]


@dataclass
class VLEncoderOutputWithAttn(VLEncoderOutput):
    attn_maps_flat: torch.Tensor # [B_i, B_t, n_i+1]


class VLEncoder(ABC):
    """
    TODO
    """
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    @abstractmethod
    @torch.inference_mode()
    def preprocess_images(
            self,
            images: torch.Tensor | list[torch.Tensor] | list[Image.Image],
            device: str = "cuda", 
            **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    @torch.inference_mode()
    def preprocess_texts(
            self,
            texts: list[str],
            device: str = "cuda",
            **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode_and_project(
            self,
            images: torch.Tensor,
            texts: torch.Tensor,
            broadcast: bool = False,
            **kwargs
    ) -> VLEncoderOutput:
        raise NotImplementedError
    
    @abstractmethod
    @torch.inference_mode()
    def get_similarity(
            self,
            images: torch.Tensor,
            texts: torch.Tensor,
            broadcast: bool = False
    ) -> torch.Tensor:
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
    ) -> torch.Tensor:
        match map_compute_mode:
            case MapComputeMode.ATTENTION:
                return self.get_attn_maps(images, texts, upsample_size=upsample_size, upsample_mode=upsample_mode, broadcast=broadcast)
            case MapComputeMode.SIMILARITY:
                return self.get_sim_maps(images, texts, upsample_size=upsample_size, upsample_mode=upsample_mode, broadcast=broadcast)
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
    ) -> torch.Tensor:
        """
        Encodes image and text and returns the attention maps from the visual projection layer.
        """
        # _, [B_i, B_t, D], [B_i, n_i, D], _, _, [B_i, B_t, n_i+1]
        # 'B_t' = 1 if 'broadcast' = False
        vle_output = self.encode_and_project(images, texts, broadcast)

        text_batch_size = vle_output.global_text_token.shape[1] # B_t
        image_batch_size, image_num_patches = vle_output.local_image_tokens.shape[:2] # B_i, n_i
        
        # reshape attention maps
        num_patches_per_axis = int(math.sqrt(image_num_patches))
        # TODO I think that [cls] token is at last position, find out if it's true
        attn_maps: torch.Tensor = vle_output.attn_maps_flat[:, :, :-1].view(image_batch_size, text_batch_size, num_patches_per_axis, num_patches_per_axis) # [B_i, B_t, sqrt(n_i), sqrt(n_i)]
        min_attn, max_attn = attn_maps.min(), attn_maps.max() # max token attention
        
        # upsampling
        if upsample_size:
            attn_maps: torch.Tensor = TF.resize(attn_maps, upsample_size, interpolation=upsample_mode) # [B_i, B_t, H, W]

        # TODO do not output 'min_attn' and 'max_attn' anymore, make it produce by downstream functions.
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
    
    def set_vision_trainable_params(self, *arg, **kwargs) -> None:
        raise NotImplementedError
    
    def create_loss(self, *args, **kwargs) -> nn.Module:
        raise NotImplementedError

    def evaluate(
            self,
            dl: DataLoader,
            criterion: nn.modules.loss._Loss,
    ) -> torch.Tensor:
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

                torch.cuda.synchronize() if CONFIG["device"] == "cuda" else None
        
        loss = running_loss / running_supcount
        
        return loss
    
    @abstractmethod
    def count_tokens(self, *args, **kwargs) -> None:
        raise NotImplementedError

@dataclass
class FLAIROutput(VLEncoderOutputWithAttn):
    local_image_features: torch.Tensor = None # [B_i, B_t, D]

@VLE_REGISTRY.register("flair")
class FLAIRAdapter(VLEncoder):
    """
    TODO
    """
    def __init__(
            self,
            version: Literal['flair-cc3m-recap.pt',
                             'flair-cc12m-recap.pt',
                             'flair-yfcc15m-recap.pt',
                             'flair-merged30m.pt'] = 'flair-cc3m-recap.pt',
            device: str = 'cuda',
            vision_adapter: bool = False
    ) -> None:
        self.device = device
        self.version = version

        # Model
        pretrained = flair.download_weights_from_hf(model_repo='xiaorui638/flair', filename=version)
        model, _, preprocess_fn = flair.create_model_and_transforms('ViT-B-16-FLAIR', pretrained=pretrained, device=self.device)
        if vision_adapter:
            # NOTE the authors do not use a bias nor an activation function, I won't use them either, but I should experiment anyway.
            vision_adapter = nn.Linear(512, 512, bias=False, device=self.device)
            model.add_module('vision_adapter', vision_adapter)
        self.model: FLAIR = model
        self.init_weights()
        self.model.requires_grad_(False)
        self.model.eval()

        # Preprocess
        self.preprocess_fn: T.Compose = preprocess_fn

        # Tokenizer
        self.tokenizer: SimpleTokenizer = flair.get_tokenizer('ViT-B-16-FLAIR')
        self.context_length = self.tokenizer.context_length

    def init_weights(self) -> None:
        if hasattr(self.model, 'vision_adapter'):
            # Use Xavier Uniform initialisation for the weight matrix
            nn.init.xavier_uniform_(self.model.vision_adapter.weight)
            # Initialise the bias to zeros
            if self.model.vision_adapter.bias is not None:
                nn.init.zeros_(self.model.vision_adapter.bias)

    @torch.inference_mode()
    def preprocess_images(
            self, 
            images: torch.Tensor | list[torch.Tensor] | list[Image.Image],
            device: str = "cuda"
    ) -> torch.Tensor:
        if isinstance(images, torch.Tensor) or isinstance(images[0], torch.Tensor):
            images = [to_pil_image(img) for img in images]
        imgs_tensor = torch.stack([self.preprocess_fn(img) for img in images]).to(device) # [1, 3, H_vle, W_vle]
        return imgs_tensor
    
    @torch.inference_mode()
    def preprocess_texts(
            self, 
            texts: list[str],
            device: str = "cuda"
    ) -> torch.Tensor:
        texts_tensor = self.tokenizer(texts).to(device) # [B, context_length]
        return texts_tensor
    
    def encode_and_project(
            self,
            images: torch.Tensor,
            texts: torch.Tensor,
            broadcast: bool = False
    ) -> FLAIROutput:

        # get the raw embeddings from the encoders
        global_image_token, local_image_tokens = self.model.encode_image(images)    # [B_i, d_i], [B_i, n_i, d_i]
        global_text_token, local_text_tokens = self.model.encode_text(texts)        # [B_t, d_t], [B_t, n_t, d_t]

        # project the raw embeddings into the same space
        global_image_token: torch.Tensor = self.model.image_post(global_image_token)    # [B_i, D]
        local_image_tokens: torch.Tensor = self.model.image_post(local_image_tokens)    # [B_i, n_i, D]
        global_text_token: torch.Tensor = self.model.text_post(global_text_token)       # [B_t, D]
        local_text_tokens: torch.Tensor = self.model.text_post(local_text_tokens)       # [B_t, n_t, D]
        
        if hasattr(self.model, 'vision_adapter'):
            global_image_token = self.model.vision_adapter(global_image_token) # [B_i, D]
            local_image_tokens = self.model.vision_adapter(local_image_tokens) # [B_i, n_i, D]

        image_batch_size = local_image_tokens.shape[0] # B_i, n_i

        if broadcast:
            # adapt text token for broadcasting
            global_text_token = global_text_token.unsqueeze(0).expand(image_batch_size, -1, -1) # [B_i, B_t, D]
        else:
            if images.shape[0] != texts.shape[0]:
                raise AttributeError(f"When not broadcasting, 'images' and 'texts' should contain the same number of elements, but got {images.shape[0]=} and {texts.shape[0]=} instead.")
            # from now on, B_t = 1
            global_text_token = global_text_token.unsqueeze(1) # [B_i, 1, D]

        # perform the attention pooling: condition the 'global_text_token' (Q) on the 'local_image_tokens' (K and V)
        local_image_features, attn_maps_flat = self.model.visual_proj(
            global_text_token,      # [B_i, B_t, D]
            local_image_tokens,     # [B_i, n_i, D]
            local_image_tokens,     # [B_i, n_i, D]
            output_attn_weights=True
        ) # [B_i, B_t, D], [B_i, B_t, n_i+1] (the +1 is there for the added 'cls' token)
        
        flair_output = FLAIROutput(
            global_image_token, # [B_i, D]
            global_text_token, # [B_i, B_t, D]
            local_image_tokens, # [B_i, n_i, D]
            local_text_tokens, # [B_t, n_t, D]
            attn_maps_flat, # [B_i, B_t, n_i+1]
            local_image_features, # [B_i, B_t, D]
        )

        return flair_output
    
    @torch.inference_mode()
    def get_similarity(
            self,
            images: torch.Tensor,
            texts: torch.Tensor,
            broadcast: bool = False
    ) -> torch.Tensor:
        if broadcast:
            flair_output = self.encode_and_project(images, texts, broadcast=True)
            image_features, text_features = F.normalize(flair_output.global_image_token, dim=-1), F.normalize(flair_output.global_text_token, dim=-1)
            text_features = text_features.squeeze(1)
            sim = torch.einsum('bf,bif->bi', image_features, text_features)
        else:
            flair_output = self.encode_and_project(images, texts, broadcast=False)
            image_features, text_features = F.normalize(flair_output.global_image_token, dim=-1), F.normalize(flair_output.global_text_token, dim=-1)
            text_features = text_features.squeeze(1)
            sim = torch.einsum('bf,bf->b', image_features, text_features)
        return sim
    
    def set_vision_trainable_params(
            self,
            trainable_modules: list[Literal['vision_adapter',
                                            'proj',
                                            'visual_proj']],
    ) -> None:
        # LUT: module name -> module
        train_modules_lut = {
            'vision_adapter': self.model.vision_adapter,
            'proj': self.model.image_post,
            'visual_proj': self.model.visual_proj
        }

        # first, gradients are disabled for all modules.
        self.model.requires_grad_(False)

        # access the right modules with the LUT
        train_params: list[nn.Module] = [train_modules_lut[m].requires_grad_(True) for m in trainable_modules]
        # enable the gradiens in the accessed modules
        [p.requires_grad_(True) for p in train_params]
    
    def create_loss(
            self,
            add_mps_loss: bool,
            world_size: int = 1,
            rank: int = 0,
            num_caps_per_img: int = 1
    ) -> FlairLoss:
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
        tokenizer = flair.get_tokenizer('ViT-B-16-FLAIR', context_length=1000) # I do not expect to use text longer than 1000 tokens. If so, increase it to working upper bound.
        return len(tokenizer.encode(text)) + 2 # 'SoT' and 'EoT' are added.


class SimSegAdapter(VLEncoder):
    """
    TODO
    """
    # too much integration work likely for unimpressive results, I would skip this.


@VLE_REGISTRY.register("fg-clip")
class FG_CLIPAdapter(VLEncoder):
    """
    TODO
    """
    def __init__(
            self,
            checkpoint: Literal['fg-clip-base', 
                                'fg-clip-large'] = 'fg-clip-base',
            device: str = 'cuda',
            long_captions = True
    ) -> None:
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
            device: str = "cuda"
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
            device: str = "cuda"
    ) -> torch.Tensor:
        texts_tensor = torch.tensor(self.tokenizer(texts, max_length=self.context_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=device) # # [B, context_length]
        return texts_tensor

    def encode_and_project(
            self,
            images: torch.Tensor,
            texts: torch.Tensor,
            broadcast: bool = False
    ) -> FLAIROutput:

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
        return len(self.tokenizer.encode(text)) # 'SoT' and 'EoT' are added.
    
    def get_attn_maps(self, *args, **kwargs) -> None:
        raise AttributeError("The VLE 'FG-CLIP' does not provide attention maps.")

def main() -> None:
    
    from typing import cast
    print(VLE_REGISTRY.registered_objects())
    model: VLEncoder = VLE_REGISTRY.get('flair')
    model = cast(FLAIRAdapter, model)
    print(model.create_loss(add_mps_loss=True, rank=0, world_size=1, num_caps_per_img=1))
    
if __name__ == '__main__':
    main()
