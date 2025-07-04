from config import CONFIG
from vendors.flair.src import flair

import torch
from torch import nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from abc import abstractmethod, ABC
from open_clip.tokenizer import HFTokenizer, SimpleTokenizer
import math

from typing import Optional
from vendors.flair.src.flair.model import FLAIR

class ImageTextEncoder(ABC):
    """
    TODO
    """
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_logits(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def get_attn_maps(self, *args, **kwargs) -> None:
        raise NotImplementedError

class FLAIRAdapter(ImageTextEncoder):
    """
    TODO
    """
    def __init__(
            self,
            version: str = 'flair-cc3m-recap.pt',
            device: str = 'cuda',
    ) -> None:
        pretrained = flair.download_weights_from_hf(model_repo='xiaorui638/flair', filename=version)
        model, _, preprocess_fn = flair.create_model_and_transforms('ViT-B-16-FLAIR', pretrained=pretrained)
        model.to(device)
        model.eval()
        self.model: FLAIR = model
        self.preprocess_fn: T.Compose = preprocess_fn
        self.tokenizer: SimpleTokenizer = flair.get_tokenizer('ViT-B-16-FLAIR')

    def encode_and_pool(
            self,
            images: torch.Tensor,
            texts: torch.Tensor,
            broadcast: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # get the raw embeddings from the encoders
        global_image_token, local_image_tokens = self.model.encode_image(images) # [B_i, d_i], [B_i, n_i, d_i]
        global_text_token, local_text_tokens = self.model.encode_text(texts) # [B_t, d_t], [B_t, n_t, d_t]

        # project the raw embeddings into the same space
        local_image_tokens: torch.Tensor = self.model.image_post(local_image_tokens) # [B_i, n_i, D]
        global_text_token: torch.Tensor = self.model.text_post(global_text_token) # [B_t, D]

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
        
        # [B_i, d_i], [B_i, B_t, D], # [B_i, n_i, D], [B_t, n_t, d_t], [B_i, B_t, D], [B_i, B_t, n_i+1]
        return global_image_token, global_text_token, local_image_tokens, local_text_tokens, local_image_features, attn_maps_flat
    
    def get_logits(
            self,
            images: torch.Tensor,
            texts: torch.Tensor,
            broadcast: bool = False
    ) -> torch.Tensor:
        if broadcast:
            image_logits, text_logits = self.model.get_logits(images, texts)
        else:
            # [B_i, B_t, D], [B_i, n_i, D]
            # 'B_t' might be 1 if 'broadcast' = False
            _, global_text_token, _, _, local_image_features, _ = self.encode_and_pool(images, texts, broadcast)

            text_features, image_features = F.normalize(global_text_token, dim=-1), F.normalize(local_image_features, dim=-1)

            image_logits = self.model.logit_scale.exp() * torch.einsum('bij,bij->bi', image_features, text_features) # (B, B*K)
            image_logits += self.model.logit_bias

            text_logits = image_logits.T
        return image_logits
    
    def get_attn_maps(
            self,
            images: torch.Tensor,
            texts: torch.Tensor,
            upsample_size: Optional[int | tuple[int]] = None,
            upsample_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
            normalize: bool = False,
            broadcast: bool = False
    ) -> torch.Tensor:
        """
        Encodes image and text and returns the attention maps from the visual projection layer.
        """
        # [B_i, B_t, D], [B_i, n_i, D]
        # 'B_t' = 1 if 'broadcast' = False
        _, global_text_token, local_image_tokens, _, _, attn_maps_flat = self.encode_and_pool(images, texts, broadcast)

        text_batch_size = global_text_token.shape[1] # B_t
        image_batch_size, image_num_patches = local_image_tokens.shape[:2] # B_i, n_i
        
        # reshape attention maps
        num_patches_per_axis = int(math.sqrt(image_num_patches))
        # TODO I think that [cls] token is at last position, find out if it's true
        attn_maps = attn_maps_flat[:, :, :-1].view(image_batch_size, text_batch_size, num_patches_per_axis, num_patches_per_axis) # [B_i, B_t, n_i, n_i]
        max_attn = attn_maps_flat.max() # max token attention
        
        # upsampling
        if upsample_size:
            attn_maps: torch.Tensor = TF.resize(attn_maps, upsample_size, interpolation=upsample_mode) # [B_i, B_t, H, W]

        # normalize
        # TODO is this the best way to normalize attentions?
        if normalize:
            norm_dims = list(range(attn_maps.ndim))[-2:] # indices of the last two dimensions (per-image normalization)
            max_per_image = attn_maps.amax(dim=norm_dims, keepdim=True)
            min_per_image = attn_maps.amin(dim=norm_dims, keepdim=True)
            attn_maps = (attn_maps - min_per_image)/(max_per_image - min_per_image)

        return attn_maps, max_attn

    def count_tokens(
            self,
            text: str
    ) -> int:
        tokenizer = flair.get_tokenizer('ViT-B-16-FLAIR', context_length=1000)
        return len(tokenizer.encode(text)) + 2 # 'SoT' and 'EoT' are added.

def main() -> None:
    pass
    
if __name__ == '__main__':
    main()
