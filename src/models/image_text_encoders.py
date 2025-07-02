from config import CONFIG
from vendors.flair.src import flair

import torch
from torch import nn
import torchvision.transforms as T
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

class FLAIRAdapter(ImageTextEncoder):
    """
    TODO
    """
    def __init__(
            self,
            version: str = 'flair-cc3m-recap.pt',
            device: str = 'cuda',
    ):
        pretrained = flair.download_weights_from_hf(model_repo='xiaorui638/flair', filename=version)
        model, _, preprocess_fn = flair.create_model_and_transforms('ViT-B-16-FLAIR', pretrained=pretrained)
        model.to(device)
        model.eval()
        self.model: FLAIR = model
        self.preprocess_fn: T.Compose = preprocess_fn
        self.tokenizer: SimpleTokenizer = flair.get_tokenizer('ViT-B-16-FLAIR')
    
    # Add this new method inside the FLAIR class
    def get_attn_maps(
            self,
            images: torch.Tensor,
            texts: torch.Tensor,
            upsample_size: Optional[int | tuple[int]] = None,
            normalize: bool = False,
            broadcast: bool = False
    ) -> torch.Tensor:
        """
        Encodes image and text and returns the attention maps from the visual projection layer.
        """
        # get the raw embeddings from the encoders
        global_image_token, local_image_tokens = self.model.encode_image(images) # [B_i, d_i], [B_i, n_i, d_i]
        global_text_token, local_text_tokens = self.model.encode_text(texts) # [B_t, d_t], [B_t, n_t, d_t]

        # project the raw embeddings into the same space
        local_image_tokens: torch.Tensor = self.model.image_post(local_image_tokens) # [B_i, n_i, D]
        global_text_token: torch.Tensor = self.model.text_post(global_text_token) # [B_t, D]
        image_batch_size, image_num_patches = local_image_tokens.shape[:2] # B_i, n_i
        text_batch_size = global_text_token.shape[0] # B_t

        # TODO move the previous lines in a separate 'encode_and_project' method

        if broadcast:
            # adapt text token for broadcasting
            global_text_token = global_text_token.unsqueeze(0).expand(image_batch_size, -1, -1) # [B_i, B_t, D]
        else:
            text_batch_size = 1 # from now on, B_t = 1
            global_text_token = global_text_token.unsqueeze(1) # [B_i, 1, D]

        # perform the attention pooling
        local_image_features, attn_maps_flat = self.model.visual_proj(
            global_text_token,
            local_image_tokens,
            local_image_tokens,
            output_attn_weights=True
        ) # [B_i, B_t, D], [B_i, B_t, 1+n_i**2]
        
        # reshape attention maps
        num_patches_per_axis = int(math.sqrt(image_num_patches))
        attn_maps = attn_maps_flat[:, :, :-1].view(image_batch_size, text_batch_size, num_patches_per_axis, num_patches_per_axis) # [B_i, B_t, n_i, n_i]
        # TODO I think that [cls] token is at last position, find out if it's true    
        
        # upsampling
        if upsample_size:
            attn_maps: torch.Tensor = F.interpolate(attn_maps, upsample_size, mode='bilinear', align_corners=False) # [B_i, B_t, H, W]

        # normalize
        # TODO is this the best way to normalize attentions?
        if normalize:
            norm_dims = list(range(attn_maps.ndim))[-2:] # indices of the last two dimensions (per-image normalization)
            max_per_image = attn_maps.amax(dim=norm_dims, keepdim=True)
            min_per_image = attn_maps.amin(dim=norm_dims, keepdim=True)
            attn_maps = (attn_maps - min_per_image)/(max_per_image - min_per_image)

        return attn_maps


def main() -> None:
    pass
    
if __name__ == '__main__':
    main()
