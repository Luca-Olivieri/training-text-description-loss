from IPython.display import Markdown, display
from path import MISC_PATH
from models.vl_encoders import VLEncoder
from utils import batch_list, flatten_list

from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.v2.functional as TF
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Optional

from utils import Prompt

def normalize_attn_maps(
        attn_maps: torch.Tensor,        
) -> torch.Tensor:
    norm_dims = list(range(attn_maps.ndim))[-2:] # indices of the last two dimensions (per-image normalization)
    max_per_image = attn_maps.amax(dim=norm_dims, keepdim=True)
    min_per_image = attn_maps.amin(dim=norm_dims, keepdim=True)
    attn_maps = (attn_maps - min_per_image)/(max_per_image - min_per_image)
    return attn_maps

def normalize_sim_maps(
        sim_maps: torch.Tensor,        
) -> torch.Tensor:
    pos_sim_maps = sim_maps.clamp(min=0)
    neg_sim_maps = -(sim_maps.clamp(max=0))
    norm_dims = list(range(sim_maps.ndim))[-2:] # indices of the last two dimensions (per-image normalization)

    abs_max_per_map = sim_maps.abs().amax(dim=norm_dims, keepdim=True)

    if torch.any(pos_sim_maps != 0):
        pos_sim_maps = pos_sim_maps/abs_max_per_map
    
    if torch.any(neg_sim_maps != 0):
        neg_sim_maps = neg_sim_maps/abs_max_per_map
        neg_sim_maps = -neg_sim_maps

    sim_maps = pos_sim_maps + neg_sim_maps

    return sim_maps

def overlay_map(
        background: torch.Tensor | Image.Image,
        map: torch.Tensor,
        alpha: float = 1.0,
        pos_rgb_fill: tuple[int, int, int] = (255, 0, 0),
        neg_rgb_fill: tuple[int, int, int] = (0, 0, 255),
        normalize: bool = True
) -> Image.Image:
    if isinstance(background, torch.Tensor):
        background_img = to_pil_image(background.cpu()).convert('RGBA')
    else:
        background_img = background.convert('RGBA')
    
    pos_mask = (map > 0).cpu() # Create a mask where overlay > 0 is True (positive), else False (negative)
    mask_expanded = pos_mask.unsqueeze(0) # Unsqueeze the mask to [1, H, W] to allow broadcasting with [3, 1, 1] for RGB values

    # Unsqueeze the RGB fill values to [3, 1, 1]
    pos_rgb_fill_expanded = (torch.tensor(pos_rgb_fill)*alpha/255.).unsqueeze(1).unsqueeze(2)
    neg_rgb_fill_expanded = (torch.tensor(neg_rgb_fill)*alpha/255.).unsqueeze(1).unsqueeze(2)

    # Use torch.where to select values based on the mask
    # This will broadcast mask_expanded to [3, H, W] and then apply the condition
    output_rgb = torch.where(mask_expanded, pos_rgb_fill_expanded, neg_rgb_fill_expanded).squeeze(0)
    
    if normalize:
        map = normalize_sim_maps(map)

    overlay_tensor = torch.concat([output_rgb.cpu(), map.abs().cpu()], dim=0) # [4, H, W]
    overlay_img = to_pil_image(overlay_tensor).convert('RGBA')
    return Image.alpha_composite(background_img, overlay_img)

def format_image_with_caption(
        image: Image.Image,
        caption: str,
        caption_height: int = 40,
        font_size: int = 32
) -> Image.Image:
    font = ImageFont.truetype(f"{MISC_PATH}/Arial.ttf", size=font_size)

    title_img = Image.new("RGB", (image.width, caption_height), "white")
    draw = ImageDraw.Draw(title_img)

    # Center the text
    bbox = draw.textbbox((0, 0), caption, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_x = (image.width - text_width) // 2
    text_y = (caption_height - text_height) // 2

    draw.text((text_x, text_y), caption, fill="black", font=font)
    
    concatenated_image = Image.new("RGB", (image.size[0], image.size[1]+caption_height), "white")
    concatenated_image.paste(image, (0, 0))  # Paste image
    concatenated_image.paste(title_img, (0, image.size[1]))  # Paste caption below image

    return concatenated_image

def display_token_length_distr(
        token_lengths: list[int],
        bins: int = 20
) -> None:
    sns.histplot(token_lengths, bins=bins)
    plt.xlabel("Token Length")
    plt.ylabel("Count")
    plt.title("Distribution of Token Lengths")
    plt.show()


def display_prompt(full_prompt: str | Prompt) -> None:
    """Displays a prompt, which can be a string or a list of strings and images, using IPython display utilities.

    Args:
        full_prompt (str | Prompt): The prompt to display.
    """
    if isinstance(full_prompt, str):
        display(Markdown(full_prompt))
    else:
        for prompt in full_prompt:
            if isinstance(prompt, Image.Image):
                display(prompt)
            else:
                display(Markdown(prompt))
