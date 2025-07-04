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

def overlay_attn_map(
        background: torch.Tensor | Image.Image,
        overlay: torch.Tensor,
        alpha: float = 1.0,
        rgb_fill: Optional[tuple[int, int, int]] = [255, 0, 0],
) -> Image.Image:
    if isinstance(background, torch.Tensor):
        background_img = to_pil_image(background.cpu()).convert('RGBA')
    else:
        background_img = background.convert('RGBA')
    rgb_values = (torch.tensor(rgb_fill)*alpha/255.).view(3, 1, 1) # [3, 1, 1]
    tensor_hw = rgb_values.expand(3, background_img.size[-1], background_img.size[-2]).cpu() # (3, H, W)
    overlay_tensor = torch.concat([tensor_hw, overlay.cpu()], dim=0) # [4, H, W]
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
