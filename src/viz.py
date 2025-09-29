from config import *
from config import torch
from prompter import Prompt

from IPython.display import Markdown, display
from path import MISC_PATH

from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from utils import blend_tensors

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
        pos_rgb_fill: tuple[int, int, int] = (0, 255, 0),
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
    pos_rgb_fill_expanded = (torch.tensor(pos_rgb_fill)/255.).unsqueeze(1).unsqueeze(2)
    neg_rgb_fill_expanded = (torch.tensor(neg_rgb_fill)/255.).unsqueeze(1).unsqueeze(2)

    # Use torch.where to select values based on the mask
    # This will broadcast mask_expanded to [3, H, W] and then apply the condition
    output_rgb = torch.where(mask_expanded, pos_rgb_fill_expanded, neg_rgb_fill_expanded).squeeze(0)
    
    if normalize:
        map = normalize_sim_maps(map)

    overlay_tensor = torch.concat([output_rgb.cpu(), map.abs().cpu()*(alpha)], dim=0) # [4, H, W]
    overlay_img = to_pil_image(overlay_tensor).convert('RGBA')
    return Image.alpha_composite(background_img, overlay_img).convert('RGB')

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
    
def write_html_multi_row_image_caption(
        title: str,
        rows: dict[str: list[Image.Image]],
        captions: list[str]
) -> None:
    row_labels = list(rows.keys())
    column_data = [list(x) for x in list(zip(*list(rows.values())))]

    generated_html = create_multi_row_gallery(title, row_labels, column_data, captions)

    try:
        with open(f"{title}.html", "w") as file:
            file.write(generated_html)
        print("Successfully created index.html. Open it in your browser to see the result.")
    except IOError as e:
        print(f"Error writing to file: {e}")

def get_image_base64_data(image):
    """
    Converts a PIL Image object to Base64 encoded string.
    """
    if image is None:
        print("Warning: Image is None. Skipping.")
        return None, None

    try:
        from io import BytesIO
        
        # Convert image to RGB if it's not already (handles RGBA, L, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save image to BytesIO buffer as PNG
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}", "image/png"
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None, None

def create_multi_row_gallery(
        title: str,
        row_labels: list[str],
        column_data: list[list[Image.Image]],
        captions: list[str]
) -> str:
    """
    Generates HTML for a multi-row image gallery where each caption has a
    column of images above it, all scrolling horizontally together.

    Args:
        row_labels (list): A list of strings for the row headers.
        column_data (list of lists): A list where each inner list contains
                                     PIL Image objects for one vertical column.
        captions (list): A list of captions, one for each column.

    Returns:
        str: The complete HTML content as a string.
    """
    # --- Data Validation ---
    num_columns = len(column_data)
    num_rows = len(row_labels)
    if num_columns != len(captions):
        print("Error: The number of columns in COLUMN_IMAGE_PATHS must match the number of CAPTION_TEXTS.")
        return "Error page: data mismatch."
    for i, col in enumerate(column_data):
        if len(col) != num_rows:
            print(f"Error: Column {i+1} has {len(col)} images, but there are {num_rows} row labels. These must match.")
            return "Error page: data mismatch."

    # --- Grid Generation ---
    grid_items_html = []

    # Generate image rows
    for row_idx, label_text in enumerate(row_labels):
        # Add the sticky row label for this row (Grid Item 1 in the row)
        grid_items_html.append(f"""
        <div class="sticky left-0 bg-gray-100 p-4 flex items-center justify-end">
            <span class="font-bold text-gray-600 text-right">{label_text}</span>
        </div>""")

        # Add all images for this row across the columns
        for col_idx in range(num_columns):
            image = column_data[col_idx][row_idx]
            base64_src, _ = get_image_base64_data(image)
            if base64_src:
                grid_items_html.append(f"""
                <div class="bg-white shadow-lg rounded-lg overflow-hidden flex items-center justify-center p-2">
                    <img src="{base64_src}" alt="Image for {captions[col_idx]}" class="max-w-full max-h-48 object-contain">
                </div>""")
            else:
                grid_items_html.append('<div class="bg-gray-200 shadow-lg rounded-lg flex items-center justify-center"><span class="text-xs text-gray-500">Image not found</span></div>')

    # Generate caption row
    # Add a spacer for the label column (Grid Item 1 in the row)
    grid_items_html.append('<div class="sticky left-0 bg-gray-100"></div>')

    for caption_text in captions:
        grid_items_html.append(f"""
        <figcaption class="pt-2 text-gray-700 text-sm leading-relaxed">
            {caption_text}
        </figcaption>""")

    # --- HTML and CSS Generation ---
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #f3f4f6; /* bg-gray-100 */
        }}
        /* The main scrollable container */
        .horizontal-scroll-container {{
            -ms-overflow-style: auto; /* IE and Edge */
            scrollbar-width: thin;   /* Firefox */
        }}
        .horizontal-scroll-container::-webkit-scrollbar {{
            height: 12px;
        }}
        .horizontal-scroll-container::-webkit-scrollbar-track {{
            background: #e5e7eb; /* bg-gray-200 */
        }}
        .horizontal-scroll-container::-webkit-scrollbar-thumb {{
            background-color: #9ca3af; /* bg-gray-400 */
            border-radius: 10px;
            border: 3px solid #f3f4f6; /* bg-gray-100 */
        }}
        .horizontal-scroll-container::-webkit-scrollbar-thumb:hover {{
            background: #6b7280; /* bg-gray-500 */
        }}
        /* Define the grid layout */
        .gallery-grid {{
            display: inline-grid;
            grid-auto-flow: row;
            /* {num_rows + 1} rows: N image rows + 1 caption row */
            grid-template-rows: repeat({num_rows}, auto) auto;
            /* {num_columns + 1} columns: 1 label column + M data columns */
            grid-template-columns: 10rem repeat({num_columns}, 20rem);
            gap: 1rem; /* Defines the visual separator space */
        }}
    </style>
</head>
<body class="min-h-screen flex flex-col items-center justify-center p-4 sm:p-8">

    <div class="w-full max-w-screen-xl mx-auto">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-8">{title}</h1>
        
        <!-- The horizontally scrolling container -->
        <div class="horizontal-scroll-container overflow-x-auto pb-4">
            <!-- The grid that holds all content and ensures alignment -->
            <div class="gallery-grid">
                {''.join(grid_items_html)}
            </div>
        </div>
    </div>

</body>
</html>
"""
    return html_content

def get_layer_numel_str(
        module: nn.Module,
        print_only_total: bool = False,
        only_trainable: bool = False
) -> str:
    s = 0
    out_str: str = ""
    layer_names = []
    layer_params = []
    for name, params in module.named_parameters():
        if (only_trainable and params.requires_grad) or not only_trainable:
            layer_names.append(name)
            layer_params.append(params)
            s += params.numel()
    max_name_len = max([len(n) for n in layer_names])
    if not print_only_total:
        out_str += '\n'.join([f"{n:<{max_name_len+1}}: {ps.numel():,}" for n, ps in zip(layer_names, layer_params)]) + '\n'
    out_str += f"Total: {s:,}"
    return out_str

def format_to_title(
        text: str,
        total_length: int = 100,
        pad_symbol: str = '-'
) -> str:
    """
    Centers a title within a given total length, padding with hyphens.

    Args:
        title (str): The string to be centered.
        total_length (int): The total desired length of the output string.

    Returns:
        str: The centered string with hyphen padding.
             Returns the original title if total_length is less than
             or equal to the length of the title.
    """
    text = f"[ {text} ]"
    if total_length <= len(text):
        return text

    padding_needed = total_length - len(text)
    
    # Calculate padding for left and right
    # Integer division might put an extra '-' on the right if padding_needed is odd
    left_padding = padding_needed // 2
    right_padding = padding_needed - left_padding

    centered_string = pad_symbol * left_padding + text + pad_symbol * right_padding
    return centered_string


def create_diff_mask(
        mask1: torch.Tensor,
        mask2: torch.Tensor,
) -> torch.Tensor:
    """
    Creates a binary difference mask from two integer-based segmentation masks.

    The operation is fully vectorized and runs efficiently on CUDA devices.

    Args:
        mask1 (torch.Tensor): The first segmentation mask with class indices.
                              Expected dtype: torch.long, torch.int, etc.
        mask2 (torch.Tensor): The second segmentation mask with class indices.
                              Must have the same shape and device as mask1.

    Returns:
        torch.Tensor: A mask with value 255 (uint8) where pixels in mask1 and mask2
                      are different, and 0 where they are the same.
    """
    # 1. Ensure the masks have the same shape for element-wise comparison.
    assert mask1.shape == mask2.shape, f"Input masks must have the same shape, but got {mask1.shape} and {mask2.shape}"

    # 2. Perform element-wise comparison. This creates a boolean tensor.
    #    'True' where elements are not equal, 'False' where they are equal.
    #    This is the functional equivalent of `torch.ne(mask1, mask2)`.
    diff = (mask1 != mask2).to(torch.uint8)

    return diff

def create_cs_diff_masks(
        sc_img: torch.Tensor,
        gt: torch.Tensor,
        pr: torch.Tensor,
        sign_classes: list[int],
        alpha: float = 0.55
) -> dict[int, torch.Tensor]:
    
    cs_diff_masks: dict[int, torch.Tensor] = {}

    for pos_c in sign_classes:
        pos_class_gt = (gt == pos_c)
        pos_class_pr = (pr == pos_c)

        diff_mask = create_diff_mask(pos_class_gt, pos_class_pr)

        # L overlay image
        ovr_diff_mask_L = blend_tensors(sc_img, diff_mask*255, alpha) # int in range [0, 255]

        cs_diff_masks[pos_c] = ovr_diff_mask_L

    # TODO validate the format of the diff masks, they have to get into the VLE.

def create_cs_masks(
        sc_img: torch.Tensor,
        mask: torch.Tensor,
        sign_classes: list[int],
        alpha: float = 0.55
) -> dict[int, torch.Tensor]:
    """
    Creates class-specific difference masks in a vectorized, GPU-friendly manner.
    """
    if not sign_classes:
        return {}
    
    # Ensure tensors are on the same device (ideally 'cuda')
    device = mask.device
    sc_img = sc_img.to(device)
    mask = mask.to(device)

    # 1. Create a tensor of class values.
    # Shape: (N,) where N is the number of classes.
    classes_tensor = torch.tensor(sign_classes, device=device, dtype=mask.dtype)

    # 2. Create masks for all classes at once using broadcasting.
    # gt is (H, W), classes_tensor is (N,).
    # We reshape to (1, H, W) and (N, 1, 1) to trigger broadcasting.
    # The result `all_pos_class_gt` will have shape (N, H, W).
    all_pos_class_gt = (mask.unsqueeze(0) == classes_tensor.view(-1, 1, 1))

    # 3. Create all diff masks at once. The op is element-wise.
    # Shape: (N, H, W)

    # 4. Prepare for blending. Convert boolean diff masks to float overlays.
    # Shape: (N, H, W)

    # 5. Blend the single source image with the entire stack of overlays at once.
    # sc_img (H, W) is unsqueezed to (1, H, W) to broadcast across the N overlays.
    # Shape: (N, H, W)
    all_blended_masks = blend_tensors(sc_img.unsqueeze(0), all_pos_class_gt.unsqueeze(1).repeat(1, 3, 1, 1)*255, alpha)

    # 6. Create the final dictionary.
    # This is a fast CPU operation. We pair each class ID with its corresponding mask.
    cs_masks = {
        class_val: mask
        for class_val, mask in zip(sign_classes, all_blended_masks)
    }

    return cs_masks
