from core.config import *
from core.torch_utils import is_list_of_tensors

import matplotlib.pyplot as plt
import io
import webcolors

from torchvision.utils import draw_segmentation_masks # TO BE REMOVED

from typing_extensions import deprecated

from core._types import deprecated, RGB_tuple

def full_color_map(
        N: int = 256,
        normalized: bool = False
) -> np.ndarray[int]:
    """Generates a full color map with N colors (integer or normalised to float).

    Args:
        N: The number of colors to account for.
        normalized: if True, colors are float in range (0., 1.). Otherwise, they are int in range (0, 255)
    
    Returns:
        NumPy ndarray enumerating the class colors.
    """
    def bitget(byteval: int, idx: int) -> bool:
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_inv_color_map_dict(
        color_map_dict: dict[int, RGB_tuple],
) -> dict[RGB_tuple, int]:
    """Gets the color map as dictionary {(r, g, b): cls_idx} for the 21 VOC classes.

    Returns:
        Dictionary mapping RGB tuple to class index.
    """
    inv_color_map_dict = {tuple(rgb): i for i, rgb in color_map_dict.items()}
    inv_color_map_dict[(255, 255, 255)] = 1 # to account for class-splitted prompts
    return inv_color_map_dict

def get_color_map_as_img(
        classes: list[str],
        with_void: bool = False,
) -> Image.Image:
    """Gets the color map as RGB image for the 21 VOC classes.

    Returns:
        Image visualizing the color map.
    """
    labels = classes.copy()
    labels.append("VOID")

    labels = [f"{l} [{i}]" for i, l in enumerate(labels)]

    nclasses = 21 if not with_void else 22
    row_size = 50
    col_size = 500 # 500 default
    cmap = full_color_map()
    array = np.empty((row_size*(nclasses), col_size, cmap.shape[1]), dtype=cmap.dtype)
    for i in range(nclasses):
        if i < 21:
            array[i*row_size:i*row_size+row_size, :] = cmap[i]
        else:
            array[i*row_size:i*row_size+row_size, :] = cmap[255]

    fig, ax = plt.subplots(figsize=(6, 12))
    ax.imshow(array)
    ax.set_yticks([row_size * i + row_size / 2 for i in range(nclasses)])
    if not with_void:
        labels.remove("VOID [21]")
    ax.set_yticklabels(labels)
    ax.set_xticks([])
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def get_color_map_as_rgb(
        classes: list[str],
        color_map_dict: dict[int, RGB_tuple],
) -> str:
    """Gets the color map as dictionary {"cls [cls_idx]": (r, g, b)}.

    Returns:
        String representation of class name to RGB mapping.
    """
    return str({f"{classes[i]} [{i}]": rgb for i, rgb in color_map_dict.items()})

def get_color_map_as_names(
        classes: list[str],
        color_map_dict: dict[int, RGB_tuple],
) -> str:
    """Gets the color map as dictionary {"cls [cls_idx]": name}.

    Each color is associated to the name of the RGB nearest-neighbour to CSS3 colors.

    Returns:
        String representation of class name to closest CSS3 color name mapping.
    """
    def closest_color_name(
            rgb: RGB_tuple
    ) -> str:
        """Gets the closest CSS3 color name for a given RGB tuple."""
        spec = "css3"
        rgb_to_names = {tuple(webcolors.name_to_rgb(c_name, spec)): c_name for c_name in webcolors.names(spec=spec)}
        try:
            return webcolors.rgb_to_name(rgb)
        except ValueError:
            # Find the closest color
            min_colors = {}
            for (r_c, g_c, b_c), name in rgb_to_names.items():
                distance = (r_c - rgb[0])**2 + (g_c - rgb[1])**2 + (b_c - rgb[2])**2
                min_colors[distance] = name
            return min_colors[min(min_colors.keys())]

    c_names = {i: closest_color_name(rgb) for i, rgb in enumerate(color_map_dict.values())}
    c_names[8] = "darkmaroon"
    return str({f"{classes[i]} [{i}]": c_names[i] for i, rgb in enumerate(color_map_dict.values())})

def get_color_map_as_patches(
        classes: list[str],
        color_map_dict: dict[int, RGB_tuple],
        patch_size: tuple[int, int] = (32, 32),
) -> tuple:
    """Gets the color map as list of patches of size 'patch_size'.

    Args:
        patch_size: Size of each patch. Defaults to (32, 32).

    Returns:
        Tuple of class name and PIL.Image.Image patch pairs.
    """
    color_patches = []
    for color_index, rgb in color_map_dict.items():
        # Create a new image with the specified size and fill it with the RGB color.
        patch = Image.new("RGB", patch_size, rgb)
        color_patches.append(f"{classes[color_index]} [{color_index}]:")
        color_patches.append(patch)
    return tuple(color_patches)

def get_color_map_as(
        format: str,
        **kwargs
) -> Any:
    """Gets the color map in a specified format.

    Args:
        format: Format type ('img', 'rgb', 'names', 'patches').

    Returns:
        The color map in the selected format.
    """
    format2fn = {
        "img": get_color_map_as_img,
        "rgb": get_color_map_as_rgb,
        "names": get_color_map_as_names,
        "patches": get_color_map_as_patches
        }    
    fn = format2fn[format]
    return fn(**kwargs)

def apply_colormap(
        input_tensor: torch.Tensor | list[torch.Tensor],
        color_map: dict[int, RGB_tuple]
) -> torch.Tensor:
    """
    Applies a color map to an integer tensor to produce a 3-channel color image tensor.

    Any integer class label in the input_tensor that is not a key in the color_map
    will be mapped to black (0, 0, 0) by default.

    This function is highly optimized for CUDA and uses a lookup table for fast,
    vectorized mapping.

    Args:
        input_tensor (torch.Tensor): A tensor of integer class labels with shape
                                    [B, 1, H, W].
        color_map (Dict[int, RGB_tuple]): A dictionary mapping each
                                                    class label (int) to an
                                                    RGB color tuple (e.g., (255, 0, 0)).

    Returns:
        torch.Tensor: The resulting color image tensor with shape [B, 3, H, W] and
                    dtype=torch.uint8.
    """
    if is_list_of_tensors(input_tensor):
        input_tensor = torch.stack(input_tensor, dim=0)

    # 1. --- Input Validation and Setup ---
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"input_tensor must be a either be torch.Tensor or a list of torch.Tensor, got {type(input_tensor)}")
    if input_tensor.dim() != 4 or input_tensor.shape[1] != 1:
        raise ValueError(f"input_tensor must have shape [B, 1, H, W], got {input_tensor.shape}")
    
    device = input_tensor.device

    # 2. --- Create the Lookup Table (LUT) ---
    # Determine the size of the LUT. It must be large enough to handle all
    # keys in the color_map AND all values in the input_tensor to avoid
    # an out-of-bounds indexing error.
    max_key = max(color_map.keys()) if color_map else -1
    max_val_in_tensor = input_tensor.max().item()
    lut_size = max(max_key, max_val_in_tensor) + 1
    
    # Create the LUT on the same device as the input tensor, initialized to the
    # default color (black).
    # The default color for any unmapped index will be (0, 0, 0).
    default_color = torch.tensor([0, 0, 0], dtype=torch.uint8, device=device)
    lut = default_color.repeat(lut_size, 1)

    # Populate the LUT from the dictionary.
    if color_map:
        keys = torch.tensor(list(color_map.keys()), device=device, dtype=torch.long)
        values = torch.tensor(list(color_map.values()), dtype=torch.uint8, device=device)
        lut.scatter_(0, keys.unsqueeze(1).repeat(1, 3), values)

    # 3. --- Perform the Mapping ---
    # Squeeze the channel dimension (C=1) and ensure indices are long.
    indices = input_tensor.squeeze(1).long()
    
    # Use advanced indexing to map indices to colors.
    # This is the core, highly parallelized operation.
    colored_tensor = lut[indices]

    # 4. --- Reshape to Standard Image Format [B, C, H, W] ---
    output_tensor = colored_tensor.permute(0, 3, 1, 2)

    return output_tensor.contiguous()
    
@deprecated("This method is used for in the class PromptBuilder, but a betetr version exists. Test it!")
def apply_colormap_(
        mask: torch.Tensor,
        color_map: dict[int, RGB_tuple],
        num_classes: int
) -> torch.Tensor:
    """Receives a tensor of shape [C, H, W] and returns a PIL Image with the color map applied.

    Args:
        mask: Segmentation mask tensor of shape [C, H, W].
        color_map: Dictionary mapping class indices to RGB tuples.

    Returns:
        Image with color map applied.
    """
    assert len(mask.shape) == 3, mask.shape
    mask_all_classes = (mask == torch.arange(num_classes).to(CONFIG["device"])[:, None, None, None]).swapaxes(0, 1)
    if mask.shape[0] == 1:
        mask = mask.repeat(3, 1, 1)
    mask = draw_segmentation_masks(mask, mask_all_classes[0], colors=list(color_map.values()), alpha=1.)
    return mask

def rgb_to_class(
        mask_np: np.ndarray
) -> np.ndarray:
    """Receives a (H, W, 3) NumPy Array encoding an image and maps it to class indices.

    Args:
        mask_np: RGB image as NumPy array of shape (H, W, 3).

    Returns:
        Segmentation mask of class indices.
    """
    H, W, _ = mask_np.shape
    segmentation_mask = np.zeros((H, W), dtype=np.uint8)
    # Map RGB values to indices
    for rgb, index in get_inv_color_map_dict(with_void=False).items():
        mask = np.all(mask_np == rgb, axis=-1)  # Find where the RGB value matches
        segmentation_mask[mask] = index  # Set the corresponding index
    return segmentation_mask

def pil_to_class_array(
        mask: Image.Image
) -> str:
    """Converts a PIL Image mask to a string representation of class indices array.

    Args:
        mask: Input mask image.

    Returns:
        String representation of class indices array.
    """
    mask_array = np.array(mask)
    mask_array = rgb_to_class(mask_array)
    mask_array = str(mask_array.tolist())
    return mask_array
