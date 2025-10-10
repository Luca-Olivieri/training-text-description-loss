"""Color map utilities for segmentation visualization and processing.

This module provides functions to generate, transform, and apply color maps
for semantic segmentation tasks. It includes utilities for:
- Generating full color maps with customizable number of colors
- Converting between RGB colors and class indices
- Visualizing color maps in various formats (images, patches, names)
- Applying color maps to segmentation masks
"""

from core.config import *
from core.torch_utils import is_list_of_tensors

import matplotlib.pyplot as plt
import io
import webcolors

from core._types import RGB_tuple

def full_color_map(
        N: int = 256,
        normalized: bool = False
) -> np.ndarray[int]:
    """Generates a full color map with N distinct colors using bit manipulation.
    
    This function creates a color map using a deterministic bit-shifting algorithm
    to ensure maximally distinct colors. This is commonly used for visualization
    of semantic segmentation masks where each class needs a unique color.

    Args:
        N: The number of colors to generate. Defaults to 256.
        normalized: If True, colors are float values in range [0.0, 1.0].
                   If False, colors are uint8 integers in range [0, 255].
                   Defaults to False.
    
    Returns:
        NumPy array of shape (N, 3) containing RGB color values for each class.
        dtype is 'float32' if normalized=True, 'uint8' otherwise.
    """
    def bitget(byteval: int, idx: int) -> bool:
        """Extracts the bit value at a specific index from a byte value.
        
        Args:
            byteval: The byte value to extract from.
            idx: The bit index to extract (0-7).
            
        Returns:
            True if the bit at the specified index is 1, False otherwise.
        """
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
    """Inverts a color map dictionary to map RGB tuples to class indices.
    
    Creates an inverse mapping from RGB color tuples to their corresponding
    class indices. Also adds a special mapping for white (255, 255, 255) to
    class index 1 to support class-splitted prompts.

    Args:
        color_map_dict: Dictionary mapping class indices to RGB tuples.

    Returns:
        Dictionary mapping RGB tuples to class indices, with an additional
        entry mapping (255, 255, 255) to class index 1.
    """
    inv_color_map_dict = {tuple(rgb): i for i, rgb in color_map_dict.items()}
    inv_color_map_dict[(255, 255, 255)] = 1 # to account for class-splitted prompts
    return inv_color_map_dict

def get_color_map_as_img(
        classes: list[str],
        with_void: bool = False,
) -> Image.Image:
    """Creates a visual representation of the color map as an RGB image.
    
    Generates an image displaying each class with its corresponding color
    and label. Each class is shown as a colored horizontal bar with its
    name and index.

    Args:
        classes: List of class names to visualize.
        with_void: If True, includes the VOID class (index 255) in the
                  visualization. Defaults to False.

    Returns:
        PIL Image showing the color map with labeled colored bars for each class.
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
    """Generates a string representation of the color map with RGB values.
    
    Creates a dictionary-style string mapping class names (with indices) to
    their RGB color tuples.

    Args:
        classes: List of class names.
        color_map_dict: Dictionary mapping class indices to RGB tuples.

    Returns:
        String representation in format {"class_name [index]": (r, g, b), ...}.
    """
    return str({f"{classes[i]} [{i}]": rgb for i, rgb in color_map_dict.items()})

def get_color_map_as_names(
        classes: list[str],
        color_map_dict: dict[int, RGB_tuple],
) -> str:
    """Generates a string representation of the color map with human-readable color names.
    
    Each RGB color is mapped to its closest CSS3 color name using Euclidean distance
    in RGB space. This provides more intuitive color descriptions than raw RGB values.

    Args:
        classes: List of class names.
        color_map_dict: Dictionary mapping class indices to RGB tuples.

    Returns:
        String representation in format {"class_name [index]": "color_name", ...},
        where color_name is the closest matching CSS3 color name.
    """
    def closest_color_name(
            rgb: RGB_tuple
    ) -> str:
        """Finds the closest CSS3 color name for a given RGB tuple.
        
        Uses Euclidean distance in RGB space to find the nearest CSS3 color.
        If an exact match exists, returns it immediately.
        
        Args:
            rgb: RGB color tuple to match.
            
        Returns:
            The name of the closest CSS3 color.
        """
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
    """Generates color patches for each class in the color map.
    
    Creates small colored image patches that can be used for visualization
    or as visual elements in UIs. Each class gets a solid-colored patch.

    Args:
        classes: List of class names.
        color_map_dict: Dictionary mapping class indices to RGB tuples.
        patch_size: Size (width, height) of each color patch in pixels.
                   Defaults to (32, 32).

    Returns:
        Tuple alternating between class name strings (format "class_name [index]:")
        and PIL Image patches filled with the corresponding class color.
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
    """Retrieves the color map in a specified format.
    
    This is a dispatcher function that calls the appropriate color map
    generation function based on the requested format.

    Args:
        format: The desired output format. Must be one of:
               - 'img': Returns a PIL Image visualization (see get_color_map_as_img)
               - 'rgb': Returns string of class to RGB mapping (see get_color_map_as_rgb)
               - 'names': Returns string of class to color names (see get_color_map_as_names)
               - 'patches': Returns tuple of color patches (see get_color_map_as_patches)
        **kwargs: Additional keyword arguments passed to the specific format function.
                 Required kwargs depend on the chosen format.

    Returns:
        The color map in the requested format. Return type depends on format parameter.
        
    Raises:
        KeyError: If format is not one of the supported types.
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
    """Applies a color map to class label tensors to produce RGB image tensors.

    Converts integer class label masks to colored RGB images using a lookup table
    (LUT) approach. This function is highly optimized for GPU acceleration and
    uses vectorized operations for fast batch processing.
    
    Any class label not present in the color_map will be mapped to black (0, 0, 0)
    by default.

    Args:
        input_tensor: Either a single tensor of shape [B, 1, H, W] containing
                     integer class labels, or a list of such tensors (which will
                     be stacked along the batch dimension).
        color_map: Dictionary mapping class indices (int) to RGB color tuples,
                  where each tuple contains three integers in range [0, 255].

    Returns:
        Tensor of shape [B, 3, H, W] with dtype torch.uint8, representing the
        colored segmentation masks. The three channels correspond to R, G, B.
        
    Raises:
        TypeError: If input_tensor is neither a torch.Tensor nor a list of tensors.
        ValueError: If input_tensor does not have shape [B, 1, H, W].
        
    Note:
        The function automatically handles both CPU and CUDA tensors, placing
        the lookup table on the same device as the input.
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

def rgb_to_class(
        mask_np: np.ndarray
) -> np.ndarray:
    """Converts an RGB segmentation mask to class index representation.
    
    Takes a colored segmentation mask and converts each RGB pixel to its
    corresponding class index using the inverse color map. This is the
    inverse operation of applying a color map.

    Args:
        mask_np: RGB segmentation mask as NumPy array of shape (H, W, 3)
                with uint8 values in range [0, 255].

    Returns:
        NumPy array of shape (H, W) with dtype uint8, where each pixel
        contains the class index corresponding to the RGB color in the
        input mask.
        
    Note:
        This function internally uses get_inv_color_map_dict to obtain
        the RGB-to-class mapping.
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
    """Converts a PIL Image segmentation mask to a string of class indices.
    
    Takes a colored PIL Image mask, converts it to class indices, and returns
    the result as a string representation of a nested list. This is useful for
    serialization or text-based logging of segmentation masks.

    Args:
        mask: PIL Image containing an RGB segmentation mask.

    Returns:
        String representation of a 2D list of class indices, obtained by
        calling str() on the result of mask.tolist().
        
    Example:
        For a 2x2 mask, might return: "[[0, 1], [2, 0]]"
    """
    mask_array = np.array(mask)
    mask_array = rgb_to_class(mask_array)
    mask_array = str(mask_array.tolist())
    return mask_array
