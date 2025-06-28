from config import *

import numpy as np
import webcolors
from PIL import Image
import matplotlib.pyplot as plt
from data import CLASSES, NUM_CLASSES
import io

from config import *

import torch
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image

def create_pascal_label_colormap() -> np.ndarray[int]:
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

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

def get_color_map_dict(with_void: bool = False) -> dict[int, tuple[int, int, int]]:
    """Gets the color map as dictionary {cls_idx: (r, g, b)} for the 21 VOC classes.

    Returns:
        Dictionary mapping class index to RGB tuple.
    """
    color_map_list = full_color_map()[:21].tolist()
    if with_void:
        color_map_list += [full_color_map()[255].tolist()]
    return {i: tuple(rgb) for i, rgb in enumerate(color_map_list)}

COLOR_MAP_DICT = get_color_map_dict()
COLOR_MAP_VOID_DICT = get_color_map_dict(with_void=True)

def get_inv_color_map_dict(with_void: bool = False) -> dict[tuple[int, int, int], int]:
    """Gets the color map as dictionary {(r, g, b): cls_idx} for the 21 VOC classes.

    Returns:
        Dictionary mapping RGB tuple to class index.
    """
    color_map_dict = get_color_map_dict(with_void)
    inv_color_map_dict = {tuple(rgb): i for i, rgb in color_map_dict.items()}
    inv_color_map_dict[(255, 255, 255)] = 1 # to account for class-splitted prompts
    return inv_color_map_dict

def get_color_map_as_img(with_void: bool = False) -> Image.Image:
    """Gets the color map as RGB image for the 21 VOC classes.

    Returns:
        Image visualizing the color map.
    """
    labels = CLASSES.copy()
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


def get_color_map_as_rgb() -> str:
    """Gets the color map as dictionary {"cls [cls_idx]": (r, g, b)}.

    Returns:
        String representation of class name to RGB mapping.
    """
    color_map = get_color_map_dict()
    return str({f"{CLASSES[i]} [{i}]": rgb for i, rgb in color_map.items()})


def get_color_map_as_names() -> str:
    """Gets the color map as dictionary {"cls [cls_idx]": name}.

    Each color is associated to the name of the RGB nearest-neighbour to CSS3 colors.

    Returns:
        String representation of class name to closest CSS3 color name mapping.
    """
    def closest_color_name(rgb: tuple[int, int, int]) -> str:
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

    color_map = get_color_map_dict()
    c_names = {i: closest_color_name(rgb) for i, rgb in enumerate(color_map.values())}
    c_names[8] = "darkmaroon"
    return str({f"{CLASSES[i]} [{i}]": c_names[i] for i, rgb in enumerate(color_map.values())})


def get_color_map_as_patches(patch_size: tuple[int, int] = (32, 32)) -> tuple:
    """Gets the color map as list of patches of size 'patch_size'.

    Args:
        patch_size: Size of each patch. Defaults to (32, 32).

    Returns:
        Tuple of class name and PIL.Image.Image patch pairs.
    """
    color_map = get_color_map_dict()
    color_patches = []
    for color_index, rgb in color_map.items():
        # Create a new image with the specified size and fill it with the RGB color.
        patch = Image.new("RGB", patch_size, rgb)
        color_patches.append(f"{CLASSES[color_index]} [{color_index}]:")
        color_patches.append(patch)
    return tuple(color_patches)


def get_color_map_as(format: str):
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
    return fn()

def apply_colormap(
        mask: torch.Tensor,
        color_map: dict[int, tuple[int, int, int]],
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

def rgb_to_class(mask_np: np.ndarray) -> np.ndarray:
    """Receives a (H, W, 3) NumPy Array encoding an image and maps it to class indices.

    Args:
        mask_np: RGB image as NumPy array of shape (H, W, 3).

    Returns:
        Segmentation mask of class indices.
    """
    H, W, _ = mask_np.shape
    segmentation_mask = np.zeros((H, W), dtype=np.uint8)
    # Map RGB values to indices
    for rgb, index in get_inv_color_map_dict().items():
        mask = np.all(mask_np == rgb, axis=-1)  # Find where the RGB value matches
        segmentation_mask[mask] = index  # Set the corresponding index
    return segmentation_mask

# def pil_to_rbg_array(mask):
#     mask_array = np.array(mask)
#     mask_array = str(mask_array.tolist())
#     return mask_array

def pil_to_class_array(mask: Image.Image) -> str:
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


def main() -> None:
    print(get_inv_color_map_dict())

if __name__ == "__main__":
    main()
