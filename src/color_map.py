import numpy as np
import webcolors
from PIL import Image
import matplotlib.pyplot as plt
from data import CLASSES, NUM_CLASSES
from utils import DEVICE
import io

import torch
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image

def _full_color_map(N=256, normalized=False):
    """
    Generate a full color map with N colors (integer or normalised to float).
    """
    def bitget(byteval, idx):
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

def get_color_map_dict():
    """
    Get the color map as dictionary {cls_idx: (r, g, b)} for the 21 VOC classes.
    """
    color_map_list = _full_color_map()[:21].tolist()
    return {i: tuple(rgb) for i, rgb in enumerate(color_map_list)}

COLOR_MAP_DICT = get_color_map_dict()

def get_inv_color_map_dict():
    """
    Get the color map as dictionary {(r, g, b): cls_idx} for the 21 VOC classes.
    """
    color_map_dict = get_color_map_dict()
    inv_color_map_dict = {tuple(rgb): i for i, rgb in color_map_dict.items()}
    inv_color_map_dict[(255, 255, 255)] = 1 # to account for class-splitted prompts
    return inv_color_map_dict

def get_color_map_as_img():
    """
    Get the color map as RGB image for the 21 VOC classes.
    """
    labels = CLASSES.copy()
    labels.append("VOID")

    labels = [f"{l} [{i}]" for i, l in enumerate(labels)]

    nclasses = 21
    row_size = 50
    col_size = 500 # 500 default
    cmap = _full_color_map()
    array = np.empty((row_size*(nclasses), col_size, cmap.shape[1]), dtype=cmap.dtype)
    for i in range(nclasses):
        array[i*row_size:i*row_size+row_size, :] = cmap[i]

    fig, ax = plt.subplots(figsize=(6, 12))
    ax.imshow(array)
    ax.set_yticks([row_size * i + row_size / 2 for i in range(nclasses)])
    labels.remove("VOID [21]")
    ax.set_yticklabels(labels)
    ax.set_xticks([])
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def get_color_map_as_rgb():
    """
    Get the color map as dictionary {"cls [cls_idx]": (r, g, b)}
    """
    color_map = get_color_map_dict()
    return str({f"{CLASSES[i]} [{i}]": rgb for i, rgb in color_map.items()})

def get_color_map_as_names():
    """
    Get the color map as dictionary {"cls [cls_idx]": name}
    Each colors is associated to the name of RGB nearest-neighbour to CSS3 colors.
    """
    def closest_color_name(rgb):
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

def get_color_map_as_patches(patch_size=(32, 32)):
    """
    Get the color map as list of patches of size 'patch_size'. 
    """
    color_map = get_color_map_dict()
    color_patches = []
    for color_index, rgb in color_map.items():
        # Create a new image with the specified size and fill it with the RGB color.
        patch = Image.new("RGB", patch_size, rgb)
        color_patches.append(f"{CLASSES[color_index]} [{color_index}]:")
        color_patches.append(patch)
    return tuple(color_patches)

def get_color_map_as(format):
    format2fn = {
        "img": get_color_map_as_img,
        "rgb": get_color_map_as_rgb,
        "names": get_color_map_as_names,
        "patches": get_color_map_as_patches
        }    
    fn = format2fn[format]
    return fn()

def apply_colormap(mask, color_map):
    """
    Receives a tensor of shape [C, H, W] and returns a PIL Image with the color map applied.
    """
    assert len(mask.shape) == 3, mask.shape
    mask_all_classes = (mask == torch.arange(NUM_CLASSES).to(DEVICE)[:, None, None, None]).swapaxes(0, 1)
    if mask.shape[0] == 1:
        mask = mask.repeat(3, 1, 1)
    mask = draw_segmentation_masks(mask, mask_all_classes[0], colors=list(color_map.values()), alpha=1.)
    mask = to_pil_image(mask)
    return mask

def rgb_to_class(mask_np):
    """
    Receives a (H, W, 3) NumPy Array encoding an image and maps it to class indices.
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

def pil_to_class_array(mask):
    mask_array = np.array(mask)
    mask_array = rgb_to_class(mask_array)
    mask_array = str(mask_array.tolist())
    return mask_array


def main() -> None:
    print(get_inv_color_map_dict())

if __name__ == "__main__":
    main()
