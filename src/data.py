import json
import random
from glob import glob

import numpy as np
import xarray as xr
import pandas as pd

from path import *
from utils import *
from config import *

from torchvision.io import decode_image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL.Image import Image as PILImage
import base64
import io

CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT", "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE", "DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP", "SOFA", "TRAIN", "TVMONITOR"]

# define the class mappings
CLASS_MAP = {i: i for i in range(len(CLASSES))} # default mapping

NUM_CLASSES = len(list(CLASS_MAP.values())) # actual number of classes

def get_image_UIDs(
        path: Path,
        split="trainval"
) -> list[int]:
    """
    Lists the UIDs of the images stored into a certain path and by split.

    Args:
        path: root directory of the images.
        split: split type ('non-splitted' or 'class-splitted').
    
    Returns:
        List of image UIDs.

    Returns a list of image UIDs read in the "splits.txt" file for a specified split.
    """
    image_UIDs = []
    with open(path / f"{split}.txt", "r") as f:
        for line in f:
            image_id = line.strip()  # Remove any leading/trailing whitespace
            image_UIDs.append(image_id)
    to_shuffle = image_UIDs[23:]
    random.shuffle(to_shuffle)
    image_UIDs[23:] = to_shuffle
    return image_UIDs

image_UIDs = np.array(get_image_UIDs(SPLITS_PATH, split="trainval"))

def get_image(
        path: str
) -> torch.Tensor:
    """
    Reads a single image from disk and encodes it in a tensor.

    Args:
        path: Path to the image file.

    Returns:
        Image as a torch.Tensor on the global device.
    """
    img = decode_image(path)
    img = img.to(DEVICE)
    return img

def image_to_base64(
        img: PILImage
) -> str:
    """
    Converts a PIL image to a base64-encoded string.

    Args:
        img: PIL image to encode.

    Returns:
        Base64-encoded string of the image.
    """
    buffered = io.BytesIO()
    img.save(buffered, format=img.format or "PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

def one_hot_encode_masks(
        masks: torch.Tensor
) -> torch.Tensor:
    """
    One-hot encodes a tensor of masks.

    Args:
        masks: Tensor of masks to encode.

    Returns:
        One-hot encoded tensor of masks.
    """
    one_hot_masks = masks == torch.arange(NUM_CLASSES).to(DEVICE)[:, None, None, None]
    one_hot_masks = one_hot_masks.swapaxes(0, 1)
    return one_hot_masks

def resize_image_(
        img: torch.Tensor,
        image_size: int | tuple[int, int],
        mode: str
) -> torch.Tensor:
    """
    Resizes an image tensor to the given size and mode.

    Args:
        img: Image tensor to resize.
        image_size: Target size as int or tuple.
        mode: Interpolation mode.

    Returns:
        Resized image tensor.
    """
    img = (img/255.).unsqueeze(0)
    img = F.interpolate(img, size=image_size, mode=mode)
    img = (img.squeeze(0)*255).clamp(0, 255).byte()
    return img

def resize_image(
        img: torch.Tensor,
        image_size: int | tuple[int, int],
        mode: str
) -> torch.Tensor:
    """
    Resizes an image tensor to the given size and mode, preserving aspect ratio if needed.

    Args:
        img: Image tensor to resize.
        image_size: Target size as int or tuple.
        mode: Interpolation mode.

    Returns:
        Resized image tensor.
    """
    img = (img / 255.).unsqueeze(0)  # shape (1, C, H, W)
    
    if isinstance(image_size, int): # 'image_size' is of type int
        _, _, h, w = img.shape
        if h < w:
            new_h = image_size
            new_w = int(w * image_size / h)
        else:
            new_w = image_size
            new_h = int(h * image_size / w)
        size = (new_h, new_w)
    else:
        size = image_size  # 'image_size' is of type tuple[int, int]
        
    img = F.interpolate(img, size=size, mode=mode, align_corners=False if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None)
    img = (img.squeeze(0) * 255).clamp(0, 255).byte()
    return img

def resize_mask(
        mask: torch.Tensor,
        image_size: None | tuple[int, int] | int,
        mode: str,
        center_crop: bool
) -> torch.Tensor:
    """
    Resizes a mask tensor using the specified mode.

    Args:
        mask: Mask tensor to resize.
        image_size: Target size as int, tuple, or None.
        mode: Interpolation mode ('bilinear' or 'nearest').
        center_crop: Whether to center crop after resizing.

    Returns:
        Resized mask tensor.
    """
    if mode not in ["bilinear", "nearest"]:
        raise AttributeError("Resizing mode must be 'bilinear' or 'nearest'.")
    if mode == "bilinear":
        one_hot_mask = one_hot_encode_masks(mask.unsqueeze(0)).squeeze(0)
        one_hot_mask = one_hot_mask.float()
        resized_mask = resize_image(one_hot_mask, image_size, "bilinear").argmax(dim=0, keepdim=True).byte()
    elif mode == "nearest":
        resized_mask = resize_image(mask, image_size, "nearest")
    return resized_mask

def get_sc(
        path: Path,
        image_size: None | int | tuple[int, int] = None,
        center_crop: bool = True
) -> torch.Tensor:
    """
    Loads and optionally resizes and center-crops an image.

    Args:
        path: Path to the image file.
        image_size: Target size as int, tuple, or None.
        center_crop: Whether to center crop the image.

    Returns:
        Processed image tensor.
    """
    sc = get_image(path)
    if image_size is not None:
        sc = resize_image(sc, image_size, mode="bilinear")
    if center_crop:
        sc = TF.center_crop(sc, output_size=min(sc.shape[1:]))
    return sc

def apply_class_map(
        mask: torch.Tensor,
        class_map: dict
) -> torch.Tensor:
    """
    Applies a class mapping to a mask tensor.

    Args:
        mask: Mask tensor to map.
        class_map: Dictionary mapping class indices.

    Returns:
        Mask tensor with mapped classes.
    """
    mask_ = mask.cpu()
    mask_.apply_(lambda x: class_map.get(x, 0)) # class mapping
    mask = mask_.to(DEVICE)
    return mask

def _get_mask(
        path: str,
        class_map: dict,
        image_size: int | tuple[int, int] | None,
        resize_mode: str,
        center_crop: bool
) -> torch.Tensor:
    """
    Loads a mask, applies class mapping, resizes, and optionally center-crops it.

    Args:
        path: Path to the mask file.
        class_map: Dictionary mapping class indices.
        image_size: Target size as int, tuple, or None.
        resize_mode: Interpolation mode.
        center_crop: Whether to center crop the mask.

    Returns:
        Processed mask tensor.
    """
    mask = get_image(path)
    class_map_ = class_map.copy()
    class_map_[255] = 0 # additionally, map UNLABELLED to BACKGROUND
    mask = apply_class_map(mask, class_map_)
    if image_size is not None:
        mask = resize_mask(mask, image_size, resize_mode, center_crop=True)
    if center_crop:
        mask = TF.center_crop(mask, output_size=min(mask.shape[1:]))
    return mask

def get_gt(
        path: Path,
        class_map: dict,
        image_size: None | tuple[int, int] | int = None,
        resize_mode: str = "nearest",
        center_crop: bool = True
) -> torch.Tensor:
    """
    Loads and processes a ground truth mask.

    Args:
        path: Path to the mask file.
        class_map: Dictionary mapping class indices.
        image_size: Target size as int, tuple, or None.
        resize_mode: Interpolation mode.
        center_crop: Whether to center crop the mask.

    Returns:
        Processed ground truth mask tensor.
    """
    return _get_mask(path, class_map, image_size, resize_mode, center_crop)

def get_pr(
        path: str,
        class_map: dict,
        image_size: int | tuple[int, int] | None = None,
        resize_mode: str = "nearest",
        center_crop: bool = True
) -> torch.Tensor:
    """
    Loads and processes a predicted mask.

    Args:
        path: Path to the mask file.
        class_map: Dictionary mapping class indices.
        image_size: Target size as int, tuple, or None.
        resize_mode: Interpolation mode.
        center_crop: Whether to center crop the mask.

    Returns:
        Processed predicted mask tensor.
    """
    return _get_mask(path, class_map, image_size, resize_mode, center_crop)

def get_significant_classes(
        path: str,
        image_size: int | tuple[int, int],
        class_map: dict
) -> list[int]:
    """
    Returns the list of significant (non-background) classes present in a mask.

    Args:
        path: Path to the mask file.
        image_size: Target size as int or tuple.
        class_map: Dictionary mapping class indices.

    Returns:
        List of significant class indices.
    """
    mask = _get_mask(path, class_map, image_size, resize_mode="nearest", center_crop=True)
    significant_classes = mask.unique().tolist() # classes that actually appear in 'gt'
    significant_classes.remove(0)
    return significant_classes

def read_txt(
        txt_path: str
) -> str:
    """
    Reads the contents of a text file.

    Args:
        txt_path: Path to the text file.

    Returns:
        Contents of the file as a string.
    """
    with open(txt_path, "r") as f:
        content = f.read()
        return content
    raise FileNotFoundError(f"Error when attempting to read '{txt_path}'")

def _read_one_jsonl_line(
        line: str
) -> dict:
    """
    Parses a single line of JSONL into a dictionary.

    Args:
        line: Line from a JSONL file.

    Returns:
        Parsed dictionary.
    """
    obj = json.loads(line)
    return obj

def is_state(
        obj: dict
) -> bool:
    """
    Checks if a dictionary represents a state object.

    Args:
        obj: Dictionary to check.

    Returns:
        True if the object is a state, False otherwise.
    """
    return "state" in obj.keys()

def read_state(
        path: str
) -> dict | None:
    """
    Reads the first line of a JSONL file and returns it if it is a state object.

    Args:
        path: Path to the JSONL file.

    Returns:
        State object dictionary or None.
    """
    with open(path, 'r') as file:
        first_line = file.readline()
        state_obj = _read_one_jsonl_line(first_line)    
    return state_obj if is_state(state_obj) else None

def read_many_from_jsonl(
        path: str
) -> list[dict]:
    """
    Reads all objects from a JSONL file, skipping the state if present.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of objects from the file.
    """
    with open(path, 'r') as file:
        data = list(map(lambda l: _read_one_jsonl_line(l), file))
    return data[1:] if is_state(data[0]) else data

def _append_one_to_jsonl(
        object_to_append: dict,
        file
) -> None:
    """
    Appends a single object to an open JSONL file.

    Args:
        object_to_append: Dictionary to append.
        file: Open file object.
    """
    json.dump(object_to_append, file)
    file.write('\n') # ensure to write a newline after each JSONL object

def append_many_to_jsonl(
        path: str,
        objects_to_append: list[dict]
) -> None:
    """
    Appends multiple objects to a JSONL file.

    Args:
        path: Path to the JSONL file.
        objects_to_append: List of dictionaries to append.
    """
    with open(path, 'a+') as file:
        list(map(lambda obj: _append_one_to_jsonl(obj, file), objects_to_append))

def read_one_from_jsonl_by(
        path: str,
        key: str,
        value
) -> dict | None:
    """
    Reads the first object from a JSONL file where the given key matches the value.

    Args:
        path: Path to the JSONL file.
        key: Key to match in the object.
        value: Value to match for the key.

    Returns:
        The first matching object, or None if not found.
    """
    # TODO maybe this method can be made faster
    """
    'key' must be at the top level of the JSON object (no nested keys).
    """
    with open(path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            if obj.get(key, None) == value:
                return obj

def _format_one_from_jsonl(
        obj: dict
) -> dict:
    """
    Formats a single JSONL object for output.

    Args:
        obj: JSONL object to format.

    Returns:
        Formatted dictionary.
    """
    if is_state(obj):
        return obj
    else:
        return {obj["img_idx"]: obj["content"]}

def _format_many_from_jsonl(
        obj_list: list[dict]
) -> dict:
    """
    Formats a list of JSONL objects into a dictionary.

    Args:
        obj_list: List of JSONL objects.

    Returns:
        Dictionary mapping image indices to content.
    """
    return {line["img_idx"]: line["content"] for line in obj_list}

def get_one_item(
        path: str,
        idx,
        return_state: bool
) -> dict:
    """
    Gets a single item from a JSONL file by index, optionally merging with state.

    Args:
        path: Path to the JSONL file.
        idx: Index of the item to retrieve.
        return_state: Whether to merge with the state object.

    Returns:
        The requested item, optionally merged with state.
    """
    state = read_state(path)
    item = read_one_from_jsonl_by(path, "img_idx", idx)
    item = _format_one_from_jsonl(item)
    return state | item if return_state else item

def get_many_item(
        path: Path,
        return_state: bool = False,
        format_to_dict: bool = False
) -> dict | tuple[dict, dict]:
    """
    Gets multiple items from a JSONL file, optionally merging with state and formatting as a dictionary.

    Args:
        path: Path to the JSONL file.
        return_state: Whether to return the state object as well.
        format_to_dict: Whether to format the output as a dictionary.

    Returns:
        Items from the file, and optionally the state object.
    """
    state = read_state(path)
    items = read_many_from_jsonl(path)
    items = _format_many_from_jsonl(items) if format_to_dict else items
    if return_state:
        return items, state
    else:
        return items

def get_one_answer_gt(
        by_model,
        idx,
        return_state: bool = False
) -> dict:
    """
    Gets the ground truth answer for a single image by model and index.

    Args:
        by_model: Model identifier.
        idx: Image index.
        return_state: Whether to include the state in the result.

    Returns:
        Ground truth answer dictionary.
    """
    answer_gt = get_one_item(get_answer_gts_path(by_model), idx, return_state)
    return answer_gt

def get_one_sup_set_answer_gt(
        by_model,
        idx,
        return_state: bool = False
) -> dict:
    """
    Gets the ground truth answer for a single image from the support set by model and index.

    Args:
        by_model: Model identifier.
        idx: Image index.
        return_state: Whether to include the state in the result.

    Returns:
        Ground truth answer dictionary.
    """
    answer_gt = get_one_item(get_sup_set_answer_gts_path(by_model), idx, return_state)
    return answer_gt

def get_one_answer_pr(
        by_model,
        split_by,
        relative_path,
        idx,
        return_state: bool = False
) -> dict:
    """
    Gets the predicted answer for a single image by model, split, and index.

    Args:
        by_model: Model identifier.
        split_by: Split identifier.
        relative_path: Relative path to the image.
        idx: Image index.
        return_state: Whether to include the state in the result.

    Returns:
        Predicted answer dictionary.
    """
    answer_pr = get_one_item(get_answer_prs_path(by_model, split_by, relative_path), idx, return_state)
    return answer_pr

def get_many_answer_gt(
        by_model,
        return_state: bool = False
) -> dict:
    """
    Gets ground truth answers for multiple images by model.

    Args:
        by_model: Model identifier.
        return_state: Whether to include the state in the result.

    Returns:
        Dictionary of ground truth answers.
    """
    answer_gts = get_many_item(get_answer_gts_path(by_model), return_state)
    return answer_gts

def get_many_answer_pr(
        path: Path,
        return_state: bool = False
) -> dict:
    """
    Gets predicted answers for multiple images from a JSONL file.

    Args:
        path: Path to the JSONL file.
        return_state: Whether to include the state in the result.

    Returns:
        Dictionary of predicted answers.
    """
    answer_prs = get_many_item(path, return_state)
    return answer_prs

def get_one_eval_gt(
        by_model: str,
        split_by: str,
        idx: int,
        return_state: bool = False
) -> dict:
    """
    Gets the evaluation ground truth for a single image by model, split, and index.

    Args:
        by_model: Model identifier.
        split_by: Split identifier.
        idx: Image index.
        return_state: Whether to include the state in the result.

    Returns:
        Evaluation ground truth dictionary.
    """
    eval_gt = get_one_item(get_eval_gts_path(by_model, split_by), idx, return_state)
    return eval_gt

def get_one_eval_pr(
        by_model: str,
        split_by: str,
        relative_path: Path,
        idx: int,
        return_state: bool = False
) -> dict:
    """
    Gets the evaluation predicted answer for a single image by model, split, and index.

    Args:
        by_model: Model identifier.
        split_by: Split identifier.
        relative_path: Relative path to the image.
        idx: Image index.
        return_state: Whether to include the state in the result.

    Returns:
        Evaluation predicted answer dictionary.
    """
    eval_pr = get_one_item(get_eval_prs_path(by_model, split_by, relative_path), idx, return_state)
    return eval_pr

def get_many_eval_gt(
        by_model: str,
        split_by: str,
        return_state: bool = False
) -> dict:
    """
    Gets evaluation ground truths for multiple images by model and split.

    Args:
        by_model: Model identifier.
        split_by: Split identifier.
        return_state: Whether to include the state in the result.

    Returns:
        Dictionary of evaluation ground truths.
    """
    eval_gts = get_many_item(get_eval_gts_path(by_model, split_by), return_state, )
    return eval_gts

def get_many_eval_pr(
        path: Path,
        return_state: bool = False,
        format_to_dict: bool = False
) -> dict:
    """
    Gets evaluation predicted answers for multiple images from a JSONL file.

    Args:
        path: Path to the JSONL file.
        return_state: Whether to include the state in the result.
        format_to_dict: Whether to format the output as a dictionary.

    Returns:
        Dictionary of evaluation predicted answers.
    """
    return get_many_item(path, return_state, format_to_dict)
    
def format_many_to_jsonl(
        objs: dict
) -> list[dict]:
    """
    Formats multiple objects for appending to a JSONL file.

    Args:
        objs: Dictionary of objects to format.

    Returns:
        List of formatted objects.
    """
    objs_list = [{"state": objs["state"]}]
    objs_list.extend([{"img_idx": img_idx, "content": content} for img_idx, content in list(objs.items())[1:]])
    return objs_list

def expand_words_to_variants(
        word: str
) -> list[str]:
    """
    Expands a word in a list of similar word to involve for the pertinence check
    """
    return [word, word.lower()]

def validate_pertinence(
        sentences: list[str],
        significant_classes: list[int]
) -> None:
    """
    Asserts if predicted answers only contain upper case words (as all and only class names should be) related to their positive class.
    The positive class name needs to be there and all other cannot.
    """
    for s, pos_class in zip(sentences, significant_classes):
        reason_upper_words = extract_uppercase_words(s)
        pos_class_name = CLASSES[pos_class]
        allowed_class_names = ["BACKGROUND", pos_class_name]
        forbidden_class_names = flatten_list([expand_words_to_variants(cn) for cn in CLASSES if cn not in allowed_class_names])
        assert all(word != fw for word in reason_upper_words for fw in forbidden_class_names), f"Forbidden words found in answer of pos. class '{pos_class_name}'"
        assert [s in reason_upper_words for s in allowed_class_names] , f"Allowed words '{allowed_class_names}' not found in answer '{pos_class_name}'"

def describe_da(
        data_da: xr.DataArray,
        dims_to_agg: list[str]
) -> pd.DataFrame:
    data_da = data_da.astype("float")
    stats = {
        "mean": data_da.mean(dim=dims_to_agg),
        "std": data_da.std(dim=dims_to_agg, ddof=1),
        "min": data_da.min(dim=dims_to_agg),
        "max": data_da.max(dim=dims_to_agg),
    }
    # Convert to DataArrays and stack
    df = xr.concat(stats.values(), dim="stat").assign_coords(stat=list(stats)).to_pandas().transpose()
    if len(df.shape) > 2: raise AttributeError 
    return df

def compute_results_da(
        exp_path: Path
) -> xr.DataArray:
    data_da = None

    var_paths = glob(f"{exp_path}/*.jsonl")
    var_names = [os.path.splitext(os.path.basename(path))[0] for path in var_paths]

    for var_n, var_p in zip(var_names, var_paths):
        
        eval_prs = get_many_eval_pr(var_p, return_state=False, format_to_dict=True)
        prs_per_img_idx_df = pd.DataFrame.from_dict(eval_prs, orient='index')
        prs_per_img_idx_df = prs_per_img_idx_df.sort_index().sort_index(axis=1)

        prs_per_img_idx_pred_df = (prs_per_img_idx_df["pred"] == "correct").astype(float)
        prs_per_img_idx_score_df = prs_per_img_idx_df["score"]
        prs_per_img_idx_reason_df = prs_per_img_idx_df["reason"]

        prs_per_img_idx_da = xr.DataArray(
            [prs_per_img_idx_pred_df, prs_per_img_idx_reason_df, prs_per_img_idx_score_df],
            coords=[["pred", "reason", "score"], prs_per_img_idx_df.index],
            dims=["metric", "img_idx"]
        ).transpose("img_idx", "metric")
        
        if data_da is None:
            coords = [var_names, prs_per_img_idx_df.index, ["pred", "reason", "score"]] # indexes names
            sorted_coords = [sorted(dim_values) for dim_values in coords]
            dims = ["var", "img_idx", "metric"] # dimensions names
            shape = [len(l) for l in sorted_coords]
            data_da = xr.DataArray(np.empty(shape, dtype=object), coords=sorted_coords, dims=dims)

        data_da.loc[var_n] = prs_per_img_idx_da

    return data_da

def compute_results_da_class_splitted(
        exp_path: Path
) -> xr.DataArray:
    data_da = None

    var_paths = glob(f"{exp_path}/*.jsonl")
    var_names = [os.path.splitext(os.path.basename(path))[0] for path in var_paths]

    for var_n, var_p in zip(var_names, var_paths):
        
        eval_prs = get_many_eval_pr(var_p, return_state=False, format_to_dict=True)
        prs_per_img_idx_df = pd.DataFrame.from_dict(eval_prs, orient='index')
        
        for column in [str(n) for n in range(0, NUM_CLASSES)]:
            if column not in prs_per_img_idx_df.columns:
                prs_per_img_idx_df[column] = pd.NA
        prs_per_img_idx_df.columns = [int(s) for s in prs_per_img_idx_df.columns]
        prs_per_img_idx_df = prs_per_img_idx_df.sort_index().sort_index(axis=1)

        prs_per_img_idx_pred_df = prs_per_img_idx_df.map(lambda x: x["pred"] == "correct" if type(x) == dict else None).astype(float)
        prs_per_img_idx_score_df = prs_per_img_idx_df.map(lambda x: x["score"] if type(x) == dict else None)
        prs_per_img_idx_reason_df = prs_per_img_idx_df.map(lambda x: x["reason"] if type(x) == dict else None)

        prs_per_img_idx_da = xr.DataArray(
            [prs_per_img_idx_pred_df, prs_per_img_idx_reason_df, prs_per_img_idx_score_df],
            coords=[["pred", "reason", "score"], prs_per_img_idx_df.index, prs_per_img_idx_df.columns],
            dims=["metric", "img_idx", "pos_class"]
        ).transpose("img_idx", "pos_class", "metric")
        
        if data_da is None:
            coords = [var_names, prs_per_img_idx_df.index, prs_per_img_idx_df.columns, ["pred", "score", "reason"]] # indexes names
            sorted_coords = [sorted(dim_values) for dim_values in coords]
            dims = ["var", "img_idx", "pos_class", "metric"] # dimensions names
            shape = [len(l) for l in sorted_coords]
            data_da = xr.DataArray(np.empty(shape, dtype=object), coords=sorted_coords, dims=dims)

        data_da.loc[var_n] = prs_per_img_idx_da

    return data_da

def main() -> None:
    da = create_empty_dataarray({"n": [0, 2, 5], "type": ["a", "b"]})
    da.loc[0, "a"] = 4
    print(da)

if __name__ == "__main__":
    main()
