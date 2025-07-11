from config import *
from path import get_answer_gts_path, SCS_PATH, GTS_PATH, get_sup_set_answer_gts_path, get_answer_prs_path, SPLITS_PATH, get_eval_gts_path, get_eval_prs_path
from utils import map_tensor, extract_uppercase_words, flatten_list

import json
import random
from glob import glob
import io
from collections import Counter

import numpy as np
import xarray as xr
import pandas as pd

import math
import torch
from torch import Tensor
from torchvision.io import decode_image
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from torchvision import tv_tensors
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import base64
import nltk
from pathlib import Path

from typing import Literal, Callable, Optional

CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT", "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE", "DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP", "SOFA", "TRAIN", "TVMONITOR"]
CLASSES_VOID = CLASSES + ["UNLABELLED"]

# define the class mappings
CLASS_MAP: dict[int, int] = {i: i for i in range(len(CLASSES))} | {255: 0} # default mapping
CLASS_MAP_VOID: dict[int, int] = CLASS_MAP | {255: 21}

NUM_CLASSES = len(set(CLASS_MAP.values())) # actual number of classes
NUM_CLASSES_VOID = len(set(CLASS_MAP_VOID.values())) # actual number of classes

class MyDataset(Dataset):
    vlm_image_size: int | tuple[int, int] = None
    seg_image_size: int | tuple[int, int] = None

    def get_image_uids(self) -> list[int]:
        ...

class SegDataset(Dataset):
    """
    TODO
    """
    def __init__(
            self,
            uids: list[int],
            resize_size: int | list[int, int],
            class_map: dict,
            mask_resize_mode: str = "nearest",
    ) -> None:
        self.scs_paths = [SCS_PATH / (UID + ".jpg") for UID in uids]
        self.gts_paths = [GTS_PATH / (UID + ".png") for UID in uids]
        self.resize_size = resize_size
        self.class_map = class_map
        self.mask_resize_mode = mask_resize_mode

    def __len__(self) -> int:
        return len(self.gts_paths)

    def __getitem__(
            self,
            idx: int
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor]:
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            scs = [get_sc(path=self.scs_paths[i], resize_size=self.resize_size, center_crop=False) for i in indices]
            gts = [get_gt(path=self.gts_paths[i], class_map=self.class_map, resize_size=self.resize_size, center_crop=False) for i in indices]
            return scs, gts
        else:
            sc = get_sc(path=self.scs_paths[idx], resize_size=self.resize_size, center_crop=False)
            gt = get_gt(path=self.gts_paths[idx], class_map=self.class_map, resize_size=self.resize_size, center_crop=False)
            return sc, gt

# TODO create a wrapper Dataset class that iterates on both SegDataset and JSONLDataset (each batch is composed by a batch of both datasets).


class JSONLDataset(Dataset):
    """
    An efficient PyTorch Dataset for lazy-loading and parsing large JSONL (JSON Lines) files.

    This class avoids loading the entire file into memory. Instead, it creates an
    index of the byte offsets for each line during initialization. When an item
    is requested, it seeks directly to the corresponding line's position in the
    file and reads only that line.

    This makes it highly memory-efficient and fast for random access, which is
    essential for use with PyTorch's DataLoader, especially with `shuffle=True`.

    It also supports slicing (e.g., `dataset[10:20]`).

    Args:
        file_path (Path): The path to the JSONL file.
        transform (Optional[Callable]): An optional function to be applied to the
            data object after it is loaded and parsed.
    """
    def __init__(
            self,
            file_path: Path,
            transform: Optional[Callable] = None
    ):
        super().__init__()
        self.file_path = file_path
        self.transform = transform
        self._file = None  # File handle will be opened lazily by each worker
        self.state_line = None

        # Create an index of byte offsets for each line
        self.line_offsets = self._build_index()

    def _build_index(self) -> list[int]:
        """
        Scans the file once to create an index of the starting byte offset
        for each line. If the first line is a state object, it's stored
        and not included in the data indices.
        """
        line_offsets = []
        # Open in binary mode to correctly handle byte offsets
        with open(self.file_path, 'rb') as f:
            # Check first line for state
            first_line_offset = f.tell()
            first_line = f.readline()

            if not first_line:
                return []

            try:
                # Need to decode for json.loads
                state_candidate = json.loads(first_line.decode('utf-8'))
                if is_state(state_candidate):
                    self.state_line = state_candidate
                    # State found, start indexing from the next line
                    offset = f.tell()
                else:
                    # No state, first line is data
                    line_offsets.append(first_line_offset)
                    offset = f.tell()
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If it's not valid JSON or can't be decoded, treat as data
                line_offsets.append(first_line_offset)
                offset = f.tell()

            while True:
                line = f.readline()
                if not line:
                    break
                line_offsets.append(offset)
                offset = f.tell()
        return line_offsets

    def __len__(self) -> int:
        """Returns the total number of lines in the file."""
        return len(self.line_offsets)

    def _get_single_item(
            self,
            idx: int
    ) -> dict[str, Any]:
        """
        Retrieves, parses, and optionally transforms a single item by its index.
        This is the core logic for data retrieval.
        """
        # Handle negative indices
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} is out of range for a dataset of size {len(self)}.")

        # Lazily open the file handle in each worker process
        if self._file is None:
            # We open in text mode here for json.loads, which expects strings.
            # The offsets were calculated in binary mode for accuracy.
            self._file = open(self.file_path, 'r', encoding='utf-8')

        # Seek to the pre-calculated byte offset
        offset = self.line_offsets[idx]
        self._file.seek(offset)

        # Read the line and parse the JSON
        line = self._file.readline()
        data = json.loads(line)

        # Apply transformation if it exists
        if self.transform:
            data = self.transform(data)

        return data

    def __getitem__(
            self,
            idx: int | slice
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Supports both integer indexing and slicing.
        - If idx is an integer, returns a single data item.
        - If idx is a slice, returns a list of data items.
        """
        if isinstance(idx, int):
            return self._get_single_item(idx)
        elif isinstance(idx, slice):
            # Resolve slice indices
            start, stop, step = idx.indices(len(self))
            return [self._get_single_item(i) for i in range(start, stop, step)]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}. Must be int or slice.")

    def __del__(self) -> None:
        """Ensures the file handle is closed when the dataset object is garbage collected."""
        if self._file:
            self.close()

    def close(self) -> None:
        """Closes the file handle."""
        if self._file:
            self._file.close()
            self._file = None


def get_image_UIDs(
        path: Path,
        split: Literal[
            'trainval',
            'train'
            'val'
            ] = "trainval",
        shuffle: bool = True
) -> list[int]:
    """
    Lists the UIDs of the images stored into a certain path and by split.

    Args:
        path: root directory of the images.
        split: split type ('train' or 'class-splitted').
    
    Returns:
        List of image UIDs.

    Returns a list of image UIDs read in the "splits.txt" file for a specified split.
    """
    image_UIDs = []
    with open(path / f"{split}.txt", "r") as f:
        for line in f:
            image_id = line.strip()  # Remove any leading/trailing whitespace
            image_UIDs.append(image_id)
    if shuffle:
        match split:
            case "trainval":
                to_shuffle = image_UIDs[23:]
                random.shuffle(to_shuffle)
                image_UIDs[23:] = to_shuffle
            case _:
                random.shuffle(image_UIDs)
    return image_UIDs

image_UIDs = np.array(get_image_UIDs(SPLITS_PATH, split="trainval"))
image_train_UIDs = np.array(get_image_UIDs(SPLITS_PATH, split="train"))
image_val_UIDs = np.array(get_image_UIDs(SPLITS_PATH, split="val"))

def get_image(
        path: str,
) -> torch.Tensor:
    """
    Reads a single image from disk and encodes it in a tensor.

    Args:
        path: Path to the image file.

    Returns:
        Image as a torch.Tensor on the global device.
    """
    img = decode_image(path)
    img = img.to(CONFIG["device"])
    return img

def image_to_base64(
        img: Image.Image
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
    one_hot_masks = masks == torch.arange(NUM_CLASSES).to(CONFIG["device"])[:, None, None, None]
    one_hot_masks = one_hot_masks.swapaxes(0, 1)
    return one_hot_masks

def get_sc(
        path: Path,
        resize_size: None | int | tuple[int, int] = None,
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
    if resize_size is not None:
        sc = TF.resize(sc, resize_size, TF.InterpolationMode.BILINEAR)
    if center_crop:
        sc = TF.center_crop(sc, output_size=min(sc.shape[1:]))
    return sc

def apply_classmap(
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
    # mask_ = mask.cpu()
    # mask_.apply_(lambda x: class_map[x]) # class mapping # TODO this is too slow
    mask = map_tensor(mask, class_map)
    return mask

def get_mask(
        path: str,
        class_map: dict,
        resize_size: int | tuple[int, int] | None,
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
    mask = decode_image(path).to(CONFIG["device"])
    mask = mask[:1, :, :]
    if class_map:
        mask = apply_classmap(mask, class_map)
    if resize_size:
        mask = TF.resize(mask, resize_size, T.InterpolationMode.NEAREST)
    if center_crop:
        mask = TF.center_crop(mask, output_size=min(mask.shape[1:]))
    return mask

def get_gt(
        path: Path,
        class_map: dict,
        resize_size: None | tuple[int, int] | int = None,
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
    return get_mask(path, class_map, resize_size, center_crop)

def get_pr(
        path: str,
        class_map: dict,
        resize_size: int | tuple[int, int] | None = None,
        center_crop: bool = True
) -> torch.Tensor:
    """
    Loads and processes a predicted mask.

    Args:
        path: Path to the mask file.
        class_map: Dictionary mapping class indices.
        resize_size: Target size as int, tuple, or None.
        resize_mode: Interpolation mode.
        center_crop: Whether to center crop the mask.

    Returns:
        Processed predicted mask tensor.
    """
    return get_mask(path, class_map, resize_size, center_crop)

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
    mask = get_mask(path, class_map, image_size, center_crop=True)
    significant_classes = mask.unique().tolist() # classes that actually appear in 'gt'
    significant_classes.remove(0)
    return significant_classes

def read_json(json_path: Path) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

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
        return_state: bool,
        format_to_dict: bool = False
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
    if format_to_dict:
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
        img_idx,
        return_state: bool = False,
        format_to_dict: bool = False
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
    answer_gt = get_one_item(get_answer_gts_path(by_model), img_idx, return_state, format_to_dict)
    return answer_gt

def get_one_sup_set_answer_gt(
        by_model,
        img_idx,
        format_to_dict: bool,
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
    answer_gt = get_one_item(get_sup_set_answer_gts_path(by_model), img_idx, return_state, format_to_dict=format_to_dict)
    return answer_gt

def get_one_answer_pr(
        by_model,
        split_by,
        relative_path,
        img_idx,
        return_state: bool = False,
        format_to_dict: bool = False
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
    answer_pr = get_one_item(get_answer_prs_path(by_model, split_by, relative_path), img_idx, return_state, format_to_dict)
    return answer_pr

def get_many_answer_gt(
        by_model,
        return_state: bool = False,
        format_to_dict: bool = False
) -> dict:
    """
    Gets ground truth answers for multiple images by model.

    Args:
        by_model: Model identifier.
        return_state: Whether to include the state in the result.

    Returns:
        Dictionary of ground truth answers.
    """
    answer_gts = get_many_item(get_answer_gts_path(by_model), return_state, format_to_dict)
    return answer_gts

def get_many_answer_pr(
        path: Path,
        return_state: bool = False,
        format_to_dict: bool = False
) -> dict:
    """
    Gets predicted answers for multiple images from a JSONL file.

    Args:
        path: Path to the JSONL file.
        return_state: Whether to include the state in the result.

    Returns:
        Dictionary of predicted answers.
    """
    answer_prs = get_many_item(path, return_state, format_to_dict)
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

def distinct_k(
        texts: list[str],
        k_max: int = 4, 
) -> float:
    """ Calculates corpus-level ngram diversity based on unique ngrams 
       (e.g., https://arxiv.org/pdf/2202.00666.pdf).

    Args:
        data (List[str]): List of documents. 
        num_n (int): Max ngrams to test up to. Defaults to 5. 

    Returns:
        float: ngram diveristy score.
    """
    score = 0 
    texts = ' '.join(texts).split(' ') # format to list of words

    for k in range(1, k_max + 1): 
        ngrams = list(nltk.ngrams(texts, k))
        # num unique ngrams / all ngrams for each size n 
        score += len(set(ngrams)) / len(ngrams)

    return round(score, 3)

def ent_k(
        texts: list[str],
        k: int = 4
) -> float:
    def generate_ngrams(
            words: list[str],
            k: int
    ):
        return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
    
    ngrams = []
    for text in texts:
        words = text.split()
        ngrams.extend(generate_ngrams(words, k))

    ngram_counts = Counter(ngrams)
    total_ngrams = sum(ngram_counts.values())

    ngram_frequencies = [count / total_ngrams for _, count in ngram_counts.items()]

    entropy = -sum(freq * math.log(freq, 2) for freq in ngram_frequencies)
        
    return entropy

def embd_diversity(
        embds: list[Tensor],
) -> float:
    """
    Computes the average cosine similarity between the tensors in a fast way.

    Args:
        embds: A list of PyTorch tensors (embeddings).

    Returns:
        The average cosine similarity between all unique pairs of tensors.
    """
    if len(embds) < 2:
        raise AttributeError("Embedding diversity is not defined for just 1 vector.")

    stacked_embds = torch.stack(embds) # stack the tensors into a single 2D tensor for efficient computation

    normalized_embds = torch.nn.functional.normalize(stacked_embds, p=2, dim=1) # normalize the embeddings to have unit L2 norm

    similarity_matrix = torch.matmul(normalized_embds, normalized_embds.T) # compute the pairwise dot products (which are now cosine similarities)
    
    total_sum = torch.sum(similarity_matrix) # get the sum of all elements in the matrix    
    diagonal_sum = torch.trace(similarity_matrix) # get the sum of diagonal elements (which are all 1s)
    sum_of_unique_pairwise_similarities = (total_sum - diagonal_sum) / 2 # sum of unique pairwise similarities

    # Calculate the number of unique pairs
    num_embeddings = len(embds)
    num_unique_pairs = num_embeddings * (num_embeddings - 1) / 2

    return sum_of_unique_pairwise_similarities / num_unique_pairs

def vendi_score(
    embds: list[Tensor],
) -> float:
    """
    Computes the VendiScore for a list of PyTorch tensors (embeddings).

    Args:
        embds: A list of PyTorch tensors (embeddings).

    Returns:
        The VendiScore.
    """
    if len(embds) < 1:
        raise AttributeError("VendiScore is not defined for an empty list of vectors.")

    stacked_embds = torch.stack(embds) # stack the tensors into a single 2D tensor

    # Normalize the embeddings to have unit L2 norm
    # This assumes a cosine similarity kernel where K(x,x) = 1
    normalized_embds = torch.nn.functional.normalize(stacked_embds, p=2, dim=1)

    # Compute the pairwise dot products (which are now cosine similarities), forming the kernel matrix K
    similarity_matrix = torch.matmul(normalized_embds, normalized_embds.T)

    # The VendiScore is typically defined with a normalized kernel matrix K/n,
    # where n is the number of embeddings.
    # However, the eigenvalues are often normalized directly after computing from K.
    # Let's follow the definition where lambda_i are the eigenvalues of K/n.
    num_embeddings = len(embds)
    normalized_kernel_matrix = similarity_matrix / num_embeddings

    # Compute the eigenvalues
    # torch.linalg.eigvalsh is for symmetric matrices, which our similarity_matrix is.
    eigenvalues = torch.linalg.eigvalsh(normalized_kernel_matrix)

    # Filter out any tiny negative eigenvalues that might arise from numerical instability
    eigenvalues = torch.relu(eigenvalues)

    # Normalize eigenvalues to sum to 1, if they don't already (they should for K/n)
    sum_eigenvalues = torch.sum(eigenvalues)
    if sum_eigenvalues == 0:
        return 0.0 # Handle case where all embeddings are identical (or zero vectors)
    normalized_eigenvalues = eigenvalues / sum_eigenvalues

    # Compute Shannon entropy of the normalized eigenvalues
    # Handle log(0) case: 0 * log(0) is defined as 0
    entropy_terms = normalized_eigenvalues * torch.log(normalized_eigenvalues + 1e-12) # Add small epsilon for log(0) stability
    shannon_entropy = -torch.sum(entropy_terms)

    # Compute VendiScore
    vendi_score_value = torch.exp(shannon_entropy)

    return vendi_score_value.item()

def flatten_class_splitted_answers(
        class_splitted_answers: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    flat_answers = []
    b = 0
    for csa in class_splitted_answers:
        new_answers = list(csa["content"].values())
        new_answers = [{"img_idx": b+i, "content": a} for i, a in enumerate(new_answers)]
        flat_answers.extend(new_answers)
        b += len(new_answers)
    return flat_answers

        
def class_pixel_distribution(
        dl: DataLoader,
        num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the pixel-wise class distribution for a segmentation dataset.

    Args:
        dl: PyTorch DataLoader providing (image, label) batches.
                                 Labels are expected to be 2D tensors (HxW) with integer class IDs.
        num_classes: The total number of classes in the dataset.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - class_pixel_counts (torch.Tensor): A 1D tensor where each element
                                                 is the total count of pixels for that class.
            - class_pixel_distribution_percentage (torch.Tensor): A 1D tensor
                                                                   with the percentage of pixels
                                                                   for each class.
    """
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("'num_classes' must be a positive integer.")

    # Initialize a tensor to store pixel counts for each class
    class_pixel_counts = torch.zeros(num_classes, dtype=torch.long, device=CONFIG["device"])

    # Iterate through the DataLoader
    for i, (_, labels) in enumerate(dl):
        # labels will be of shape [batch_size, H, W]
        # For each label in the batch:
        for label_map in labels:
            # label_map is [H, W]
            # Flatten the label map to easily count pixel values
            flattened_label = label_map.view(-1)

            # Use torch.bincount to count occurrences of each class ID
            # minlength ensures the tensor has 'num_classes' elements even if some classes are missing
            counts = torch.bincount(flattened_label, minlength=num_classes)

            # Add these counts to our total
            class_pixel_counts += counts
        
    total_pixels = torch.sum(class_pixel_counts).item()

    if total_pixels == 0:
        print("Warning: No pixels processed. Check your dataset and labels.")
        return class_pixel_counts, torch.zeros(num_classes, dtype=torch.float)
    else:
        # Calculate percentages
        class_pixel_distribution_percentage = (class_pixel_counts.float() / total_pixels) * 100
        return class_pixel_counts, class_pixel_distribution_percentage
    
def crop_augment_preprocess_batch(
        batch: list,
        crop_fn: Callable,
        augment_fn: Callable,
        preprocess_fn: Callable
) -> tuple[Tensor, Tensor]:
    # Has to be made into a 'collate_fn' by fixing the parameters other than 'batch'!
    x, y = zip(*batch)

    # when the images are sampled in the batch, they are:
    #   1. in Float32 in the range [0, 1],
    #   2. in shape [B, C, H, W],
    
    x, y = zip(*[crop_fn(x_, tv_tensors.Mask(y_)) for x_, y_ in zip(x, y)])

    x = (torch.stack(x)/255.).float()
    y = torch.stack(y).long().squeeze(1)

    if augment_fn:
        x, y = augment_fn(x, tv_tensors.Mask(y))

    if preprocess_fn:
        x = preprocess_fn(x)

    return x, y

def main() -> None:
    jsonl_ds = JSONLDataset(Path("/home/olivieri/exp/data/data_gen/VOC2012/flat/train_no_aug_flat.jsonl"))
    print(jsonl_ds)
    print(len(jsonl_ds))

if __name__ == "__main__":
    main()
