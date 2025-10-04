from core.config import *
from core.path import get_answer_gts_path, get_sup_set_answer_gts_path, get_answer_prs_path, get_eval_gts_path, get_eval_prs_path
from core.utils import extract_uppercase_words
from core.data_utils import flatten_list, is_state
from core.torch_utils import map_tensors

import json
from glob import glob
import io

import numpy as np
import xarray as xr
import pandas as pd

import torch
from torchvision.io import decode_image
import torchvision.transforms.functional as TF
from torchvision import tv_tensors
from torch.utils.data import DataLoader
from PIL import Image
import base64
from pathlib import Path

from core._types import Optional, Callable, deprecated

# TODO move from default uint8 [0, 255] to float32 for images

def read_image(
        path: Path,
        device: torch.device,
        mode: str = 'RGB',
) -> torch.Tensor:
    return decode_image(path, mode=mode).to(device)

def read_mask(
        path: Path,
        device: torch.device,
) -> torch.Tensor:
    return read_image(path, mode='GRAY', device=device)

@deprecated("Substitued with read_image")
def get_image(
        path: Path,
        resize_size: None | int | tuple[int, int] = None,
        resize_mode: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
        center_crop: bool = True,
        mode: str = "RGB"
) -> torch.Tensor:
    """
    Reads a single image from disk and encodes it in a tensor.
    Optionally resizes and center-crops an image

    Args:
        path: Path to the image file.
        image_size: Target size as int, tuple, or None.
        center_crop: Whether to center crop the image.

    Returns:
        Image as a torch.Tensor on the global device.
    """
    img = decode_image(path, mode=mode).to(CONFIG["device"])
    if img.shape[0] == 1:
        img = img.expand(3, -1, -1) # shapes [1, H, W] are expanded to [3, H, W]
    if resize_size is not None:
        img = TF.resize(img, resize_size, resize_mode)
    if center_crop:
        img = TF.center_crop(img, output_size=min(img.shape[-2:]))
    return img

# TODO there are two implementation, this and the one in 'viz.py', select one and use it everywhere.
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

@deprecated("Substitued with read_mask")
def get_mask(
        path: str,
        class_map: Optional[dict] = None,
        resize_size: Optional[int | tuple[int, int]] = None,
        resize_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
        center_crop: Optional[bool] = None
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
        mask = map_tensors(mask, class_map)
    if resize_size:
        mask = TF.resize(mask, resize_size, interpolation=resize_mode)
    if center_crop:
        mask = TF.center_crop(mask, output_size=min(mask.shape[-2:]))
    return mask

@deprecated("The best one is in 'Prompter.py'")
def get_significant_classes_(
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

# TODO turn prompts to .txt to .md
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

class JsonlIO:
    """
    TODO
    """
    def read_one_jsonl_line(
            self,
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



    def read_state(
            self,
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
            state_obj = self.read_one_jsonl_line(first_line)    
        return state_obj if is_state(state_obj) else None

    def read_many_from_jsonl(
            self,
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
            data = list(map(lambda l: self.read_one_jsonl_line(l), file))
        return data[1:] if is_state(data[0]) else data

    def _append_one_to_jsonl(
            self,
            object_to_append: dict,
            file,
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
            self,
            path: str,
            objects_to_append: list[dict],
    ) -> None:
        """
        Appends multiple objects to a JSONL file.

        Args:
            path: Path to the JSONL file.
            objects_to_append: List of dictionaries to append.
        """
        with open(path, 'a+') as file:
            list(map(lambda obj: self._append_one_to_jsonl(obj, file), objects_to_append))

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
        # NOTE 'key' must be at the top level of the JSON object (no nested keys).
        with open(path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                if obj.get(key, None) == value:
                    return obj

    def format_one_from_jsonl(
            self,
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

    def format_many_from_jsonl(
            self,
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
            self,
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
        state = self.read_state(path)
        item = self.read_one_from_jsonl_by(path, "img_idx", idx)
        if format_to_dict:
            item = self.format_one_from_jsonl(item)
        return state | item if return_state else item

    def get_many_item(
            self,
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
        state = self.read_state(path)
        items = self.read_many_from_jsonl(path)
        items = self.format_many_from_jsonl(items) if format_to_dict else items
        if return_state:
            return items, state
        else:
            return items

# TODO answers and ground truth retrieval logic should place in a dataset

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
        classes: list[str],
        significant_classes: list[int]
) -> None:
    """
    Asserts if predicted answers only contain upper case words (as all and only class names should be) related to their positive class.
    The positive class name needs to be there and all other cannot.
    """
    for s, pos_class in zip(sentences, significant_classes):
        reason_upper_words = extract_uppercase_words(s)
        pos_class_name = classes[pos_class]
        allowed_class_names = ["BACKGROUND", pos_class_name]
        forbidden_class_names = flatten_list([expand_words_to_variants(cn) for cn in classes if cn not in allowed_class_names])
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
        exp_path: Path,
        num_classes: int
) -> xr.DataArray:
    data_da = None

    var_paths = glob(f"{exp_path}/*.jsonl")
    var_names = [os.path.splitext(os.path.basename(path))[0] for path in var_paths]

    for var_n, var_p in zip(var_names, var_paths):
        
        eval_prs = get_many_eval_pr(var_p, return_state=False, format_to_dict=True)
        prs_per_img_idx_df = pd.DataFrame.from_dict(eval_prs, orient='index')
        
        for column in [str(n) for n in range(0, num_classes)]:
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
        preprocess_fn: Callable,
        output_uids: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    # Has to be made into a 'collate_fn' by fixing the parameters other than 'batch'!
    if output_uids:
        uids, x, y = zip(*batch)
    else:
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

    if output_uids:
        return uids, x, y
    else:
        return x, y

def crop_image_preprocess_image_text_batch(
        batch: list,
        crop_fn: Callable,
        preprocess_images_fn: Optional[Callable],
        preprocess_texts_fn: Optional[Callable],
        output_text_metadata: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    imgs, texts = zip(*batch)

    # when the images are sampled in the batch, they are:
    #   1. in Float32 in the range [0, 1],
    #   2. in shape [B, C, H, W],
    
    imgs = [crop_fn(x_) for x_ in imgs]
    imgs = (torch.stack(imgs)/255.).float()

    if preprocess_images_fn:
        imgs = preprocess_images_fn(imgs)

    texts_metadata = [d for d in texts]
    texts = [d['content'] for d in texts]

    if preprocess_texts_fn:
        texts = preprocess_texts_fn(texts, imgs.device)

    if output_text_metadata:
        return imgs, texts, texts_metadata
    else:
        return imgs, texts

def main() -> None:
    # jsonl_ds = JSONLDataset(Path("/home/olivieri/exp/data/data_gen/VOC2012/flat/train_no_aug_flat.jsonl"))
    # print(jsonl_ds)
    # print(len(jsonl_ds))
    ...

if __name__ == "__main__":
    main()
