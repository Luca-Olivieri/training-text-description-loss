import json
from typing import Tuple

from sympy import flatten

from path import *
from utils import *

from torchvision.io import decode_image
import torch.nn.functional as F
import torchvision.transforms.functional as TF

CLASSES = ["BACKGROUND", "AEROPLANE", "BYCICLE", "BIRD", "BOAT", "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE", "DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP", "SOFA", "TRAIN", "TVMONITOR"]

# define the class mappings
CLASS_MAP = {i: i for i in range(len(CLASSES))} # default mapping

NUM_CLASSES = len(list(CLASS_MAP.values())) # actual number of classes

def get_image_UIDs(path, split="trainval"):
    """
    Returns a list of image UIDs read in the "splits.txt" file for a specified split.
    """
    image_UIDs = []
    with open(path / f"{split}.txt", "r") as f:
        for line in f:
            image_id = line.strip()  # Remove any leading/trailing whitespace
            image_UIDs.append(image_id)
    return image_UIDs

image_UIDs = get_image_UIDs(SPLITS_PATH, split="trainval")

def get_image(path):
    """
    Reads a single image from disk and encodes it in a tensor of 
    - shape (N, H, W). N could be 3 with RGB or 1 with grayscale images.
    - dtype 'uint8'.
    Then moves the tensor on the device globally used.
    """
    img = decode_image(path)
    img = img.to(DEVICE)
    return img

def one_hot_encode_masks(masks: torch.Tensor) -> torch.Tensor:
    one_hot_masks = masks == torch.arange(NUM_CLASSES).to(DEVICE)[:, None, None, None]
    one_hot_masks = one_hot_masks.swapaxes(0, 1)
    return one_hot_masks

def resize_image_(img: torch.Tensor, image_size: int | Tuple[int, int], mode: str) -> torch.Tensor:
    img = (img/255.).unsqueeze(0)
    img = F.interpolate(img, size=image_size, mode=mode)
    img = (img.squeeze(0)*255).clamp(0, 255).byte()
    return img

def resize_image(img: torch.Tensor, image_size: int | Tuple[int, int], mode: str) -> torch.Tensor:
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
        size = image_size  # 'image_size' is of type Tuple[int, int]
        
    img = F.interpolate(img, size=size, mode=mode, align_corners=False if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None)
    img = (img.squeeze(0) * 255).clamp(0, 255).byte()
    return img

def resize_mask(mask: torch.Tensor, image_size: None | Tuple[int, int] | int, mode: str, center_crop: bool) -> torch.Tensor:
    """
    Resized mask in two ways, as defined by 'mode' parameter:
    - "bilinear": one-hots the mask image and performs bilinear interpolation of the probabilities to provide a smoother resizing.
    - "nearest": applies nearest interpolation to the original mask.
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

def get_sc(path: Path, image_size: None | int | Tuple[int, int] = None, center_crop: bool = True):
    sc = get_image(path)
    if image_size is not None:
        sc = resize_image(sc, image_size, mode="bilinear")
    if center_crop:
        sc = TF.center_crop(sc, output_size=min(sc.shape[1:]))
    return sc

def apply_class_map(mask: torch.Tensor, class_map: dict) -> torch.Tensor:
    mask_ = mask.cpu()
    mask_.apply_(lambda x: class_map.get(x, 0)) # class mapping
    mask = mask_.to(DEVICE)
    return mask

def _get_mask(path, class_map, image_size, resize_mode, center_crop: bool):
    mask = get_image(path)
    class_map_ = class_map.copy()
    class_map_[255] = 0 # additionally, map UNLABELLED to BACKGROUND
    mask = apply_class_map(mask, class_map_)
    if image_size is not None:
        mask = resize_mask(mask, image_size, resize_mode, center_crop=True)
    if center_crop:
        mask = TF.center_crop(mask, output_size=min(mask.shape[1:]))
    return mask

def get_gt(path: Path, class_map: dict, image_size: None | Tuple[int, int] | int = None, resize_mode: str = "nearest", center_crop: bool = True):
    return _get_mask(path, class_map, image_size, resize_mode, center_crop)

def get_pr(path, class_map, image_size=None, resize_mode="nearest", center_crop: bool = True):
    return _get_mask(path, class_map, image_size, resize_mode, center_crop)

def get_significant_classes(path, image_size, class_map):
    mask = _get_mask(path, image_size, class_map)
    significant_classes = mask.unique().tolist() # classes that actually appear in 'gt'
    significant_classes.remove(0) # TODO: should I remove the background class?
    return significant_classes

def read_txt(txt_path):
    with open(txt_path, "r") as f:
        content = f.read()
        return content
    raise FileNotFoundError(f"Error when attempting to read '{txt_path}'")

def _read_one_jsonl_line(line):
    obj = json.loads(line)
    return obj

def is_state(obj):
    return "state" in obj.keys()

def read_state(path):
    with open(path, 'r') as file:
        first_line = file.readline()
        state_obj = _read_one_jsonl_line(first_line)    
    return state_obj if is_state(state_obj) else None

def read_many_from_jsonl(path):
    with open(path, 'r') as file:
        data = list(map(lambda l: _read_one_jsonl_line(l), file))
    return data[1:] if is_state(data[0]) else data

def _append_one_to_jsonl(object_to_append, file):
    json.dump(object_to_append, file)
    file.write('\n') # ensure to write a newline after each JSONL object

def append_many_to_jsonl(path, objects_to_append):
    with open(path, 'a+') as file:
        list(map(lambda obj: _append_one_to_jsonl(obj, file), objects_to_append))

def read_one_from_jsonl_by(path, key, value):
    # TODO maybe this method can be made faster
    """
    'key' must be at the top level of the JSON object (no nested keys).
    """
    with open(path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            if obj.get(key, None) == value:
                return obj

def _format_one_from_jsonl(obj):
    if is_state(obj):
        return obj
    else:
        return {obj["img_idx"]: obj["content"]}

def _format_many_from_jsonl(obj_list):
    return {line["img_idx"]: line["content"] for line in obj_list}

def get_one_item(path, idx, return_state):
    state = read_state(path)
    item = read_one_from_jsonl_by(path, "img_idx", idx)
    item = _format_one_from_jsonl(item)
    return state | item if return_state else item

def get_many_item(path, return_state):
    state = read_state(path)
    items = read_many_from_jsonl(path)
    items = _format_many_from_jsonl(items)
    return state | items if return_state else items

def get_one_answer_gt(by_model, split_by, idx, return_state=False):
    answer_gt = get_one_item(get_answer_gts_path(by_model, split_by), idx, return_state)
    return answer_gt

def get_one_sup_set_answer_gt(by_model, split_by, idx, return_state=False):
    answer_gt = get_one_item(get_sup_set_answer_gts_path(by_model, split_by), idx, return_state)
    return answer_gt

def get_one_answer_pr(by_model, split_by, exp, variation, idx, return_state=False):
    answer_pr = get_one_item(get_answer_prs_path(by_model, split_by, f"{exp}/{variation}"), idx, return_state)
    return answer_pr

def get_many_answer_gt(by_model, split_by, return_state=False):
    answer_gts = get_many_item(get_answer_gts_path(by_model, split_by), return_state)
    return answer_gts

def get_many_answer_pr(by_model, split_by, exp, variation, return_state=False):
    answer_prs = get_many_item(get_answer_prs_path(by_model, split_by, f"{exp}/{variation}"), return_state)
    return answer_prs

def get_one_eval_gt(by_model, split_by, idx, return_state=False):
    eval_gt = get_one_item(get_eval_gts_path(by_model, split_by), idx, return_state)
    return eval_gt

def get_one_eval_pr(by_model, split_by, exp, variation, idx, return_state=False):
    eval_pr = get_one_item(get_eval_prs_path(by_model, split_by, f"{exp}/{variation}"), idx, return_state)
    return eval_pr

def get_many_eval_gt(by_model, split_by, return_state=False):
    eval_gts = get_many_item(get_eval_gts_path(by_model, split_by), return_state)
    return eval_gts

def get_many_eval_pr(by_model, split_by, exp, variation, return_state=False):
    eval_prs = get_many_item(get_eval_prs_path(by_model, split_by, f"{exp}/{variation}"), return_state)
    return eval_prs
    
def format_many_to_jsonl(objs):
    objs_list = [{"state": objs["state"]}]
    objs_list.extend([{"img_idx": img_idx, "content": content} for img_idx, content in list(objs.items())[1:]])
    return objs_list

def expand_words_to_variants(word):
    """
    Expands a word in a list of similar word to involve for the pertinence check
    """
    return [word, word.lower()]

def validate_pertinence(sentences, significant_classes):
    """
    Asserts if predicted answers only contain upper case words (as all and only class names should be) related to their positive class.
    The positive class name needs to be there and all other cannot.
    """
    for s, pos_class in zip(sentences, significant_classes):
        reason_upper_words = extract_uppercase_words(s)
        pos_class_name = CLASSES[pos_class]
        forbidden_class_names = flatten_list([expand_words_to_variants(cn) for cn in CLASSES if cn != pos_class_name])
        assert all(word != fw for word in reason_upper_words for fw in forbidden_class_names), f"Forbidden words found in answer of pos. class '{pos_class_name}'"
        assert pos_class_name in reason_upper_words, f"Allowed word '{pos_class_name}' not found in answer '{pos_class_name}'"

def main() -> None:
    pass

if __name__ == "__main__":
    main()
