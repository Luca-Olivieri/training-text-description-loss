import json

from sympy import flatten

from path import *
from utils import *

import torchvision.transforms.functional as F
from torchvision.io import decode_image
from torch.nn.functional import interpolate

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

def letterbox_tensor(x):
    """
    Letterboxes a tensor 'x' of shape (C, H, W) encoding an image.
    """
    _, h, w = x.shape  # shape (C, H, W)
    max_wh = max(h, w)
    hp = (max_wh - w) // 2
    vp = (max_wh - h) // 2
    padding = (hp, vp, hp, vp)  # (left, right, top, bottom)
    return F.pad(x, padding, 0)

def _get_image(path, image_resizing_mode):
    img = decode_image(path)
    if image_resizing_mode == "letterboxed":
        img = letterbox_tensor(img)
    img = img.to(DEVICE)
    return img

def get_sc(path, image_size, image_resizing_mode):
    sc = _get_image(path, image_resizing_mode)
    sc = interpolate(sc.unsqueeze(0)/255., size=image_size, mode="bilinear").squeeze(0)
    return sc

def _get_mask(path, image_size, class_map, image_resizing_mode):
    mask = _get_image(path, image_resizing_mode)
    class_map_ = class_map.copy()
    class_map_[255] = 0 # additionally, map UNLABELLED to BACKGROUND
    mask_ = mask.cpu()
    mask_.apply_(lambda x: class_map_.get(x, 0)) # class mapping
    mask = interpolate(mask_.to(DEVICE).unsqueeze(0), size=image_size, mode="nearest").squeeze(0)
    return mask

def get_gt(path, image_size, class_map, image_resizing_mode):
    return _get_mask(path, image_size, class_map, image_resizing_mode)

def get_pr(path, image_size, class_map, image_resizing_mode):
    return _get_mask(path, image_size, class_map, image_resizing_mode)

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

def get_one_answer_gt(by_model, image_resizing_mode, output_mode, split_by, idx, return_state=False):
    answer_gt = get_one_item(get_answer_gts_path(by_model, image_resizing_mode, output_mode, split_by), idx, return_state)
    return answer_gt

def get_one_sup_set_answer_gt(by_model, image_resizing_mode, output_mode, split_by, idx, return_state=False):
    answer_gt = get_one_item(get_sup_set_answer_gts_path(by_model, image_resizing_mode, output_mode, split_by), idx, return_state)
    return answer_gt

def get_one_answer_pr(by_model, image_resizing_mode, output_mode, split_by, exp, variation, idx, return_state=False):
    answer_pr = get_one_item(get_answer_prs_path(by_model, image_resizing_mode, output_mode, split_by, f"{exp}/{variation}"), idx, return_state)
    return answer_pr

def get_many_answer_gt(by_model, image_resizing_mode, output_mode, split_by, return_state=False):
    answer_gts = get_many_item(get_answer_gts_path(by_model, image_resizing_mode, output_mode, split_by), return_state)
    return answer_gts

def get_many_answer_pr(by_model, image_resizing_mode, output_mode, split_by, exp, variation, return_state=False):
    answer_prs = get_many_item(get_answer_prs_path(by_model, image_resizing_mode, output_mode, split_by, f"{exp}/{variation}"), return_state)
    return answer_prs

def get_one_eval_gt(by_model, image_resizing_mode, output_mode, split_by, idx, return_state=False):
    eval_gt = get_one_item(get_eval_gts_path(by_model, image_resizing_mode, output_mode, split_by), idx, return_state)
    return eval_gt

def get_one_eval_pr(by_model, image_resizing_mode, output_mode, split_by, exp, variation, idx, return_state=False):
    eval_pr = get_one_item(get_eval_prs_path(by_model, image_resizing_mode, output_mode, split_by, f"{exp}/{variation}"), idx, return_state)
    return eval_pr

def get_many_eval_gt(by_model, image_resizing_mode, output_mode, split_by, return_state=False):
    eval_gts = get_many_item(get_eval_gts_path(by_model, image_resizing_mode, output_mode, split_by), return_state)
    return eval_gts

def get_many_eval_pr(by_model, image_resizing_mode, output_mode, split_by, exp, variation, return_state=False):
    eval_prs = get_many_item(get_eval_prs_path(by_model, image_resizing_mode, output_mode, split_by, f"{exp}/{variation}"), return_state)
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
