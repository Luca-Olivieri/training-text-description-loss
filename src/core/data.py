from core.config import *
from core.utils import extract_uppercase_words
from core.data_utils import flatten_list, is_state
from core.torch_utils import map_tensors

import json
import io

import torch
from torchvision.io import decode_image
import torchvision.transforms.functional as TF
from torchvision import tv_tensors
from PIL import Image
import base64
from pathlib import Path

from core._types import Optional, Callable

def get_image(
        path: Path,
        device: torch.device,
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
    img = decode_image(path, mode=mode).to(device)
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

def get_mask(
        path: Path,
        device: torch.device,
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
    mask = decode_image(path).to(device)
    mask = mask[:1, :, :]
    if class_map:
        mask = map_tensors(mask, class_map)
    if resize_size:
        mask = TF.resize(mask, resize_size, interpolation=resize_mode)
    if center_crop:
        mask = TF.center_crop(mask, output_size=min(mask.shape[-2:]))
    return mask

def read_json(json_path: Path) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# TODO turn prompts to .txt to .md
def read_txt(
        txt_path: Path
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
    @staticmethod
    def read_one_jsonl_line(
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
            path: Path
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
            path: Path
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

    @staticmethod
    def _append_one_to_jsonl(
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
            path: Path,
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

    @staticmethod
    def read_one_from_jsonl_by(
            path: Path,
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

    @staticmethod
    def format_one_from_jsonl(
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

    @staticmethod
    def format_many_from_jsonl(
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
            path: Path,
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
        
    @staticmethod
    def format_many_to_jsonl(
        objs: dict,
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
    # jsonl_ds = JSONLDataset(Path("/home/olivieri/exp/data/private/data_gen/VOC2012/flat/train_no_aug_flat.jsonl"))
    # print(jsonl_ds)
    # print(len(jsonl_ds))
    ...

if __name__ == "__main__":
    main()
