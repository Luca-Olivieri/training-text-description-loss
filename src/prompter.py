from config import *
from utils import blend_tensors, flatten_list
from data import VOC2012SegDataset, read_txt, get_one_answer_gt, get_one_sup_set_answer_gt, get_significant_classes_, read_json
from color_map import pil_to_class_array
# from data import VOC2012SegDataset, read_txt, get_image_UIDs, get_one_answer_gt, get_one_sup_set_answer_gt, get_significant_classes_, read_json
from path import get_prompts_path, get_data_gen_prompts_path, LOCAL_ANNOT_IMGS_PATH, MISC_PATH, SPLITS_PATH
from color_map import get_color_map_as, apply_colormap

from PIL import Image, ImageFont, ImageDraw, ImageOps
from pathlib import Path
import json
import pformat as pf
from copy import deepcopy
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as TF
from collections import OrderedDict
import re

from typing import Self, Any, List, Optional

# TODO check if it's better to use 'transforms.v2' instead of plain 'transforms'

Prompt = list[str | Image.Image]

def _concat_images_fn(
        images: list[Image.Image],
        titles: list[str],
        scene_mode: str, 
        align: str
) -> Image.Image:
    """
    Concatenates images according to the values of the parameters 'scene_mode' and 'align' in various combinations.
    Some of them mark the images with titles given as parameters.

    Args:
        images: List of PIL Image objects to concatenate.
        titles: List of titles for each image.
        scene_mode: Scene mode string, affects which images are included.
        align: Alignment direction, either 'horizontal' or 'vertical'.

    Returns:
        Concatenated PIL Image object.
    """
    # Constants
    border_size = 10  # Space between images
    padding_size = 20  # Padding around each image
    border_color = "white"  # Background color
    padding_color = "white"  # Padding color
    title_height = 40  # Height allocated for titles

    assert len(images) == 3
    assert len(titles) == 3

    # Handle scene options
    if scene_mode in ["no", "overlay"]:
        _ = images.pop(0)
        titles.pop(0)

    # Load a specific font
    font = ImageFont.truetype(f"{MISC_PATH}/Arial.ttf", size=32)

    # Ensure images have the same dimension in the correct direction
    if align == "horizontal":
        min_height = min(img.height for img in images)
        resized_images = [img.resize((int(img.width * min_height / img.height), min_height)) for img in images]
    elif align == "vertical":
        min_width = min(img.width for img in images)
        resized_images = [img.resize((min_width, int(img.height * min_width / img.width))) for img in images]
    else:
        raise ValueError("Invalid layout type. Choose 'horizontal' or 'vertical'.")

    # Add padding around each image
    padded_images = [ImageOps.expand(img, border=padding_size, fill=padding_color) for img in resized_images]

    # Create title images
    title_images = []
    for title, img in zip(titles, padded_images):
        title_img = Image.new("RGB", (img.width, title_height), padding_color)
        draw = ImageDraw.Draw(title_img)

        # Center the text
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_x = (img.width - text_width) // 2
        text_y = (title_height - text_height) // 2

        draw.text((text_x, text_y), title, fill="black", font=font)
        title_images.append(title_img)

    # Calculate total width and height
    if align == "horizontal":
        total_width = sum(img.width for img in padded_images) + border_size * (len(images) - 1)
        total_height = padded_images[0].height + title_height  # Space for titles
    else:  # Vertical layout
        total_width = max(img.width for img in padded_images)
        total_height = sum(img.height for img in padded_images) + border_size * (len(images) - 1) + len(images) * title_height

    # Create the final concatenated image
    concatenated_image = Image.new("RGB", (total_width, total_height), border_color)

    # Paste images and their respective titles
    x_offset = 0
    y_offset = 0
    for title_img, img in zip(title_images, padded_images):
        if align == "horizontal":
            concatenated_image.paste(title_img, (x_offset, 0))  # Paste title
            concatenated_image.paste(img, (x_offset, title_height))  # Paste image below title
            x_offset += img.width + border_size
        else:  # Vertical layout
            concatenated_image.paste(title_img, (0, y_offset))  # Paste title
            y_offset += title_height
            concatenated_image.paste(img, (0, y_offset))  # Paste image below title
            y_offset += img.height + border_size

    return concatenated_image

def _format_images(
        sc: Image.Image,
        gt: Image.Image,
        pr: Image.Image,
        idx: int,
        layout: str,
        scene_mode: str,
        align: str,
        alpha: str
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """
    Returns formatted images for the prompt.

    Args:
        sc: Scene image (PIL Image).
        gt: Ground truth image (PIL Image).
        pr: Prediction image (PIL Image).
        idx: Index of the image.
        layout: Layout type ('concat', 'separate', 'array').
        scene_mode: Scene mode string.
        align: Alignment direction.
        alpha: Alpha blending value.

    Returns:
        Tuple of formatted images (sc, gt, pr).
    """
    if scene_mode == "overlay":
        gt = Image.blend(sc, gt, alpha)
        pr = Image.blend(sc, pr, alpha)
    if layout == "concat":
        concat_img = _concat_images_fn([sc, gt, pr], [f"Scene {idx}", f"Ground Truth {idx}", f"Prediction {idx}"], scene_mode, align)
        res = [concat_img] * 3
    if layout == "separate":
        if scene_mode in ["no", "overlay"]:
            res = [sc, gt, pr]
        if scene_mode == "yes":
            res = [sc, gt, pr]
    if layout == "array":
        gt_arr = pil_to_class_array(gt)
        pr_arr = pil_to_class_array(pr)
        res = [sc, gt_arr, pr_arr]
    return res

def map_placeholders(
        text: str,
        placeholder: str,
        objects_list: list[Any]
) -> list[str | Any]:
    """
    Substitutes placeholders in a text string with objects from a list.

    The function splits the string into a list of strings and objects,
    interleaving text and the provided objects.

    Args:
        text_string (str): The input text string containing placeholders.
        placeholder_symbol (str): The symbol representing the placeholder (e.g., '[img]').
        objects_list (list): A list of objects to insert into the string.

    Returns:
        A list interleaving text segments and the inserted objects.
        Returns the original text string in a list if no placeholders are found.
        Returns the original text string in a list if the number of placeholders
        does not match the number of objects.
    """
    parts = re.split(f"({re.escape(placeholder)})", text) # split the string by the placeholder symbol
    parts = [part for part in parts if part] # filter out empty strings that might result from splitting

    placeholder_amount = sum(1 for part in parts if part == placeholder) # count actual placeholders found in the split parts

    if not placeholder_amount:
        return text # no placeholders found, return original string
    
    if placeholder_amount != len(objects_list):
        raise AttributeError(f"The string '{text}' contains {placeholder_amount} placeholders, but only {len(objects_list)} objects were given.")

    result = []
    object_index = 0
    for part in parts:
        if part == placeholder:
            if object_index < len(objects_list):
                result.append(objects_list[object_index])
                object_index += 1
        else:
            result.append(part)
    
    return result

# A more specific type hint for the items in the final list
Interleaved = list[str | Any]

def map_list_placeholders(
    texts: List[str],
    placeholder: str,
    objects_list: List[Any]
) -> List[List[Interleaved]]:
    """
    Substitutes placeholders in a list of strings with objects from a list.

    This function iterates through each string, replacing placeholders with objects
    in the order they appear. It first verifies that the total number of placeholders
    across all strings matches the number of objects provided.

    The key difference in this version is its handling of object types:
    - If an object is a string, it is merged directly into the surrounding text.
    - If an object is not a string, it is inserted as a separate element in the list.

    Args:
        texts (List[str]): The input list of text strings containing placeholders.
        placeholder (str): The symbol representing the placeholder (e.g., '[img]').
        objects_list (List[Any]): A list of objects to insert into the strings.

    Returns:
        A list of lists. Each inner list contains the interleaved text segments
        and inserted objects. For example:
        - `map_list_placeholders(["A [P] C"], "[P]", ["B"])` -> `[['A B C']]`
        - `map_list_placeholders(["A [P] C"], "[P]", [ImageObject])` -> `[['A ', ImageObject, ' C']]`

    Raises:
        ValueError: If the total number of placeholders in all strings
                    does not match the number of objects in objects_list.
    """
    # 1. Count total placeholders and validate against the number of objects.
    total_placeholder_count = sum(text.count(placeholder) for text in texts)
    if total_placeholder_count != len(objects_list):
        raise ValueError(
            f"The input list contains {total_placeholder_count} placeholders, "
            f"but {len(objects_list)} objects were provided."
        )

    # If there are no placeholders, return each original string inside its own list.
    if not total_placeholder_count:
        return [[text] for text in texts]

    final_result: List[List[Interleaved]] = []
    object_iterator = iter(objects_list)
    escaped_placeholder = re.escape(placeholder)

    # 2. Iterate through each string to perform the substitution.
    for text in texts:
        # If the placeholder isn't in this specific text, wrap it in a list and continue.
        if placeholder not in text:
            final_result.append([text] if text else [])
            continue

        # Step 1: Split the text by the placeholder, but keep the placeholder as a delimiter.
        # e.g., "A [P] C" -> ['A ', '[P]', ' C']
        # This handles placeholders at the start/end correctly, e.g., "[P]C" -> ['', '[P]', 'C']
        parts = re.split(f"({escaped_placeholder})", text)

        # Step 2: Substitute placeholders with their corresponding objects.
        substituted_parts: List[Interleaved] = []
        for part in parts:
            if part == placeholder:
                # This is a placeholder; get the next object for substitution.
                substituted_parts.append(next(object_iterator))
            elif part: # Only append non-empty text parts.
                substituted_parts.append(part)

        # Step 3: Merge adjacent string parts.
        processed_parts: List[Interleaved] = []
        for item in substituted_parts:
            # If the current item is a string and the last processed part was also a string,
            # merge them together.
            if isinstance(item, str) and processed_parts and isinstance(processed_parts[-1], str):
                processed_parts[-1] += item
            else:
                # Otherwise, append the item (text or object) as a new element.
                processed_parts.append(item)
        
        final_result.append(processed_parts)

    return flatten_list(final_result)

def substitute_list_placeholders(
    texts: List[str],
    substitutions: dict[str, str]
) -> List[str]:
    """
    Substitutes multiple placeholders in a list of strings using a dictionary.

    This function iterates through each string in the input list and replaces
    any placeholders (defined as keys in the substitutions dictionary) with their
    corresponding values. The process is optimized for speed by compiling a
    single regular expression from all placeholder keys.

    Args:
        texts (List[str]): The input list of text strings that may contain
                           placeholders.
        substitutions (Dict[str, str]): A dictionary where keys are the
                                        placeholders to find and values are the
                                        strings to replace them with.

    Returns:
        A new list of strings with all specified placeholders substituted.
        If the substitutions dictionary is empty, it returns a copy of the
        original list.
    """
    if not substitutions:
        return texts.copy()

    # 1. Create a single regular expression from the dictionary keys.
    #    The keys are escaped to ensure they are treated as literal strings.
    #    For example, '[user]' becomes '\\[user\\]'.
    #    This is much faster than iterating and calling str.replace() multiple times.
    placeholder_regex = re.compile(
        "|".join(map(re.escape, substitutions.keys()))
    )

    # 2. Iterate through each text and use the compiled regex to substitute.
    #    A lambda function looks up the matched placeholder in the dictionary
    #    to find its corresponding replacement value.
    return [
        placeholder_regex.sub(lambda match: substitutions[match.group(0)], text)
        for text in texts
    ]

### Prompt Modules ###

class PromptModule:
    """
    Base class for prompt textual modules that form the full prompts.
    Subclasses implement their own logic for prompt construction.

    Attributes:
        prompts_path: Shared path for prompt files.
    """
    prompts_path = None # variable shared among all sub-classes, it has to be initialized externally when building the prompt.
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the prompt module with a specific variation.

        Args:
            variation: Name of the prompt variation.
        """
        self.full_path = Path(self.prompts_path / f"{variation}.txt") # full path complete of the prompt variation.
    
    def import_variation(
            self,
            variation_path: str
    ) -> str:
        """
        Returns the .txt file found at the path 'prompts_path/content_path'.
        Not to be re-implemented by subclasses.

        Args:
            variation_path: Path to the variation file.
        Returns:
            String content of the variation file.
        """
        return read_txt(self.full_path)

    def __call__(self) -> str:
        """
        Returns the textual prompt module complete with items to display.
        Subclasses should re-implement this method as needed.
        """
        return self.import_variation(self.full_path)
    
    def __to_dict__(self) -> dict:
        """
        Returns the module and its attributes as a dictionary.
        Not to be re-implemented by subclasses.

        Returns:
            Dictionary representation of the module.
        """
        return {"class": self.__class__.__name__} | vars(self)
    
### 1. Context ###

class ContextModule(PromptModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the context module with a specific variation.

        Args:
            variation: Name of the context variation.
        """
        super().__init__(f"1_context/{variation}")
    def __call__(self) -> str:
        """
        Returns the context prompt string.
        """
        return super().__call__()

### 2. Color Map ###

class ColorMapModule(PromptModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the color map module with a specific variation.

        Args:
            variation: Name of the color map variation.
        """
        super().__init__(f"2_color_map/{variation}")
    def __call__(
            self,
            color_map_item: Any
    ) -> tuple[str, Any]:
        """
        Returns the color map prompt and the color map item.

        Args:
            color_map_item: Color map item to include in the prompt.
        Returns:
            Tuple of prompt string and color map item.
        """
        return super().__call__(), color_map_item
    
class Image_ColorMapModule(ColorMapModule):
    def __call__(self) -> tuple[str, Image.Image]:
        """
        Returns the color map prompt and image.
        """
        return super().__call__(get_color_map_as("img"))

class RGB_ColorMapModule(ColorMapModule):
    def __call__(self) -> tuple[str, str]:
        """
        Returns the color map prompt and RGB string.
        """
        return super().__call__(get_color_map_as("rgb"))

class Names_ColorMapModule(ColorMapModule):
    def __call__(self) -> tuple[str, str]:
        """
        Returns the color map prompt and names string.
        """
        return super().__call__(get_color_map_as("names"))

class Patches_ColorMapModule(ColorMapModule):
    def __call__(self) -> tuple[str, str]:
        """
        Returns the color map prompt and patches string.
        """
        return super().__call__(get_color_map_as("patches"))
    
class ClassSplitted_ColorMapModule(ColorMapModule):
    def __call__(self) -> str:
        """
        Returns the class-splitted color map prompt string.
        """
        text_prompt, _ = super().__call__(None) # no color map item needed
        return text_prompt

### 3. Input Format ###

class InputFormatModule(PromptModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the input format module with a specific variation.

        Args:
            variation: Name of the input format variation.
        """
        super().__init__(f"3_input_format/{variation}")
    def __call__(self) -> str:
        """
        Returns the input format prompt string.
        """
        return super().__call__()

# Concatenated Images # 

class ConcatMasks_Sc_Hz_InputFormatModule(InputFormatModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the concatenated masks (scene, horizontal) input format module.

        Args:
            variation: Name of the input format variation.
        """
        super().__init__(variation=f"{variation}/concat_sc_hz")
        self._layout_ = "concat"
        self._scene_mode_ = "yes"
        self._align_ = "horizontal"

class ConcatMasks_Sc_Vr_InputFormatModule(InputFormatModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the concatenated masks (scene, vertical) input format module.

        Args:
            variation: Name of the input format variation.
        """
        super().__init__(variation=f"{variation}/concat_sc_vr")
        self._layout_ = "concat"
        self._scene_mode_ = "yes"
        self._align_ = "vertical"

class ConcatMasks_Ovr_Hz_InputFormatModule(InputFormatModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the concatenated masks (overlay, horizontal) input format module.

        Args:
            variation: Name of the input format variation.
        """
        super().__init__(variation=f"{variation}/concat_ovr_hz")
        self._layout_ = "concat"
        self._scene_mode_ = "overlay"
        self._align_ = "horizontal"

class ConcatMasks_Ovr_Vr_InputFormatModule(InputFormatModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the concatenated masks (overlay, vertical) input format module.

        Args:
            variation: Name of the input format variation.
        """
        super().__init__(variation=f"{variation}/concat_ovr_vr")
        self._layout_ = "concat"
        self._scene_mode_ = "overlay"
        self._align_ = "vertical"

class ConcatMasks_NoSc_Hz_InputFormatModule(InputFormatModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the concatenated masks (no scene, horizontal) input format module.

        Args:
            variation: Name of the input format variation.
        """
        super().__init__(variation=f"{variation}/concat_noSc_hz")
        self._layout_ = "concat"
        self._scene_mode_ = "no"
        self._align_ = "horizontal"

class ConcatMasks_NoSc_Vr_InputFormatModule(InputFormatModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the concatenated masks (no scene, vertical) input format module.

        Args:
            variation: Name of the input format variation.
        """
        super().__init__(variation=f"{variation}/concat_noSc_vr")
        self._layout_ = "concat"
        self._scene_mode_ = "no"
        self._align_ = "vertical"

# Separated Images # 

class SepMasks_NoSc_InputFormatModule(InputFormatModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the separated masks (no scene) input format module.

        Args:
            variation: Name of the input format variation.
        """
        super().__init__(variation=f"{variation}/sep_noSc")
        self._layout_ = "separate"
        self._scene_mode_ = "no"
        self._align_ = None

class SepMasks_Ovr_InputFormatModule(InputFormatModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the separated masks (overlay) input format module.

        Args:
            variation: Name of the input format variation.
        """
        super().__init__(variation=f"{variation}/sep_ovr")
        self._layout_ = "separate"
        self._scene_mode_ = "overlay"
        self._align_ = None

class SepMasks_Sc_InputFormatModule(InputFormatModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the separated masks (scene) input format module.

        Args:
            variation: Name of the input format variation.
        """
        super().__init__(variation=f"{variation}/sep_sc")
        self._layout_ = "separate"
        self._scene_mode_ = "yes"
        self._align_ = None

# Arrays # 

class ArrayMasks_InputFormatModule(InputFormatModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the array masks input format module.

        Args:
            variation: Name of the input format variation.
        """
        super().__init__(variation=f"{variation}/array_noImgs")
        self._layout_ = "array"
        self._scene_mode_ = "no"
        self._align_ = None

class ArrayMasks_Imgs_InputFormatModule(InputFormatModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the array masks with images input format module.

        Args:
            variation: Name of the input format variation.
        """
        super().__init__(variation=f"{variation}/array_imgs")
        self._layout_ = "array_with_imgs"
        self._scene_mode_ = "no"
        self._align_ = None

class ArrayMasks_Imgs_Ovr_InputFormatModule(InputFormatModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the array masks with images (overlay) input format module.

        Args:
            variation: Name of the input format variation.
        """
        super().__init__(variation=f"{variation}/array_imgs_ovr")
        self._layout_ = "array_with_imgs"
        self._scene_mode_ = "overlay"
        self._align_ = None

### 4. Task ###

class TaskModule(PromptModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the task module with a specific variation.

        Args:
            variation: Name of the task variation.
        """
        super().__init__(f"4_task/{variation}")
    def __call__(self) -> str:
        """
        Returns the task prompt string.
        """
        return super().__call__()
    
### 5. Output Format ###

class OutputFormatModule(PromptModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the output format module with a specific variation.

        Args:
            variation: Name of the output format variation.
        """
        super().__init__(f"5_output_format/{variation}")
    def __call__(self) -> Prompt:
        """
        Returns the output format prompt string.
        """
        return super().__call__()
    
### 6. Support Set ###

class SupportSetModule(PromptModule):
    def __init__(
            self,
            variation: str, 
            sup_set_idxs: list[int]
    ) -> None:
        """
        Initializes the support set module with a specific variation and support set indices.

        Args:
            variation: Name of the support set variation.
            sup_set_idxs: List of support set indices.
        """
        super().__init__(f"6_support_set/{variation}")
        self.__sup_set_idxs__ = sup_set_idxs
    def __call__(
            self,
            sup_set_items: list[int]
    ) -> Prompt:
        """
        Returns the support set prompt with the given items.

        Args:
            sup_set_items: List of support set items.
        Returns:
            Prompt containing the support set examples.
        """
        prompt = []
        if len(sup_set_items) != 0:
            prompt.append(super().__call__())
            for i, item in enumerate(sup_set_items):
                prompt.append(f"EXAMPLE {i+1}.")
                prompt.append(item)
        return prompt
    
### 7. Query ###
    
class QueryModule(PromptModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the query module with a specific variation.

        Args:
            variation: Name of the query variation.
        """
        super().__init__(f"7_query/{variation}")
    def __call__(
            self,
            query_item: Image.Image
    ) -> Prompt:
        """
        Returns the query prompt with the given item.

        Args:
            query_item: Query item to include in the prompt.
        Returns:
            Prompt containing the query example.
        """
        prompt = []
        prompt.append(super().__call__())
        prompt.append(query_item)
        return prompt

### 8. Evaluation ###

class EvalModule(PromptModule):
    def __init__(
            self,
            variation: str
    ) -> None:
        """
        Initializes the evaluation module with a specific variation.

        Args:
            variation: Name of the evaluation variation.
        """
        super().__init__(f"8_eval/{variation}")
    def __call__(
            self,
            target: str,
            answer: str
    ) -> None:
        """
        Returns the evaluation prompt with the given target and answer.

        Args:
            target: Target answer string.
            answer: Predicted answer string.
        Returns:
            Formatted evaluation prompt string.
        """
        return pf.pformat(super().__call__(), target=target, answer=answer)
    
### Prompts Logic ###

class PromptBuilder():

    # TODO: can we parallelise or speed up the methods that build the prompts?

    def __init__(
            self,
            seg_dataset: VOC2012SegDataset,
            by_model: str,
            alpha: float,
            split_by: str,
            image_size: int | tuple[int, int],
            array_size: int | tuple[int, int],
            class_map: dict,
            color_map: dict
    ) -> None:
        self.seg_dataset = seg_dataset
        
        # Attributes to inherit from modules
        self.layout = None
        self.scene_mode = None
        self.align = None

        # Proper attributes
        self.by_model = by_model
        self.alpha = alpha
        self.image_size = image_size
        self.array_size = array_size
        self.class_map = class_map
        self.color_map = color_map
        self.split_by = split_by

        # sets the shared static root path for the prompts
        PromptModule.prompts_path = get_prompts_path(self.split_by)

    def read_sc_gt_pr(
            self,
            idx: int,
            resize_size: int | tuple[int, int]
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        """
        Reads the scene, ground truth and prediction masks from disk and formats them in RGB format ready to be visualised.
        The color map is applied to the masks.

        Args:
            idx: Index of the image.
            image_size_: Desired image size.

        Returns:
            Tuple of scene, ground truth, and prediction PIL Image objects.
        """
        sc, gt, pr = self.seg_dataset[idx]
        sc = TF.resize(sc, size=resize_size, interpolation=TF.InterpolationMode.BILINEAR)
        gt = TF.resize(gt, size=resize_size, interpolation=TF.InterpolationMode.NEAREST)
        pr = TF.resize(pr, size=resize_size, interpolation=TF.InterpolationMode.NEAREST)
        sc = to_pil_image(sc)
        gt = to_pil_image(apply_colormap([gt], self.color_map).squeeze(0))
        pr = to_pil_image(apply_colormap([pr], self.color_map).squeeze(0))

        assert sc.size == gt.size == pr.size
        return sc, gt, pr

    def get_state(self) -> dict:
        """
        Returns the current state of the prompter as a dictionary.
        """
        def to_dict(obj: Any) -> dict | str:
            if hasattr(obj, "__to_dict__"):
                return obj.__to_dict__()
            else:
                return obj.__repr__()
        
        state = {}
        for attr, value in vars(self).items():
                state[attr] = value

        formatted_json = json.dumps(state, default=to_dict)
        state = json.loads(formatted_json)
        return state
    
    def build_img_prompt(
            self,
            img_idx: int,
            with_answer_gt: bool = False
    ) -> Prompt:
        """
        Receives an image idx and optionally its target, and builds the individual formatted image prompt.
        The query and few-shot modules make use of this method to serve the images.

        Args:
            img_idx: Index of the image to build the prompt for.
            with_answer_gt: Whether to include the ground truth answer in the prompt.

        Returns:
            Formatted image prompt.
        """
        img_prompts = []
        img_prompts.append(f"Input:")
        if self.layout not in ["array", "array_with_imgs"]:
            sc, gt, pr = self.read_sc_gt_pr(img_idx, self.image_size)
            sc, gt, pr = _format_images(sc, gt, pr, img_idx, self.layout, self.scene_mode, self.align, self.alpha)
        if self.layout == "array":
            sc, gt, pr = self.read_sc_gt_pr(img_idx, self.array_size)
            sc, gt, pr = _format_images(sc, gt, pr, img_idx, self.layout, self.scene_mode, self.align, self.alpha)
        if self.layout == "concat":
            concat_img = sc
            if self.scene_mode == "yes":
                img_prompts.append("Scene, Ground Truth and Prediction.")
                img_prompts.append(concat_img)
            else:
                img_prompts.append("Ground Truth and Prediction.")
                img_prompts.append(concat_img)
        if self.layout in ["separate", "array"]:
            if self.scene_mode == "yes":
                img_prompts.append("Scene.")
                img_prompts.append(sc)
            img_prompts.append("Ground Truth.")
            img_prompts.append(gt)
            img_prompts.append("Prediction.")
            img_prompts.append(pr)
        if self.layout == "array_with_imgs":
            sc, gt, pr = self.read_sc_gt_pr(img_idx, self.image_size)
            _, gt_img, pr_img = _format_images(sc, gt, pr, img_idx, "separate", self.scene_mode, self.align, self.alpha)
            sc, gt, pr = self.read_sc_gt_pr(img_idx, self.array_size)
            _, gt_arr, pr_arr = _format_images(sc, gt, pr, img_idx, "array", "no", self.align, self.alpha)
            img_prompts.append("Ground Truth.")
            img_prompts.append(gt_img)
            img_prompts.append(gt_arr)
            img_prompts.append("Prediction.")
            img_prompts.append(pr_img)
            img_prompts.append(pr_arr)

        img_prompts.append(f"Output:")
        if with_answer_gt is True:
            if self.split_by == "non-splitted":
                answer_gt = get_one_answer_gt(self.by_model, img_idx, format_to_dict=True)[img_idx]
            elif self.split_by == "class-splitted":
                answer_gt = get_one_sup_set_answer_gt(self.by_model, img_idx, format_to_dict=True)[img_idx]
            img_prompts.append(answer_gt) # add target answer if specified
        return flatten_list(img_prompts)

    def inherit_settings_from_modules(self) -> None:
        """
        Inherit variables enclosed by '_' from the modules in 'self.modules_dict'.
        """
        for m in self.modules_dict.values():
            for attr, value in vars(m).items():
                if attr.startswith('_') and attr.endswith('_'):
                    setattr(self, attr.strip('_'), value)
    
    def load_modules(
            self,
            context_module: ContextModule,
            color_map_module: ColorMapModule,
            input_format_module: InputFormatModule,
            task_module: TaskModule,
            output_format_module: OutputFormatModule,
            support_set_module: SupportSetModule,
            query_module: QueryModule,
            eval_module: EvalModule
    ) -> None:
        """
        Load the prompt modules (in a strict order) and inherits their shared variables.

        Args:
            context_module: Instance of ContextModule.
            color_map_module: Instance of ColorMapModule.
            input_format_module: Instance of InputFormatModule.
            task_module: Instance of TaskModule.
            output_format_module: Instance of OutputFormatModule.
            support_set_module: Instance of SupportSetModule.
            query_module: Instance of QueryModule.
            eval_module: Instance of EvalModule.
        """
        assert issubclass(type(context_module), ContextModule)
        assert issubclass(type(color_map_module), ColorMapModule)
        assert issubclass(type(input_format_module), InputFormatModule)
        assert issubclass(type(task_module), TaskModule)
        assert issubclass(type(output_format_module), OutputFormatModule)
        assert issubclass(type(support_set_module), SupportSetModule)
        assert issubclass(type(query_module), QueryModule)
        assert issubclass(type(eval_module), EvalModule)

        if self.split_by == "class-splitted" and not isinstance(color_map_module, ClassSplitted_ColorMapModule):
            raise AttributeError("When splitting by class, the color map module must be of type 'ClassSplitted_ColorMapModule'.")

        self.modules_dict = {
            "context": context_module,
            "color_map": color_map_module,
            "input_format": input_format_module,
            "task": task_module,
            "output_format": output_format_module,
            "support_set": support_set_module,
            "query": query_module,
            "eval": eval_module,
        }

        # Inherit shared variables from modules
        self.inherit_settings_from_modules()

    def build_inference_prompt(
            self,
            query_idx: int
    ) -> Prompt:
        """
        Builds the inference prompt (in which the VLM performs the differencing in textual form).

        Args:
            query_idx: Index of the query image.

        Returns:
            Formatted inference prompt.
        """
        sup_set_items = [self.build_img_prompt(idx, with_answer_gt=True) for idx in self.sup_set_idxs]
        query_item = self.build_img_prompt(query_idx)
        prompt = [
            self.modules_dict["context"](),
            self.modules_dict["color_map"](),
            self.modules_dict["input_format"](),
            self.modules_dict["task"](),
            self.modules_dict["output_format"](),
            self.modules_dict["support_set"](sup_set_items),
            self.modules_dict["query"](query_item)]
        return flatten_list(prompt)
    
    def create_class_specific_promptBuilder(
            self,
            pos_class: int
    ) -> Self:
        """
        Creates a clone of this 'PromptBuilder' class, but changes the color map so that:
        - The positive class is white (255, 255, 255).
        - All the other classes are black (0, 0, 0).

        Args:
            pos_class: Class index to be treated as the positive class.

        Returns:
            Class-specific PromptBuilder instance.
        """
        class_specific_promptBuilder = deepcopy(self)
        class_specific_promptBuilder.color_map = {c: [255, 255, 255] if c == pos_class else [0, 0, 0] for c in range(self.seg_dataset.get_num_classes(with_unlabelled=False))}
        return class_specific_promptBuilder
    
    def build_class_splitted_support_set_items(self) -> Prompt:
        """
        Generates the few-shot example items in the class-splitted scenario.
        """
        sup_set_items = []
        for idx in self.sup_set_idxs:
            sup_set_items.append(self.build_class_specific_support_set_item(idx))
        return sup_set_items
    
    def build_class_specific_support_set_item(
            self,
            img_idx: int
    ) -> Prompt:
        """
        Generates a single few-shot example in a class-specific scenario.
        In this one, since the items have to class-specific, each image is associated with a single class index, marking the class the few-shot is going to display.
        However, the actual evaluation text is hard-coded (can be adapted in future to select change dynamically the positive class).

        Args:
            img_idx: Index of the image to generate the support set item for.

        Returns:
            Formatted class-specific support set item.
        """
        img_idx_to_class_ = {
            2: 20,
            16: 1,
            18: 13,
        }
        class_specific_promptBuilder = self.create_class_specific_promptBuilder(img_idx_to_class_[img_idx])
        return class_specific_promptBuilder.build_img_prompt(img_idx, with_answer_gt=True)
    
    def build_class_specific_inference_prompt(
            self,
            query_idx: int,
            pos_class: int
    ) -> Prompt:
        """
        Builds the inference prompt (in which the VLM performs the differencing in textual form) in a class-specific scenario.

        Args:
            query_idx: Index of the query image.
            pos_class: Class index to be treated as the positive class.

        Returns:
            Formatted class-specific inference prompt.
        """
        significant_class_name = self.seg_dataset.get_classes(with_unlabelled=False)[pos_class]
        sup_set_items = self.build_class_splitted_support_set_items()
        class_specific_promptBuilder = self.create_class_specific_promptBuilder(pos_class)
        query_item = class_specific_promptBuilder.build_img_prompt(query_idx)
        class_specific_prompt = [
            self.modules_dict["context"](),
            self.modules_dict["color_map"](),
            self.modules_dict["input_format"](),
            self.modules_dict["task"](),
            self.modules_dict["output_format"](),
            self.modules_dict["support_set"](sup_set_items),
            self.modules_dict["query"](query_item)]
        class_specific_prompt = flatten_list(class_specific_prompt)
        class_specific_prompt = [pf.pformat(s, pos_class=significant_class_name) if isinstance(s, str) else s for s in class_specific_prompt]
        return class_specific_prompt

    def build_eval_prompt(
            self,
            query_idx: int,
            answer_pr: str
    ) -> Prompt:
        """
        Builds the evaluation prompt (in which the LLM-as-a-Judge evaluates the differencing in textual form).

        Args:
            query_idx: Index of the query image.
            answer_pr: Predicted answer string.

        Returns:
            Evaluation prompt.
        """
        answer_gt = get_one_answer_gt(self.by_model, query_idx)['content']
        prompt = [self.modules_dict["eval"](answer_gt, answer_pr)]
        return prompt
    
    def build_class_splitted_inference_prompts(
            self,
            query_idx: int
    ) -> dict[int, Prompt]:
        """
        Builds a list of full inference prompts for a given 'query_idx' split by class masks. 
        Each inference prompt masks only consider one class at a time.

        Args:
            query_idx: Index of the query image.

        Returns:
            Dictionary of class-splitted inference prompts.
        """
        # FIXME: if the masks only have BACKGROUND class, there might be an error when trying to build the prompt.
        _, gt, pr = self.seg_dataset[query_idx]
        significant_classes_gt = get_significant_classes(gt)
        significant_classes_pr = get_significant_classes(pr)
        significant_classes = sorted(list(set(significant_classes_gt + significant_classes_pr))) # all appearing classes
        if significant_classes != [0]:
            significant_classes.remove(0)
        class_splitted_prompts = {}
        for pos_class in significant_classes:
            class_specific_prompt = self.build_class_specific_inference_prompt(query_idx, pos_class)
            class_splitted_prompts[pos_class] = class_specific_prompt
        return class_splitted_prompts

    def build_class_splitted_eval_prompt(
            self,
            query_idx: int,
            pos_class_2_answer_pr: dict[int, str]
    ) -> dict[int, Prompt]:
        """
        Builds a list of full evaluation prompts for a given 'query_idx' split by class masks. 
        Each evaluation prompt masks only consider one class at a time.

        Args:
            query_idx: Index of the query image.
            pos_class_2_answer_pr: Dictionary mapping class positions to predicted answers.

        Returns:
            Dictionary of class-splitted evaluation prompts.
        """
        pos_class_2_eval_prompt = {}
        significant_classes = pos_class_2_answer_pr.keys()
        class_splitted_answer_pr = pos_class_2_answer_pr.values()
        for pos_class, answer_pr in zip(significant_classes, class_splitted_answer_pr):
            pos_class = int(pos_class)
            pos_class_2_eval_prompt[pos_class] = [pf.pformat(self.build_eval_prompt(query_idx, answer_pr)[0], pos_class=self.seg_dataset.get_classes(with_unlabelled=False)[pos_class])]
        return pos_class_2_eval_prompt
    
# 'with_unlabelled' in both Prompters should be given as argument to __init__ and propagated to all other functions.

def get_significant_classes(
        input_tensor: torch.Tensor,
        batched: bool = False,
) -> torch.Tensor:
    if batched:
        return [img_t.unique().tolist() for img_t in input_tensor]
    else:
        return input_tensor.unique().tolist()

class FastPromptBuilder:
    def __init__(
            self,
            seg_dataset: VOC2012SegDataset,
            seed: int,
            prompt_blueprint: OrderedDict[str, str],
            by_model: str,
            alpha: float,
            class_map: dict[int, int],
            color_map: dict[int, tuple[int, int, int]],
            image_size: int | tuple[int, int],
            sup_set_seg_dataset: VOC2012SegDataset,
            sup_set_img_idxs: list[int],
            str_formats: dict[str, str],
    ) -> None:
        """
        Initializea GenParams with optional generation parameters.

        Args:
            seed: Random seed for generation.
            shuffle_seeds: Whether to shuffle the examples.
        """
        self.seg_dataset = seg_dataset
        self.seed = seed
        self.prompt_blueprint = prompt_blueprint
        self.by_model = by_model
        self.alpha = alpha
        self.class_map = class_map
        self.color_map = color_map
        self.image_size = image_size
        self.str_formats = str_formats
        self.sup_set_img_idxs = sup_set_img_idxs
        self.sup_set_seg_dataset = sup_set_seg_dataset
        
        base_head_prompt = self.build_cs_base_prompt(self.sup_set_img_idxs, alpha)
        self.base_prompt = base_head_prompt[:-1]
        self.head_prompt = base_head_prompt[-1]

    def build_base_prompt_(
            self,
            sup_set_imgs: list[tuple[Image.Image, Image.Image]],
            sup_set_answers: list[str]
    ) -> Prompt:
        
        def populate_sup_set(
            sup_set_module: str,
            sup_set_imgs: list[tuple[Image.Image, Image.Image]],
            sup_set_answers: list[str]
        ) -> Prompt:
            expanded_sup_set_module = sup_set_module*len(sup_set_imgs) # replicate the placeholder for each support set example.
            expanded_sup_set_module = map_list_placeholders(expanded_sup_set_module, placeholder="[sup_set_count]", objects_list=[str(n+1) for n in range(len(sup_set_imgs))])
            expanded_sup_set_module = map_list_placeholders(expanded_sup_set_module, placeholder="[answer_gt]", objects_list=sup_set_answers)
            expanded_sup_set_module = map_list_placeholders(expanded_sup_set_module, placeholder="[img]", objects_list=flatten_list(sup_set_imgs))
            return expanded_sup_set_module
        
        prompt_corpus = read_json(get_data_gen_prompts_path() / "fast_cs_prompt.json")
        prompt = [prompt_corpus[mod][var] for mod, var in self.prompt_blueprint.items()]
        
        if len(sup_set_imgs) != 0:
            sup_set_module_idx = list(self.prompt_blueprint.keys()).index("support_set_item")
            prompt[sup_set_module_idx] = populate_sup_set(prompt[sup_set_module_idx], sup_set_imgs, sup_set_answers)

        if self.str_formats:
            prompt = substitute_list_placeholders(prompt, self.str_formats)
        
        return prompt

    def build_cs_base_prompt(
            self,
            sup_set_img_idxs: list[int],
            alpha: Optional[float]
    ) -> Prompt:
        img_idx_to_class_ = {
            2: 20,
            16: 1,
            18: 13,
        }

        # WARNING: this implementation sets the the same color map for all the pos. classes (the values of the dict above).
        # This way, the color mapping can be computed in parallel (faster) and the code is simpler.
        # This method works only if the sup sets have disjointed classes. Ideally, each support example would have its own color map.
        color_map = {pos_c: (255, 255, 255) for pos_c in img_idx_to_class_.values()}

        # image_UIDs = get_image_UIDs(SPLITS_PATH, split="trainval", shuffle=True)

        # sup_set_uids = [uid for uid in self.seg_dataset.image_UIDs[sup_set_img_idxs]]
        # sup_set_uids = image_UIDs[sup_set_img_idxs]
        # gts_paths = [GTS_PATH / (UID + ".png") for UID in sup_set_uids]
        # prs_paths = [self.prs_mask_paths / f"mask_pr_{i}.png" for i in sup_set_img_idxs]
        # scs_paths = [SCS_PATH / (UID + ".jpg") for UID in sup_set_uids]
        
        scs, gts, prs = self.sup_set_seg_dataset[sup_set_img_idxs]

        scs = torch.stack(scs)
        gts = torch.stack(gts)
        prs = torch.stack(prs)

        scs = TF.resize(scs, self.image_size, TF.InterpolationMode.BILINEAR)
        gts = TF.resize(gts, self.image_size, TF.InterpolationMode.NEAREST)
        prs = TF.resize(prs, self.image_size, TF.InterpolationMode.NEAREST)
        
        # gts = torch.stack([get_gt(p, class_map=self.class_map, resize_size=self.image_size, center_crop=True) for p in gts_paths])
        # prs = torch.stack([get_pr(p, class_map=self.class_map, resize_size=self.image_size, center_crop=True) for p in prs_paths]) # tensors (3, H, W)
        # scs = torch.stack([get_sc(p, resize_size=self.image_size, center_crop=True) for p in scs_paths])

        gts = apply_colormap(gts, color_map)
        prs = apply_colormap(prs, color_map)
        
        if alpha:
            alpha_tensor = torch.full(gts.size(), alpha, device=gts.device)
            gts = blend_tensors(scs, gts, alpha_tensor)
            prs = blend_tensors(scs, prs, alpha_tensor)

        gts_img = [to_pil_image(col_gt) for col_gt in gts]
        prs_img = [to_pil_image(col_pr) for col_pr in prs]

        sup_set_imgs = list(zip(gts_img, prs_img))
        sup_set_answers = [get_one_sup_set_answer_gt(self.by_model, sup_set_img_idx, return_state=False, format_to_dict=False)['content'] for sup_set_img_idx in self.sup_set_img_idxs]

        base_prompt = self.build_base_prompt_(sup_set_imgs, sup_set_answers)

        return base_prompt

    def extract_significant_classes(
            self,
            query_gt: torch.Tensor,
            query_pr: torch.Tensor,
    ) -> list[int]:
        gt_sign_classes = get_significant_classes(query_gt)
        pr_sign_classes = get_significant_classes(query_pr)
        sign_classes = list(set(gt_sign_classes + pr_sign_classes))

        # Remove the BACKGROUND class only if it is not the only one.
        # Cropping can leave out all meaningful classes: in this case, the BACKGROUND class is considered positive
        # and both masks are completely white.
        if 0 in sign_classes and sign_classes != [0]:
            sign_classes.remove(0)

        return sorted(sign_classes)
    
    def expand_head_to_cs(
            self,
            query_gt: torch.Tensor,
            query_pr: torch.Tensor,
            query_sc: Optional[torch.Tensor],
            alpha: Optional[float]
    ) -> list[Prompt]:
        
        sign_classes = self.extract_significant_classes(query_gt, query_pr)

        concat_masks = torch.stack([query_gt, query_pr], dim=0)

        splitted_masks = torch.stack([apply_colormap(concat_masks, {pos_c: (255, 255, 255)}) for pos_c in sign_classes], dim=0)

        if alpha:
            splitted_masks = blend_tensors(query_sc, splitted_masks, alpha)

        splitted_gt_masks = splitted_masks[:, 0, ...]
        splitted_pr_masks = splitted_masks[:, 1, ...]

        # return splitted_gt_masks, splitted_pr_masks, sign_classes
        return {pos_c: [to_pil_image(splitted_gt_masks[i]), to_pil_image(splitted_pr_masks[i])] for i, pos_c in enumerate(sign_classes)}
    
    def populate_query(
            self,
            query_module: str,
            query_gt: Image.Image,
            query_pr: Image.Image,
    ) -> Prompt:
        return map_list_placeholders(query_module, placeholder="[img]", objects_list=[query_gt, query_pr])
    
    def build_cs_inference_prompts(
            self,
            gts_tensor: torch.Tensor,
            prs_tensor: torch.Tensor,
            scs_tensor: torch.Tensor,
    ) -> list[Prompt]:
        
        gts_tensor = TF.resize(gts_tensor, self.image_size, TF.InterpolationMode.NEAREST)
        prs_tensor = TF.resize(prs_tensor, self.image_size, TF.InterpolationMode.NEAREST)
        scs_tensor = TF.resize(scs_tensor, self.image_size, TF.InterpolationMode.BILINEAR)

        splitted_elements_dict = [self.expand_head_to_cs(gt, pr, sc, self.alpha) for gt, pr, sc in zip(gts_tensor, prs_tensor, scs_tensor)]

        cs_gts_imgs_list = [[gt for pos_c, (gt, pr) in cs_elements.items()] for cs_elements in splitted_elements_dict]
        cs_prs_imgs_list = [[pr for pos_c, (gt, pr) in cs_elements.items()] for cs_elements in splitted_elements_dict]
        cs_pos_c_list = [[pos_c for pos_c, (gt, pr) in cs_elements.items()] for cs_elements in splitted_elements_dict]

        zipped_gts_prs_list = list(zip(cs_gts_imgs_list, cs_prs_imgs_list, cs_pos_c_list))

        cs_prompts_list = [{pos_c: flatten_list(self.base_prompt + self.populate_query(deepcopy(self.head_prompt), gt_img, pr_img)) for gt_img, pr_img, pos_c in zip(cs_gts, cs_prs, cs_pos_c)} for cs_gts, cs_prs, cs_pos_c in zipped_gts_prs_list]
        cs_prompts_list = [{pos_c: [pf.pformat(piece, pos_class=self.seg_dataset.get_classes(with_unlabelled=False)[pos_c]) if isinstance(piece, str) else piece for piece in prompt] for pos_c, prompt in cs_prompts.items()} for cs_prompts in cs_prompts_list]
        
        return cs_prompts_list
    
    def build_cs_inference_prompts_from_disk(
            self,
            query_idxs: list[int]
    ) -> list[Prompt]:
        # image_UIDs = get_image_UIDs(SPLITS_PATH, split="trainval", shuffle=True)
        # query_uids = image_UIDs[query_idxs]
        
        scs, gts, prs = self.seg_dataset[query_idxs]

        scs = torch.stack(scs)
        gts = torch.stack(gts)
        prs = torch.stack(prs)

        # gts_paths = [GTS_PATH / (UID + ".png") for UID in query_uids]
        # prs_paths = [self.prs_mask_paths / f"mask_pr_{i}.png" for i in query_idxs]
        # scs_paths = [SCS_PATH / (UID + ".jpg") for UID in query_uids]
        # gts = torch.stack([get_gt(p, class_map=self.class_map, resize_size=self.image_size, center_crop=True) for p in gts_paths])
        # prs = torch.stack([get_pr(p, class_map=self.class_map, resize_size=self.image_size, center_crop=True) for p in prs_paths]) # tensors (3, H, W)
        # scs = torch.stack([get_sc(p, resize_size=self.image_size, center_crop=True) for p in scs_paths])
        
        cs_prompts_list =  self.build_cs_inference_prompts(gts, prs, scs)

        return cs_prompts_list
    
    def get_state(self) -> dict:
        """
        Returns the current state of the prompter as a dictionary.
        """
        def to_dict(obj: Any) -> dict | str:
            if hasattr(obj, "__to_dict__"):
                return obj.__to_dict__()
            else:
                return obj.__repr__()
        
        state = {}
        for attr, value in vars(self).items():
                state[attr] = value

        formatted_json = json.dumps(state, default=to_dict)
        state = json.loads(formatted_json)
        return state
    
def save_formatted_images(
        promptBuilder: PromptBuilder,
        img_idxs: tuple[int]
) -> None:
    """
    Saves formatted images for the given indices using the provided prompt builder.

    Args:
        promptBuilder: Instance of PromptBuilder.
        img_idxs: Tuple of image indices to process.
    """
    for img_idx in img_idxs:
        sc, gt, pr = promptBuilder.read_sc_gt_pr(img_idx, promptBuilder.image_size)
        formatted_image = _format_images(sc, gt, pr, img_idx, promptBuilder.layout, promptBuilder.scene_mode, promptBuilder.align, promptBuilder.alpha)[0]
        formatted_image.save(LOCAL_ANNOT_IMGS_PATH / f"annot_image_{img_idx}.png")

def main() -> None:
    ...

if __name__ == '__main__':
    main()
