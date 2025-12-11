"""Modular prompt construction system for vision-language model evaluation of semantic segmentation.

Provides PromptModule subclasses for context, color maps, input formatting, task description,
output format, support sets, queries, and evaluation. Includes PromptBuilder for assembling
complete prompts and FastPromptBuilder for optimized batch processing in class-splitted scenarios.
"""

from core.config import *
from core.data_utils import flatten_list
from core.torch_utils import blend_tensors
from core.datasets import VOC2012SegDataset, get_answer_objects, JsonlIO
from core.data import read_txt, read_json
from core.color_map import get_color_map_as, apply_colormap, pil_to_class_array

from PIL import Image, ImageFont, ImageDraw, ImageOps
from pathlib import Path
import json
import pformat as pf
from copy import deepcopy
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as TF
from collections import OrderedDict
import re

from core._types import Self, Any, Optional, Callable, Prompt, RGB_tuple

def _concat_images_fn(
        images: list[Image.Image],
        titles: list[str],
        scene_mode: str, 
        align: str,
        font_path: Path
) -> Image.Image:
    """Concatenate multiple images with titles into a single composite image.

    Creates a composite image by arranging multiple input images either horizontally
    or vertically, with each image labeled by a title. The images are resized to have
    consistent dimensions in the alignment direction, padded with white borders, and
    separated by consistent spacing.

    The function handles different scene modes:
    - "no" or "overlay": Only includes the last 2 images (GT and prediction)
    - Other modes: Includes all 3 images (scene, GT, and prediction)

    Args:
        images: List of exactly 3 PIL Image objects to concatenate. Expected order is
            [scene, ground_truth, prediction].
        titles: List of exactly 3 title strings corresponding to each image.
        scene_mode: Scene display mode. Valid values are "no", "overlay", or "yes".
            Determines which images are included in the final concatenation.
        align: Alignment direction for concatenation. Must be either "horizontal" or "vertical".
        font_path: Path to a TrueType font file (.ttf) used for rendering titles.

    Returns:
        A PIL Image object containing the concatenated images with titles.

    Raises:
        ValueError: If align is not "horizontal" or "vertical".
        AssertionError: If images or titles lists don't contain exactly 3 elements.

    Note:
        - Images are resized to maintain aspect ratio while matching dimensions in the alignment direction
        - Title height is fixed at 40 pixels with 32pt font size
        - Padding around each image is 20 pixels
        - Border spacing between images is 10 pixels
        - Background and padding colors are white
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
    font = ImageFont.truetype(font_path, size=32)

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
    """Format and arrange scene, ground truth, and prediction images based on layout specifications.

    Applies formatting transformations to input images based on the specified layout and scene mode.
    Supports alpha blending for overlay modes, image concatenation, and conversion to class arrays.

    Args:
        sc: Scene/input RGB image.
        gt: Ground truth segmentation mask (colorized).
        pr: Prediction segmentation mask (colorized).
        idx: Index identifier for the image (used in concatenated layout titles).
        layout: Formatting layout. Valid values:
            - 'concat': Concatenates all images into one composite image
            - 'separate': Keeps images separate
            - 'array': Converts masks to class index arrays (numpy format)
        scene_mode: How to display the scene image. Valid values:
            - 'yes': Include scene as separate image
            - 'no': Exclude scene image
            - 'overlay': Blend masks with scene using alpha transparency
        align: Alignment direction for concatenated layout ('horizontal' or 'vertical').
            Only used when layout='concat'.
        alpha: Alpha blending factor (0.0 to 1.0) for overlay mode. Higher values give
            more weight to the mask. Only used when scene_mode='overlay'.

    Returns:
        A tuple of three formatted images (sc, gt, pr). The specific content depends on layout:
        - concat: All three elements are the same concatenated image
        - separate: Original images, potentially alpha-blended if scene_mode='overlay'
        - array: (sc, gt_array, pr_array) where arrays are numpy class indices

    Note:
        - When layout='concat', the function internally calls _concat_images_fn
        - Font path for concatenation is hardcoded to Path("resources/Arial.ttf")
        - For 'array' layout, pil_to_class_array is used to convert colorized masks to class indices
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
    """Replace placeholders in a text string with objects from a list.

    Splits a text string containing placeholders into a list that interleaves text segments
    with the provided objects. Each placeholder in the text is replaced in order with the
    corresponding object from objects_list.

    Args:
        text: Input text string containing placeholder symbols.
        placeholder: The placeholder symbol to search for and replace (e.g., '[img]', '[mask]').
        objects_list: List of objects to substitute for placeholders. Can contain any type
            (images, arrays, strings, etc.).

    Returns:
        A list containing interleaved text segments and objects. For example:
        - Input: "Hello [img] world", placeholder='[img]', objects_list=[Image]
        - Output: ['Hello ', Image, ' world']

    Raises:
        AttributeError: If the number of placeholders in text doesn't match the length
            of objects_list.

    Note:
        - If no placeholders are found, returns the original text string unchanged
        - Empty strings from splitting are filtered out
        - Placeholders are matched using exact string comparison
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
    texts: list[str],
    placeholder: str,
    objects_list: list[Any]
) -> list[list[Interleaved]]:
    """Replace placeholders across multiple text strings with objects, intelligently merging strings.

    Processes a list of text strings, replacing placeholders with corresponding objects from
    objects_list. The function intelligently handles string vs non-string objects:
    - String objects are merged with adjacent text
    - Non-string objects (images, arrays, etc.) are kept as separate list elements

    This ensures that prompts remain cleanly formatted when combining text with multimodal
    elements like images.

    Args:
        texts: List of text strings that may contain placeholder symbols.
        placeholder: The placeholder symbol to search for and replace (e.g., '[img]', '[mask]').
        objects_list: List of objects to substitute for placeholders in sequential order.
            Can contain strings, images, arrays, or any other objects.

    Returns:
        A flattened list containing interleaved text segments and objects. Adjacent strings
        are automatically merged. For example:
        - `map_list_placeholders(["A [P] C"], "[P]", ["B"])` → `['A B C']`
        - `map_list_placeholders(["A [P] C"], "[P]", [ImageObject])` → `['A ', ImageObject, ' C']`
        - `map_list_placeholders(["Text [P]", "[P] more"], "[P]", [img1, img2])` →
          `['Text ', img1, img2, ' more']`

    Raises:
        ValueError: If the total count of placeholders across all texts doesn't match
            the length of objects_list.

    Note:
        - The function performs validation before processing to ensure placeholder count matches
        - Empty text segments are filtered out
        - If no placeholders are found, each text is returned wrapped in its own list
        - The result is flattened at the end using flatten_list
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

    final_result: list[list[Interleaved]] = []
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
        substituted_parts: list[Interleaved] = []
        for part in parts:
            if part == placeholder:
                # This is a placeholder; get the next object for substitution.
                substituted_parts.append(next(object_iterator))
            elif part: # Only append non-empty text parts.
                substituted_parts.append(part)

        # Step 3: Merge adjacent string parts.
        processed_parts: list[Interleaved] = []
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
    texts: list[str],
    substitutions: dict[str, str]
) -> list[str]:
    """Efficiently replace multiple placeholders in a list of strings using dictionary mapping.

    Performs batch string substitution across multiple text strings using a single compiled
    regular expression for optimal performance. Unlike map_list_placeholders which handles
    arbitrary objects, this function only handles string-to-string substitutions.

    Args:
        texts: List of text strings that may contain placeholder symbols.
        substitutions: Dictionary mapping placeholder symbols (keys) to their replacement
            strings (values). For example: {'[user]': 'Alice', '[date]': '2025-10-10'}.

    Returns:
        A new list of strings with all placeholders replaced by their corresponding values.
        If substitutions is empty, returns a copy of the original list.

    Note:
        - Uses compiled regex for efficient multi-placeholder substitution
        - Placeholder keys are automatically escaped to handle special regex characters
        - All substitutions are performed in a single pass per string
        - Order of substitution is determined by the regex engine, not dict order
        - This is significantly faster than calling str.replace() multiple times

    Example:
        ```python
        texts = ["Hello [user]!", "Date: [date]"]
        subs = {"[user]": "Alice", "[date]": "2025-10-10"}
        result = substitute_list_placeholders(texts, subs)
        # result: ["Hello Alice!", "Date: 2025-10-10"]
        ```
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
    """Base class for modular prompt components that compose complete VLM prompts.

    PromptModule provides the foundation for a hierarchical prompt construction system where
    each module represents a distinct component of the final prompt (context, color map, task
    description, etc.). Subclasses customize the behavior by implementing their __call__ method
    and optionally adding module-specific attributes.

    The class uses a shared class-level prompts_path that must be initialized before use,
    allowing all modules to access prompt template files from a common directory structure.

    Class Attributes:
        prompts_path: Shared Path object pointing to the root directory containing prompt
            template files. Must be set externally before instantiating modules. Set to None
            by default.

    Instance Attributes:
        full_path: Complete path to this module's specific prompt variation file, constructed
            as prompts_path / "{variation}.txt".

    Module Hierarchy:
        1. ContextModule: Task context and background information
        2. ColorMapModule: Class-to-color mapping specification
        3. InputFormatModule: Image presentation format
        4. TaskModule: Task description and objectives
        5. OutputFormatModule: Expected output format specification
        6. SupportSetModule: Few-shot examples
        7. QueryModule: Query/test image presentation
        8. EvalModule: Evaluation criteria for LLM-as-a-Judge

    Note:
        - Subclasses should not override __to_dict__ or import_variation
        - The prompts_path must be set before creating any module instances
        - Attributes prefixed and suffixed with '_' are inherited by PromptBuilder
    """
    prompts_path = None # variable shared among all sub-classes, it has to be initialized externally when building the prompt.
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize the prompt module with a specific variation.

        Args:
            variation: Name of the prompt variation file (without .txt extension).
                The full path will be constructed as prompts_path / "{variation}.txt".
        """
        self.full_path = Path(self.prompts_path / f"{variation}.txt") # full path complete of the prompt variation.
    
    def import_variation(
            self,
            variation_path: str
    ) -> str:
        """Load the text content from the prompt variation file.

        Reads and returns the complete text content from the module's prompt file.
        This method should not be overridden by subclasses.

        Args:
            variation_path: Path to the variation file (currently unused, uses self.full_path).

        Returns:
            String content of the prompt variation file.

        Note:
            Despite accepting variation_path parameter, the method uses self.full_path
            internally for file reading.
        """
        return read_txt(self.full_path)

    def __call__(self) -> str:
        """Generate the prompt text for this module.

        Returns the text content for this prompt component. Subclasses can override
        this method to add custom logic, formatting, or to include additional objects
        (like images or data structures) alongside the text.

        Returns:
            The prompt text, or a more complex structure in subclass implementations.

        Note:
            Subclasses may return types other than str (e.g., tuple[str, Image])
            to include additional prompt components.
        """
        return self.import_variation(self.full_path)
    
    def __to_dict__(self) -> dict:
        """Serialize the module to a dictionary representation.

        Creates a dictionary containing the module's class name and all instance attributes.
        Used for saving prompt configuration state. Should not be overridden by subclasses.

        Returns:
            Dictionary with 'class' key containing class name, merged with all instance
            attributes from vars(self).
        """
        return {"class": self.__class__.__name__} | vars(self)
    
### 1. Context ###

class ContextModule(PromptModule):
    """Module for providing task context and background information in prompts.

    The context module sets the stage for the VLM by providing background information,
    task domain knowledge, and general instructions that frame the subsequent prompt components.

    Prompt files are located in: prompts_path / "1_context" / "{variation}.txt"
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize the context module with a specific variation.

        Args:
            variation: Name of the context variation file (e.g., "base", "detailed", "minimal").
        """
        super().__init__(f"1_context/{variation}")
    def __call__(self) -> str:
        """Generate the context prompt text.

        Returns:
            The context prompt string loaded from the variation file.
        """
        return super().__call__()

### 2. Color Map ###

class ColorMapModule(PromptModule):
    """Base module for specifying class-to-color mapping in segmentation prompts.

    The color map module explains to the VLM how segmentation classes map to colors in
    the masks. Different subclasses provide the mapping in various formats (image, RGB values,
    color names, patches, or class-splitted).

    Prompt files are located in: prompts_path / "2_color_map" / "{variation}.txt"

    Note:
        This is an abstract base class. Use specific subclasses like Image_ColorMapModule,
        RGB_ColorMapModule, etc. for concrete implementations.
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize the color map module with a specific variation.

        Args:
            variation: Name of the color map variation file.
        """
        super().__init__(f"2_color_map/{variation}")
    def __call__(
            self,
            color_map_item: Any
    ) -> tuple[str, Any]:
        """Generate the color map prompt with associated color mapping item.

        Args:
            color_map_item: The color map representation to include (image, text, etc.).

        Returns:
            Tuple containing the prompt text and the color map item.
        """
        return super().__call__(), color_map_item
    
class Image_ColorMapModule(ColorMapModule):
    """Color map module that provides mapping as a visual image.

    Returns a color map visualization image showing each class with its corresponding color,
    suitable for visual learners and multimodal VLMs that can process images.
    """
    def __call__(self) -> tuple[str, Image.Image]:
        """Generate color map prompt with a visual image representation.

        Returns:
            Tuple of (prompt text, color map image showing class colors).
        """
        return super().__call__(get_color_map_as("img"))

class RGB_ColorMapModule(ColorMapModule):
    """Color map module that provides mapping as RGB value strings.

    Returns color mapping as text listing RGB tuples for each class, e.g.:
    "Class 0 - Background: (0, 0, 0)\nClass 1 - Aeroplane: (128, 0, 0)\n..."
    """
    def __call__(self) -> tuple[str, str]:
        """Generate color map prompt with RGB values as text.

        Returns:
            Tuple of (prompt text, RGB color mapping string).
        """
        return super().__call__(get_color_map_as("rgb"))

class Names_ColorMapModule(ColorMapModule):
    """Color map module that provides mapping using color names.

    Returns color mapping as text using human-readable color names, e.g.:
    "Class 0 - Background: black\nClass 1 - Aeroplane: maroon\n..."
    """
    def __call__(self) -> tuple[str, str]:
        """Generate color map prompt with color names as text.

        Returns:
            Tuple of (prompt text, color names mapping string).
        """
        return super().__call__(get_color_map_as("names"))

class Patches_ColorMapModule(ColorMapModule):
    """Color map module that provides mapping as colored text patches.

    Returns color mapping as text with inline color patch representations,
    useful for rich text environments that support color formatting.
    """
    def __init__(
            self,
            variation: str,
            classes: list[str],
            color_map_dict: dict[int, RGB_tuple],
            patch_size: tuple[int, int] = (32, 32),
    ) -> None:
        # TODO this init() metho should be replicated and adapter for other ColorMapModules.
        super().__init__(variation)
        self.classes = classes
        self.color_map_dict = color_map_dict
        self.patch_size = patch_size

    def __call__(self) -> tuple[str, str]:
        """Generate color map prompt with colored patches.

        Returns:
            Tuple of (prompt text, color patches representation string).
        """
        return super().__call__(get_color_map_as("patches", classes=self.classes, color_map_dict=self.color_map_dict, patch_size=self.patch_size))
    
class ClassSplitted_ColorMapModule(ColorMapModule):
    """Color map module for class-splitted scenarios where only one class is evaluated at a time.

    In class-splitted mode, each prompt focuses on a single class with a binary color scheme:
    - Target class: white (255, 255, 255)
    - All other classes: black (0, 0, 0)

    This module returns only the text prompt without an actual color map item, as the
    color mapping is dynamically generated per class in PromptBuilder.
    """
    def __call__(self) -> str:
        """Generate color map prompt for class-splitted scenario.

        Returns:
            The prompt text only (no color map item, as it's generated per-class).
        """
        text_prompt, _ = super().__call__(None) # no color map item needed
        return text_prompt

### 3. Input Format ###

class InputFormatModule(PromptModule):
    """Base module for specifying how input images are formatted and presented.

    The input format module defines the visual layout and organization of scene images,
    ground truth masks, and prediction masks. Different subclasses implement various
    presentation strategies optimized for different VLM capabilities.

    Attributes added by subclasses (inherited by PromptBuilder via _attribute_ naming):
        _layout_: Image arrangement strategy ('concat', 'separate', 'array', 'array_with_imgs')
        _scene_mode_: Scene image inclusion mode ('yes', 'no', 'overlay')
        _align_: Concatenation direction ('horizontal', 'vertical', or None)

    Prompt files are located in: prompts_path / "3_input_format" / "{variation}" / ...

    Note:
        This is an abstract base class. Use specific subclasses for concrete implementations.
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize the input format module with a specific variation.

        Args:
            variation: Name of the input format variation file.
        """
        super().__init__(f"3_input_format/{variation}")
    def __call__(self) -> str:
        """Generate the input format prompt text.

        Returns:
            The input format description string.
        """
        return super().__call__()

# Concatenated Images # 

class ConcatMasks_Sc_Hz_InputFormatModule(InputFormatModule):
    """Input format with scene, GT, and prediction concatenated horizontally.

    Presents all three images (scene, ground truth, prediction) side-by-side in a single
    composite image with titles. Optimizes screen space horizontally.

    Attributes:
        _layout_: "concat"
        _scene_mode_: "yes"
        _align_: "horizontal"
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize concatenated horizontal format with scene.

        Args:
            variation: Base variation name (subdirectory appended automatically).
        """
        super().__init__(variation=f"{variation}/concat_sc_hz")
        self._layout_ = "concat"
        self._scene_mode_ = "yes"
        self._align_ = "horizontal"

class ConcatMasks_Sc_Vr_InputFormatModule(InputFormatModule):
    """Input format with scene, GT, and prediction concatenated vertically.

    Presents all three images stacked vertically in a single composite image with titles.
    Better for narrow displays or when horizontal space is limited.

    Attributes:
        _layout_: "concat"
        _scene_mode_: "yes"
        _align_: "vertical"
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize concatenated vertical format with scene.

        Args:
            variation: Base variation name (subdirectory appended automatically).
        """
        super().__init__(variation=f"{variation}/concat_sc_vr")
        self._layout_ = "concat"
        self._scene_mode_ = "yes"
        self._align_ = "vertical"

class ConcatMasks_Ovr_Hz_InputFormatModule(InputFormatModule):
    """Input format with alpha-blended overlay masks concatenated horizontally.

    Presents GT and prediction masks overlaid on the scene image using alpha blending,
    then concatenates horizontally. Helps VLM see masks in context of original scene.

    Attributes:
        _layout_: "concat"
        _scene_mode_: "overlay"
        _align_: "horizontal"
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize concatenated horizontal format with overlay.

        Args:
            variation: Base variation name (subdirectory appended automatically).
        """
        super().__init__(variation=f"{variation}/concat_ovr_hz")
        self._layout_ = "concat"
        self._scene_mode_ = "overlay"
        self._align_ = "horizontal"

class ConcatMasks_Ovr_Vr_InputFormatModule(InputFormatModule):
    """Input format with alpha-blended overlay masks concatenated vertically.

    Presents GT and prediction masks overlaid on the scene image using alpha blending,
    then concatenates vertically.

    Attributes:
        _layout_: "concat"
        _scene_mode_: "overlay"
        _align_: "vertical"
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize concatenated vertical format with overlay.

        Args:
            variation: Base variation name (subdirectory appended automatically).
        """
        super().__init__(variation=f"{variation}/concat_ovr_vr")
        self._layout_ = "concat"
        self._scene_mode_ = "overlay"
        self._align_ = "vertical"

class ConcatMasks_NoSc_Hz_InputFormatModule(InputFormatModule):
    """Input format with GT and prediction concatenated horizontally, no scene.

    Presents only the two masks (ground truth and prediction) side-by-side, excluding
    the scene image. Focuses attention purely on mask comparison.

    Attributes:
        _layout_: "concat"
        _scene_mode_: "no"
        _align_: "horizontal"
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize concatenated horizontal format without scene.

        Args:
            variation: Base variation name (subdirectory appended automatically).
        """
        super().__init__(variation=f"{variation}/concat_noSc_hz")
        self._layout_ = "concat"
        self._scene_mode_ = "no"
        self._align_ = "horizontal"

class ConcatMasks_NoSc_Vr_InputFormatModule(InputFormatModule):
    """Input format with GT and prediction concatenated vertically, no scene.

    Presents only the two masks (ground truth and prediction) stacked vertically,
    excluding the scene image.

    Attributes:
        _layout_: "concat"
        _scene_mode_: "no"
        _align_: "vertical"
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize concatenated vertical format without scene.

        Args:
            variation: Base variation name (subdirectory appended automatically).
        """
        super().__init__(variation=f"{variation}/concat_noSc_vr")
        self._layout_ = "concat"
        self._scene_mode_ = "no"
        self._align_ = "vertical"

# Separated Images # 

class SepMasks_NoSc_InputFormatModule(InputFormatModule):
    """Input format with GT and prediction as separate images, no scene.

    Presents ground truth and prediction masks as distinct, separate images in the prompt
    sequence, without including the scene image. Allows VLM to process each mask independently.

    Attributes:
        _layout_: "separate"
        _scene_mode_: "no"
        _align_: None
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize separated format without scene.

        Args:
            variation: Base variation name (subdirectory appended automatically).
        """
        super().__init__(variation=f"{variation}/sep_noSc")
        self._layout_ = "separate"
        self._scene_mode_ = "no"
        self._align_ = None

class SepMasks_Ovr_InputFormatModule(InputFormatModule):
    """Input format with alpha-blended overlay masks as separate images.

    Presents GT and prediction masks overlaid on the scene image as separate images in
    the prompt sequence. Each mask is visible in its original context.

    Attributes:
        _layout_: "separate"
        _scene_mode_: "overlay"
        _align_: None
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize separated format with overlay.

        Args:
            variation: Base variation name (subdirectory appended automatically).
        """
        super().__init__(variation=f"{variation}/sep_ovr")
        self._layout_ = "separate"
        self._scene_mode_ = "overlay"
        self._align_ = None

class SepMasks_Sc_InputFormatModule(InputFormatModule):
    """Input format with scene, GT, and prediction as separate images.

    Presents all three images (scene, ground truth, prediction) as distinct, separate images
    in the prompt sequence. Provides maximum clarity and allows independent processing.

    Attributes:
        _layout_: "separate"
        _scene_mode_: "yes"
        _align_: None
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize separated format with scene.

        Args:
            variation: Base variation name (subdirectory appended automatically).
        """
        super().__init__(variation=f"{variation}/sep_sc")
        self._layout_ = "separate"
        self._scene_mode_ = "yes"
        self._align_ = None
        
# Arrays # 

class ArrayMasks_InputFormatModule(InputFormatModule):
    """Input format with masks represented as class index arrays (no images).

    Presents GT and prediction masks as raw numerical arrays containing class indices,
    without any visual images. Suitable for VLMs that can process structured data or
    when testing pure textual understanding.

    Attributes:
        _layout_: "array"
        _scene_mode_: "no"
        _align_: None
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize array-only format.

        Args:
            variation: Base variation name (subdirectory appended automatically).
        """
        super().__init__(variation=f"{variation}/array_noImgs")
        self._layout_ = "array"
        self._scene_mode_ = "no"
        self._align_ = None

class ArrayMasks_Imgs_InputFormatModule(InputFormatModule):
    """Input format with both visual images and class index arrays.

    Presents GT and prediction masks both as visual images and as raw numerical arrays.
    Provides dual representation for VLMs that can benefit from both modalities.

    Attributes:
        _layout_: "array_with_imgs"
        _scene_mode_: "no"
        _align_: None
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize array format with visual images.

        Args:
            variation: Base variation name (subdirectory appended automatically).
        """
        super().__init__(variation=f"{variation}/array_imgs")
        self._layout_ = "array_with_imgs"
        self._scene_mode_ = "no"
        self._align_ = None

class ArrayMasks_Imgs_Ovr_InputFormatModule(InputFormatModule):
    """Input format with alpha-blended overlay images and class index arrays.

    Presents GT and prediction masks as both alpha-blended overlay images and raw
    numerical arrays. Combines contextual visual information with precise numerical data.

    Attributes:
        _layout_: "array_with_imgs"
        _scene_mode_: "overlay"
        _align_: None
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize array format with overlay images.

        Args:
            variation: Base variation name (subdirectory appended automatically).
        """
        super().__init__(variation=f"{variation}/array_imgs_ovr")
        self._layout_ = "array_with_imgs"
        self._scene_mode_ = "overlay"
        self._align_ = None

### 4. Task ###

class TaskModule(PromptModule):
    """Module for describing the task objective and instructions.

    The task module provides clear instructions about what the VLM should do with the
    input images. This typically includes explaining the segmentation comparison task,
    what to look for in the masks, and what kind of analysis to perform.

    Prompt files are located in: prompts_path / "4_task" / "{variation}.txt"
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize the task module with a specific variation.

        Args:
            variation: Name of the task variation file (e.g., "compare_masks", "analyze_errors").
        """
        super().__init__(f"4_task/{variation}")
    def __call__(self) -> str:
        """Generate the task description prompt text.

        Returns:
            The task description string loaded from the variation file.
        """
        return super().__call__()
    
### 5. Output Format ###

class OutputFormatModule(PromptModule):
    """Module for specifying the expected output format and structure.

    The output format module defines how the VLM should structure its response, including
    format specifications (JSON, XML, plain text), required fields, and formatting conventions.

    Prompt files are located in: prompts_path / "5_output_format" / "{variation}.txt"
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize the output format module with a specific variation.

        Args:
            variation: Name of the output format variation file (e.g., "json", "structured_text").
        """
        super().__init__(f"5_output_format/{variation}")
    def __call__(self) -> Prompt:
        """Generate the output format specification prompt text.

        Returns:
            The output format specification string loaded from the variation file.
        """
        return super().__call__()
    
### 6. Support Set ###

class SupportSetModule(PromptModule):
    """Module for managing few-shot learning examples (support set).

    The support set module handles the presentation of example inputs and outputs to guide
    the VLM's behavior through in-context learning. It formats multiple examples with
    numbering and appropriate spacing.

    Prompt files are located in: prompts_path / "6_support_set" / "{variation}.txt"

    Attributes:
        __sup_set_idxs__: List of dataset indices used for support set examples.
    """
    def __init__(
            self,
            variation: str, 
            sup_set_idxs: list[int]
    ) -> None:
        """Initialize the support set module with variation and example indices.

        Args:
            variation: Name of the support set variation file.
            sup_set_idxs: List of dataset indices to use as few-shot examples.
        """
        super().__init__(f"6_support_set/{variation}")
        self.__sup_set_idxs__ = sup_set_idxs
    def __call__(
            self,
            sup_set_items: list[int]
    ) -> Prompt:
        """Generate the support set prompt with formatted examples.

        Args:
            sup_set_items: List of formatted support set example items (each item is a complete
                example including images and expected output).

        Returns:
            Prompt list containing the support set header followed by numbered examples.
            Returns empty list if sup_set_items is empty.
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
    """Module for presenting the query/test image to be analyzed.

    The query module handles the presentation of the image that the VLM should analyze
    and respond to, following the patterns established by the support set examples.

    Prompt files are located in: prompts_path / "7_query" / "{variation}.txt"
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize the query module with a specific variation.

        Args:
            variation: Name of the query variation file.
        """
        super().__init__(f"7_query/{variation}")
    def __call__(
            self,
            query_item: Image.Image
    ) -> Prompt:
        """Generate the query prompt with the test image.

        Args:
            query_item: The formatted query image or image sequence to be analyzed.

        Returns:
            Prompt list containing the query header followed by the query item.
        """
        prompt = []
        prompt.append(super().__call__())
        prompt.append(query_item)
        return prompt

### 8. Evaluation ###

class EvalModule(PromptModule):
    """Module for LLM-as-a-Judge evaluation prompts.

    The evaluation module generates prompts for a separate LLM to evaluate the quality
    of VLM responses by comparing predicted answers against ground truth targets.

    Prompt files are located in: prompts_path / "8_eval" / "{variation}.txt"
    """
    def __init__(
            self,
            variation: str
    ) -> None:
        """Initialize the evaluation module with a specific variation.

        Args:
            variation: Name of the evaluation variation file (e.g., "llm_judge", "scoring").
        """
        super().__init__(f"8_eval/{variation}")
    def __call__(
            self,
            target: str,
            answer: str
    ) -> str:
        """Generate an evaluation prompt comparing target and predicted answers.

        Args:
            target: The ground truth answer string.
            answer: The VLM's predicted answer string.

        Returns:
            Formatted evaluation prompt string with target and answer filled in.
        """
        return pf.pformat(super().__call__(), target=target, answer=answer)
    
### Prompts Logic ###

class PromptBuilder():
    """Comprehensive builder for constructing VLM prompts with flexible formatting options.

    PromptBuilder is the main class for creating complete prompts for vision-language models
    in segmentation comparison tasks. It orchestrates multiple PromptModule components and
    handles image formatting, few-shot learning, and both standard and class-splitted scenarios.

    The builder supports two main operation modes:
    1. Non-splitted: Single prompt comparing all classes at once
    2. Class-splitted: Separate prompts for each class (binary white/black masks)

    Key Features:
        - Multiple layout options (concatenated, separated, array-based)
        - Flexible scene modes (with scene, overlay, no scene)
        - Alpha blending for overlay visualization
        - Few-shot learning support via support sets
        - Class-specific prompt generation
        - Evaluation prompt generation for LLM-as-a-Judge

    Attributes:
        seg_dataset: Dataset providing segmentation images and masks.
        answers_gt_path: Path to ground truth answers (JSONL format).
        sup_set_gt_path: Path to support set ground truth (JSONL format).
        jsonlio: JSON Lines I/O handler.
        layout: Image arrangement strategy (inherited from InputFormatModule).
        scene_mode: Scene display mode (inherited from InputFormatModule).
        align: Concatenation direction (inherited from InputFormatModule).
        by_model: Target VLM model name (e.g., "gemini", "gpt4").
        alpha: Alpha blending factor for overlay mode (0.0 to 1.0).
        image_size: Resize dimension for visual images.
        array_size: Resize dimension for array representations.
        class_map: Mapping from dataset classes to output classes.
        color_map: Mapping from class indices to RGB colors.
        split_by: Operation mode ("non-splitted" or "class-splitted").
        modules_dict: Dictionary of loaded PromptModule components.

    Example:
        ```python
        builder = PromptBuilder(
            seg_dataset=voc_dataset,
            prompts_path=Path("prompts/class-splitted"),
            answers_gt_path=Path("data/answers.jsonl"),
            sup_set_gt_path=Path("data/support_set.jsonl"),
            by_model="gemini",
            alpha=0.5,
            split_by="class-splitted",
            image_size=512,
            array_size=64,
            class_map=class_mapping,
            color_map=color_mapping
        )
        
        # Load modules
        builder.load_modules(...)
        
        # Build inference prompt
        prompt = builder.build_inference_prompt(query_idx=10)
        ```
    """
    def __init__(
            self,
            seg_dataset: VOC2012SegDataset,
            prompts_path: Path,
            answers_gt_path: Path,
            sup_set_gt_path: Path,
            by_model: str,
            alpha: float,
            split_by: str,
            image_size: int | tuple[int, int],
            array_size: int | tuple[int, int],
            class_map: dict,
            color_map: dict
    ) -> None:
        """Initialize the PromptBuilder with dataset and configuration parameters.

        Args:
            seg_dataset: VOC2012SegDataset instance providing images and masks.
            prompts_path: Base path to prompt template files.
            answers_gt_path: Path to JSONL file containing ground truth answers.
            sup_set_gt_path: Path to JSONL file containing support set ground truth.
            by_model: Target VLM model name (e.g., "gemini", "gpt4-vision").
            alpha: Alpha blending factor for overlay mode (0.0=scene only, 1.0=mask only).
            split_by: Operation mode, either "non-splitted" or "class-splitted".
            image_size: Target size for visual images (int for square, tuple for (H, W)).
            array_size: Target size for array representations (int for square, tuple for (H, W)).
            class_map: Dictionary mapping dataset class indices to output class indices.
            color_map: Dictionary mapping class indices to RGB color tuples.

        Note:
            - Sets PromptModule.prompts_path class variable to prompts_path / split_by
            - The layout, scene_mode, and align attributes are initialized to None and
              populated when modules are loaded via load_modules()
        """
        self.seg_dataset = seg_dataset
        self.answers_gt_path = answers_gt_path
        self.sup_set_gt_path = sup_set_gt_path
        self.jsonlio = JsonlIO()
        
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
        PromptModule.prompts_path = prompts_path / f"{self.split_by}"

    def read_sc_gt_pr(
            self,
            idx: int,
            resize_size: int | tuple[int, int]
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        """Load and format scene, ground truth, and prediction images from the dataset.

        Reads the three images for a given index, resizes them, applies color mapping to
        the segmentation masks, and converts everything to PIL Images ready for visualization.

        Args:
            idx: Dataset index of the image to load.
            resize_size: Target size for resizing (int for square, tuple for (H, W)).

        Returns:
            Tuple of (scene, ground_truth, prediction) as RGB PIL Images with color-mapped masks.

        Note:
            - Scene is resized using bilinear interpolation
            - Masks are resized using nearest-neighbor interpolation (preserves class boundaries)
            - Color map from self.color_map is applied to both GT and prediction masks
            - All three images are guaranteed to have the same size
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
        """Serialize the current PromptBuilder state to a dictionary.

        Creates a JSON-serializable dictionary representation of all instance attributes,
        including nested objects like PromptModule instances. Useful for logging, debugging,
        and reproducing prompt configurations.

        Returns:
            Dictionary containing all instance attributes with nested objects converted
            to their dictionary representations where possible.

        Note:
            - Objects with __to_dict__ method are serialized using that method
            - Other objects are converted to their string representation via __repr__
            - The result can be saved to JSON for configuration reproducibility
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
        """Build a formatted prompt for a single image with optional ground truth answer.

        Creates a complete single-image prompt including formatted images (scene, GT, prediction)
        according to the configured layout and scene mode. Optionally appends the ground truth
        answer for use in few-shot examples.

        This method is used both for query images (without answer) and support set examples
        (with answer).

        Args:
            img_idx: Dataset index of the image to build prompt for.
            with_answer_gt: If True, appends the ground truth answer to the prompt.
                Used for few-shot examples.

        Returns:
            Flattened list containing prompt text and image objects arranged according to
            the configured layout:
            - concat: Single concatenated image
            - separate: Multiple individual images with labels
            - array: Text labels followed by class index arrays
            - array_with_imgs: Both visual images and arrays

        Note:
            - Uses self.image_size for visual images
            - Uses self.array_size for array representations
            - Ground truth answers are loaded from answers_gt_path (non-splitted) or
              sup_set_gt_path (class-splitted)
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
        if with_answer_gt:
            if self.split_by == 'non-splitted':
                answer_gt = get_answer_objects(self.answers_gt_path, img_idx, self.jsonlio, format_to_dict=True)[img_idx]
            elif self.split_by == 'class-splitted':
                answer_gt = get_answer_objects(self.sup_set_gt_path, img_idx, self.jsonlio, format_to_dict=True)[img_idx]
            img_prompts.append(answer_gt) # add target answer if specified
        return flatten_list(img_prompts)

    def inherit_settings_from_modules(self) -> None:
        """Inherit configuration attributes from loaded modules.

        Scans all loaded modules for attributes marked with leading and trailing underscores
        (e.g., _layout_, _scene_mode_) and copies them to the PromptBuilder instance as
        plain attributes (e.g., self.layout, self.scene_mode).

        This allows InputFormatModule subclasses to communicate their configuration to the
        builder without explicit parameter passing.

        Note:
            Only attributes that start AND end with underscore are inherited.
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
        """Load and validate all prompt modules in their canonical order.

        Registers the eight required prompt modules and inherits configuration attributes
        from them (particularly from input_format_module). Validates that module types
        match their expected base classes and that class-splitted mode uses appropriate
        color map modules.

        Args:
            context_module: Module providing task context.
            color_map_module: Module specifying class-to-color mapping.
            input_format_module: Module defining image presentation format.
            task_module: Module describing the task objective.
            output_format_module: Module specifying output structure.
            support_set_module: Module managing few-shot examples.
            query_module: Module presenting the query image.
            eval_module: Module for LLM-as-a-Judge evaluation.

        Raises:
            AssertionError: If any module is not an instance of its expected base class.
            AttributeError: If class-splitted mode is used without ClassSplitted_ColorMapModule.

        Note:
            Modules must be provided in the order shown above, as they define the canonical
            prompt structure.
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
        """Build a complete inference prompt for VLM analysis (non-splitted mode).

        Constructs a full prompt by combining all module components in order, with support
        set examples (few-shot) and the query image. This is the main method for generating
        prompts in non-splitted mode where all classes are evaluated together.

        Args:
            query_idx: Dataset index of the query image to analyze.

        Returns:
            Flattened list containing the complete prompt with all components:
            1. Context text
            2. Color map (text and/or image)
            3. Input format description
            4. Task description
            5. Output format specification
            6. Support set examples (if any)
            7. Query image

        Note:
            - Support set indices are read from self.sup_set_idxs
            - For class-splitted mode, use build_class_splitted_inference_prompts() instead
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
        """Create a deep copy of this builder with binary color mapping for one class.

        Clones the current PromptBuilder and modifies its color map to create a binary
        visualization where:
        - The specified positive class is rendered as white (255, 255, 255)
        - All other classes are rendered as black (0, 0, 0)

        This is used for class-splitted mode where each class is evaluated independently.

        Args:
            pos_class: Class index to highlight as the positive class.

        Returns:
            A new PromptBuilder instance identical to this one except for the binary
            color map focused on pos_class.

        Note:
            Uses deepcopy to ensure complete independence from the original instance.
        """
        class_specific_promptBuilder = deepcopy(self)
        class_specific_promptBuilder.color_map = {c: [255, 255, 255] if c == pos_class else [0, 0, 0] for c in range(self.seg_dataset.get_num_classes(with_unlabelled=False))}
        return class_specific_promptBuilder
    
    def build_class_splitted_support_set_items(self) -> Prompt:
        """Generate all few-shot examples for class-splitted mode.

        Creates the complete set of support examples for class-splitted prompts, where
        each example focuses on a specific class with binary color mapping.

        Returns:
            List of formatted support set items, each created with class-specific color mapping.

        Note:
            Calls build_class_specific_support_set_item() for each index in self.sup_set_idxs.
        """
        sup_set_items = []
        for idx in self.sup_set_idxs:
            sup_set_items.append(self.build_class_specific_support_set_item(idx))
        return sup_set_items
    
    def build_class_specific_support_set_item(
            self,
            img_idx: int
    ) -> Prompt:
        """Build a single few-shot example with class-specific binary color mapping.

        Creates a support set example where masks are rendered with binary colors focused
        on a single class. The class to highlight is determined by a hardcoded mapping
        (img_idx_to_class_) for demonstration purposes.

        Args:
            img_idx: Dataset index of the support set image.

        Returns:
            Formatted prompt item with binary-colored masks and ground truth answer.

        Note:
            - Uses hardcoded img_idx_to_class_ mapping: {2: 20, 16: 1, 18: 13}
            - This mapping specifies which class each support example demonstrates
            - Future versions could make this mapping configurable
            - The ground truth answer is included in the output
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
        """Build a complete inference prompt for a single class in class-splitted mode.

        Creates a full prompt focusing on one specific class, using binary color mapping
        (target class in white, others in black). The prompt includes all standard components
        with the class name substituted in placeholders.

        Args:
            query_idx: Dataset index of the query image to analyze.
            pos_class: Class index to treat as the positive class for this prompt.

        Returns:
            Flattened list containing the complete class-specific prompt with:
            1. Context text
            2. Color map description
            3. Input format description
            4. Task description
            5. Output format specification
            6. Support set examples (class-splitted)
            7. Query image (with binary color mapping)
            
            All text containing '[pos_class]' placeholders is formatted with the class name.

        Note:
            Uses a class-specific PromptBuilder clone with binary color mapping for the query.
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
        """Build an LLM-as-a-Judge evaluation prompt comparing predicted and ground truth answers.

        Creates a prompt for a separate LLM to evaluate the quality of a VLM's response
        by comparing it against the ground truth answer.

        Args:
            query_idx: Dataset index of the query image (to retrieve ground truth).
            answer_pr: The predicted answer string generated by the VLM.

        Returns:
            List containing the formatted evaluation prompt with ground truth and
            prediction filled in.

        Note:
            - Ground truth is loaded from self.answers_gt_path
            - Uses the 'content' field from the answer object
        """
        answer_gt = get_answer_objects(self.answers_gt_path, query_idx, self.jsonlio, format_to_dict=False)['content']
        prompt = [self.modules_dict["eval"](answer_gt, answer_pr)]
        return prompt
    
    def build_class_splitted_inference_prompts(
            self,
            query_idx: int
    ) -> dict[int, Prompt]:
        """Build multiple inference prompts, one for each significant class in the image.

        Creates a dictionary of prompts where each prompt focuses on a single class using
        binary color mapping. Only classes that appear in either the ground truth or
        prediction masks are included (excluding background unless it's the only class).

        Args:
            query_idx: Dataset index of the query image to analyze.

        Returns:
            Dictionary mapping class indices to their corresponding class-specific prompts.
            For example: {1: prompt_for_class_1, 5: prompt_for_class_5, ...}

        Note:
            - Automatically extracts significant classes from GT and prediction masks
            - Background (class 0) is excluded unless it's the only class present
            - Each prompt uses build_class_specific_inference_prompt() internally
            - May return empty dict if only background class is present
        """
        # NOTE: the masks must not only have BACKGROUND class, otherwise the method crashes.
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
        """Build multiple evaluation prompts, one for each class-specific prediction.

        Creates a dictionary of LLM-as-a-Judge evaluation prompts where each prompt
        compares a class-specific predicted answer against its ground truth. Class names
        are substituted into placeholder text.

        Args:
            query_idx: Dataset index of the query image (to retrieve ground truths).
            pos_class_2_answer_pr: Dictionary mapping class indices to their predicted
                answer strings from the VLM.

        Returns:
            Dictionary mapping class indices to their formatted evaluation prompts.
            For example: {1: eval_prompt_for_class_1, 5: eval_prompt_for_class_5, ...}

        Note:
            - Class names are retrieved from self.seg_dataset.get_classes()
            - Substitutes class name into '[pos_class]' placeholders in the eval prompt
        """
        pos_class_2_eval_prompt = {}
        significant_classes = pos_class_2_answer_pr.keys()
        class_splitted_answer_pr = pos_class_2_answer_pr.values()
        for pos_class, answer_pr in zip(significant_classes, class_splitted_answer_pr):
            pos_class = int(pos_class)
            pos_class_2_eval_prompt[pos_class] = [pf.pformat(self.build_eval_prompt(query_idx, answer_pr)[0], pos_class=self.seg_dataset.get_classes(with_unlabelled=False)[pos_class])]
        return pos_class_2_eval_prompt

    def build_cs_eval_prompt(
            self,
            query_idx: int,
            answer_pr: str,
            pos_class: int,
    ) -> str:
        """Build a single class-specific evaluation prompt.

        Creates an LLM-as-a-Judge evaluation prompt for one specific class, with the
        class name substituted into placeholder text.

        Args:
            query_idx: Dataset index of the query image (to retrieve ground truth).
            answer_pr: The predicted answer string from the VLM for this class.
            pos_class: Class index being evaluated.

        Returns:
            Formatted evaluation prompt string with class name substituted.

        Note:
            This is a simpler alternative to build_class_splitted_eval_prompt() for
            evaluating a single class at a time.
        """
        return pf.pformat(self.build_eval_prompt(query_idx, answer_pr)[0], pos_class=self.seg_dataset.get_classes(with_unlabelled=False)[pos_class])
    
# 'with_unlabelled' in both Prompters should be given as argument to __init__ and propagated to all other functions.

def get_significant_classes(
        input_tensor: torch.Tensor,
        batched: bool = False,
) -> list[int] | list[list[int]]:
    """Extract unique class indices present in segmentation tensor(s).

    Identifies all unique class labels appearing in a segmentation mask tensor,
    which represents the "significant" classes that are actually present in the image.

    Args:
        input_tensor: Segmentation mask tensor containing class indices. Can be:
            - Single mask: shape (H, W) or (1, H, W)
            - Batch: shape (B, H, W) when batched=True
        batched: If True, treats first dimension as batch and returns list of lists.
            If False, treats as single mask and returns single list.

    Returns:
        If batched=False: List of unique class indices in the mask (e.g., [0, 1, 5, 12])
        If batched=True: List of lists, one per batch element (e.g., [[0, 1], [0, 5, 12]])

    Example:
        ```python
        # Single mask
        mask = torch.tensor([[0, 1], [1, 2]])
        classes = get_significant_classes(mask)  # [0, 1, 2]
        
        # Batch of masks
        masks = torch.stack([mask1, mask2])
        classes = get_significant_classes(masks, batched=True)  # [[0, 1], [0, 2, 5]]
        ```
    """
    if batched:
        return [img_t.unique().tolist() for img_t in input_tensor]
    else:
        return input_tensor.unique().tolist()

class FastPromptBuilder:
    """Optimized batch-processing prompt builder for class-splitted inference.

    FastPromptBuilder is an alternative to PromptBuilder optimized for high-throughput
    batch processing, particularly in class-splitted scenarios. It precomputes the base
    prompt (context, color map, task, etc.) once and efficiently generates class-specific
    query prompts by splicing in query images.

    Key Differences from PromptBuilder:
        - Uses template-based prompts from JSON files instead of modular PromptModule system
        - Supports batch processing of multiple images simultaneously
        - Precomputes support set to avoid redundant processing
        - Optimized tensor operations for color mapping and blending
        - Designed specifically for class-splitted mode

    Attributes:
        seg_dataset: Dataset for query images.
        prompts_file_path: Path to JSON file containing prompt templates.
        jsonlio: JSON Lines I/O handler.
        seed: Random seed for reproducibility.
        prompt_blueprint: Ordered dict defining prompt structure (module -> variation mapping).
        by_model: Target VLM model name.
        alpha: Alpha blending factor for overlay mode (None to disable).
        class_map: Class index mapping.
        color_map: RGB color mapping for classes.
        image_size: Target size for visual images.
        str_formats: Dictionary of string substitutions (e.g., {'[user]': 'Alice'}).
        sup_set_img_idxs: Dataset indices for support set examples.
        sup_set_gt_path: Path to support set ground truth answers.
        sup_set_seg_dataset: Dataset for support set images (may differ from query dataset).
        base_prompt: Precomputed prompt components (context through support set).
        head_prompt: Template for query section (to be filled per-query).

    Example:
        ```python
        builder = FastPromptBuilder(
            seg_dataset=voc_dataset,
            prompts_file_path=Path("prompts/templates.json"),
            prompt_blueprint=OrderedDict([
                ("context", "base"),
                ("color_map", "binary"),
                # ...
            ]),
            by_model="gemini",
            alpha=0.5,
            class_map=class_mapping,
            color_map=color_mapping,
            image_size=512,
            sup_set_seg_dataset=voc_support,
            sup_set_gt_path=Path("support_answers.jsonl"),
            sup_set_img_idxs=[0, 1, 2],
            str_formats={},
            seed=42
        )
        
        # Build prompts for batch of images
        prompts = builder.build_cs_inference_prompts_from_disk([10, 20, 30])
        ```
    """
    def __init__(
            self,
            seg_dataset: VOC2012SegDataset,
            prompts_file_path: Path,
            prompt_blueprint: OrderedDict[str, str],
            by_model: str,
            alpha: float,
            class_map: dict[int, int],
            color_map: dict[int, tuple[int, int, int]],
            image_size: int | tuple[int, int],
            sup_set_seg_dataset: VOC2012SegDataset,
            sup_set_gt_path: Path,
            sup_set_img_idxs: list[int],
            str_formats: dict[str, str],
            seed: int,
    ) -> None:
        """Initialize FastPromptBuilder with configuration and precompute base prompt.

        Args:
            seg_dataset: VOC2012SegDataset for query images.
            prompts_file_path: Path to JSON file with prompt templates.
            prompt_blueprint: OrderedDict mapping module names to variation names.
            by_model: Target VLM model identifier.
            alpha: Alpha blending factor (0.0-1.0) for overlay mode, or None to disable.
            class_map: Dictionary mapping class indices (dataset -> output).
            color_map: Dictionary mapping class indices to RGB tuples.
            image_size: Target resize dimension (int for square, tuple for (H, W)).
            sup_set_seg_dataset: VOC2012SegDataset for support set examples.
            sup_set_gt_path: Path to JSONL file with support set ground truth answers.
            sup_set_img_idxs: List of dataset indices for support set examples.
            str_formats: Dictionary of placeholder substitutions (e.g., {'[var]': 'value'}).
            seed: Random seed for reproducibility.

        Note:
            - Automatically precomputes base_prompt and head_prompt during initialization
            - base_prompt contains all components up to (but not including) the query
            - head_prompt is a template for the query section
        """
        self.seg_dataset = seg_dataset
        self.prompts_file_path = prompts_file_path
        self.jsonlio = JsonlIO()
        self.seed = seed
        self.prompt_blueprint = prompt_blueprint
        self.by_model = by_model
        self.alpha = alpha
        self.class_map = class_map
        self.color_map = color_map
        self.image_size = image_size
        self.str_formats = str_formats
        self.sup_set_img_idxs = sup_set_img_idxs
        self.sup_set_gt_path = sup_set_gt_path
        self.sup_set_seg_dataset = sup_set_seg_dataset
        
        base_head_prompt = self.build_cs_base_prompt(self.sup_set_img_idxs, alpha)
        self.base_prompt = base_head_prompt[:-1]
        self.head_prompt = base_head_prompt[-1]

    def build_base_prompt_(
            self,
            sup_set_imgs: list[tuple[Image.Image, Image.Image]],
            sup_set_answers: list[str]
    ) -> Prompt:
        """Build the base prompt with support set examples from templates.

        Constructs the prompt foundation by loading templates from the JSON file,
        expanding the support set module for each example, and applying string substitutions.

        Args:
            sup_set_imgs: List of tuples, each containing (GT image, prediction image).
            sup_set_answers: List of ground truth answer strings, one per support example.

        Returns:
            Flattened list containing the base prompt with all components through
            the support set.

        Note:
            - Loads templates based on self.prompt_blueprint
            - Support set examples are numbered and populated with images and answers
            - String substitutions from self.str_formats are applied at the end
        """
        
        def populate_sup_set(
            sup_set_module: str,
            sup_set_imgs: list[tuple[Image.Image, Image.Image]],
            sup_set_answers: list[str]
        ) -> Prompt:
            """Expand support set module template with all examples.

            Args:
                sup_set_module: Template string for a single support set item.
                sup_set_imgs: List of (GT, prediction) image tuples.
                sup_set_answers: List of answer strings.

            Returns:
                Expanded support set with all examples populated.
            """
            expanded_sup_set_module = sup_set_module*len(sup_set_imgs) # replicate the placeholder for each support set example.
            expanded_sup_set_module = map_list_placeholders(expanded_sup_set_module, placeholder="[sup_set_count]", objects_list=[str(n+1) for n in range(len(sup_set_imgs))])
            expanded_sup_set_module = map_list_placeholders(expanded_sup_set_module, placeholder="[answer_gt]", objects_list=sup_set_answers)
            expanded_sup_set_module = map_list_placeholders(expanded_sup_set_module, placeholder="[img]", objects_list=flatten_list(sup_set_imgs))
            return expanded_sup_set_module
        
        prompt_corpus = read_json(self.prompts_file_path)
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
        """Build the class-splitted base prompt with support set examples.

        Loads support set images, applies class-specific binary color mapping, optionally
        blends with scene images, and constructs the base prompt using these processed images.

        This method uses batch processing for efficiency, applying color mapping to all
        support images simultaneously.

        Args:
            sup_set_img_idxs: Dataset indices for support set examples.
            alpha: Alpha blending factor (0.0-1.0) for overlay mode, or None to disable.

        Returns:
            Prompt list containing all components through the support set, with the
            final element being the query template.

        Note:
            - Uses hardcoded img_idx_to_class_ mapping: {2: 20, 16: 1, 18: 13}
            - Applies same color map to all support examples (assumes disjoint classes)
            - Processes all images in batch for efficiency using tensor operations
            - Resizes images to self.image_size
            - Returns base_prompt[:-1] and head_prompt[-1] separately
        """
        img_idx_to_class_ = {
            2: 20,
            16: 1,
            18: 13,
        }

        # WARNING: this implementation sets the the same color map for all the pos. classes (the values of the dict above).
        # This way, the color mapping can be computed in parallel (faster) and the code is simpler.
        # This method works only if the sup sets have disjointed classes. Ideally, each support example would have its own color map.
        color_map = {pos_c: (255, 255, 255) for pos_c in img_idx_to_class_.values()}
        
        scs, gts, prs = self.sup_set_seg_dataset[sup_set_img_idxs]

        scs = torch.stack(scs)
        gts = torch.stack(gts)
        prs = torch.stack(prs)

        scs = TF.resize(scs, self.image_size, TF.InterpolationMode.BILINEAR)
        gts = TF.resize(gts, self.image_size, TF.InterpolationMode.NEAREST)
        prs = TF.resize(prs, self.image_size, TF.InterpolationMode.NEAREST)

        gts = apply_colormap(gts, color_map)
        prs = apply_colormap(prs, color_map)
        
        if alpha:
            alpha_tensor = torch.full(gts.size(), alpha, device=gts.device)
            gts = blend_tensors(scs, gts, alpha_tensor)
            prs = blend_tensors(scs, prs, alpha_tensor)

        gts_img = [to_pil_image(col_gt) for col_gt in gts]
        prs_img = [to_pil_image(col_pr) for col_pr in prs]

        sup_set_imgs = list(zip(gts_img, prs_img))
        sup_set_answers = [get_answer_objects(self.sup_set_gt_path, sup_set_img_idx, jsonlio=self.jsonlio, return_state=False, format_to_dict=False)['content'] for sup_set_img_idx in self.sup_set_img_idxs]

        base_prompt = self.build_base_prompt_(sup_set_imgs, sup_set_answers)

        return base_prompt

    def extract_significant_classes(
            self,
            query_gt: torch.Tensor,
            query_pr: torch.Tensor,
            sign_classes_filter: Optional[Callable[[list[int]], list[int]]] = None
    ) -> list[int]:
        """Extract and filter significant classes from GT and prediction masks.

        Identifies all unique classes present in either the ground truth or prediction
        mask, optionally applies a filter function, and removes background class unless
        it's the only class present.

        Args:
            query_gt: Ground truth segmentation mask tensor (H, W).
            query_pr: Prediction segmentation mask tensor (H, W).
            sign_classes_filter: Optional callable to filter prediction classes before
                merging with GT classes. For example, could filter out rare classes.

        Returns:
            Sorted list of significant class indices, excluding background (class 0)
            unless it's the only class present.

        Note:
            - Background (class 0) is special-cased to handle images where cropping
              removes all meaningful classes
            - Filter is applied only to prediction classes, not GT classes
        """
        gt_sign_classes = get_significant_classes(query_gt)
        pr_sign_classes = get_significant_classes(query_pr)

        if sign_classes_filter:
            pr_sign_classes = sign_classes_filter(pr_sign_classes)
        
        sign_classes = list(set(gt_sign_classes + pr_sign_classes))
        # sign_classes = gt_sign_classes

        # Remove the BACKGROUND class only if it is not the only one.
        # Cropping can leave out all meaningful classes: in this case, the BACKGROUND class is considered positive
        # and both masks are completely white.
        if 0 in sign_classes and sign_classes != [0]:
            sign_classes.remove(0)

        return sorted(sign_classes)
    
    def get_cs_splitted_masks(
            self,
            full_mask: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Split a multi-class mask into multiple binary masks, one per class.

        Creates a dictionary where each key is a class index and each value is a
        binary mask (boolean tensor) indicating where that class appears.

        Args:
            full_mask: Multi-class segmentation mask tensor (H, W) with class indices.

        Returns:
            Dictionary mapping class indices to boolean masks. For example:
            {1: tensor([[True, False], ...]), 5: tensor([[False, True], ...]), ...}

        Note:
            Only includes classes that actually appear in the mask (significant classes).
        """
        sign_classes = self.extract_significant_classes(full_mask, full_mask, None)
        return {pos_c: full_mask == pos_c for pos_c in sign_classes}
    
    def expand_head_to_cs(
            self,
            query_gt: torch.Tensor,
            query_pr: torch.Tensor,
            query_sc: Optional[torch.Tensor],
            alpha: Optional[float],
            sign_classes_filter: Optional[Callable[[list[int]], list[int]]] = None
    ) -> dict[int, list[Image.Image]]:
        """Generate class-splitted image pairs (GT, prediction) for all significant classes.

        Takes multi-class GT and prediction masks and creates binary versions for each
        significant class, with optional alpha blending against the scene image.

        Args:
            query_gt: Ground truth mask tensor (H, W) with class indices.
            query_pr: Prediction mask tensor (H, W) with class indices.
            query_sc: Optional scene image tensor (C, H, W) for overlay blending.
            alpha: Alpha blending factor (0.0-1.0) for overlay, or None to disable.
            sign_classes_filter: Optional filter for prediction classes.

        Returns:
            Dictionary mapping class indices to [GT_image, prediction_image] pairs.
            For example: {1: [gt_img_for_class_1, pr_img_for_class_1], ...}

        Note:
            - Uses binary color mapping (target class = white, others = black)
            - Processes all classes in batch for efficiency
            - Returns PIL Images, not tensors
        """
        
        sign_classes = self.extract_significant_classes(query_gt, query_pr, sign_classes_filter)

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
        """Fill query module template with GT and prediction images.

        Replaces '[img]' placeholders in the query template with the provided images.

        Args:
            query_module: Query template string(s) with '[img]' placeholders.
            query_gt: Ground truth image to insert.
            query_pr: Prediction image to insert.

        Returns:
            List with template text and images interleaved (e.g., ['Text ', img1, ' more text ', img2]).
        """
        return map_list_placeholders(query_module, placeholder="[img]", objects_list=[query_gt, query_pr])
    
    def build_cs_inference_prompts(
            self,
            gts_tensor: torch.Tensor,
            prs_tensor: torch.Tensor,
            scs_tensor: torch.Tensor,
            sign_classes_filter: Optional[Callable[[list[int]], list[int]]] = None,
    ) -> list[dict[int, Prompt]]:
        """Build class-splitted inference prompts for a batch of images.

        Processes multiple query images in batch, generating class-specific prompts for
        each significant class in each image. Uses optimized tensor operations for
        color mapping and blending.

        Args:
            gts_tensor: Batch of GT masks, shape (B, H, W).
            prs_tensor: Batch of prediction masks, shape (B, H, W).
            scs_tensor: Batch of scene images, shape (B, C, H, W).
            sign_classes_filter: Optional filter for prediction classes.

        Returns:
            List of dictionaries, one per batch element. Each dictionary maps class
            indices to their complete prompts. For example:
            [
                {1: prompt_img0_class1, 5: prompt_img0_class5},
                {1: prompt_img1_class1, 2: prompt_img1_class2},
                ...
            ]

        Note:
            - Resizes all images to self.image_size
            - Applies class name substitution to '[pos_class]' placeholders
            - Uses self.alpha for overlay blending if set
            - Processes entire batch efficiently using vectorized operations
        """
        
        gts_tensor = TF.resize(gts_tensor, self.image_size, TF.InterpolationMode.NEAREST)
        prs_tensor = TF.resize(prs_tensor, self.image_size, TF.InterpolationMode.NEAREST)
        scs_tensor = TF.resize(scs_tensor, self.image_size, TF.InterpolationMode.BILINEAR)

        splitted_elements_dict = [self.expand_head_to_cs(gt, pr, sc, self.alpha, sign_classes_filter) for gt, pr, sc in zip(gts_tensor, prs_tensor, scs_tensor)]

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
    ) -> list[dict[int, Prompt]]:
        """Build class-splitted inference prompts by loading images from dataset.

        Convenience method that loads images from the dataset and calls
        build_cs_inference_prompts() for batch processing.

        Args:
            query_idxs: List of dataset indices for query images.

        Returns:
            List of dictionaries, one per query image, mapping class indices to prompts.
            Same format as build_cs_inference_prompts().

        Note:
            - Loads images using self.seg_dataset[query_idxs]
            - Automatically stacks images into batched tensors
        """
        
        scs, gts, prs = self.seg_dataset[query_idxs]

        scs = torch.stack(scs)
        gts = torch.stack(gts)
        prs = torch.stack(prs)
        
        cs_prompts_list =  self.build_cs_inference_prompts(gts, prs, scs)

        return cs_prompts_list
    
    def get_state(self) -> dict:
        """Serialize the current FastPromptBuilder state to a dictionary.

        Creates a JSON-serializable dictionary representation of all instance attributes.
        Useful for logging, debugging, and reproducing prompt configurations.

        Returns:
            Dictionary containing all instance attributes with nested objects converted
            to their dictionary representations where possible.

        Note:
            - Objects with __to_dict__ method are serialized using that method
            - Other objects are converted to their string representation via __repr__
            - The result can be saved to JSON for configuration reproducibility
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
        img_idxs: tuple[int],
        local_annot_imgs_path: Path,
) -> None:
    """Save formatted visualization images for specified dataset indices.

    Generates and saves formatted images (scene, GT, prediction) according to the
    PromptBuilder's configuration (layout, scene mode, alignment). Useful for creating
    annotation materials or visual documentation.

    Args:
        promptBuilder: Configured PromptBuilder instance defining formatting options.
        img_idxs: Tuple of dataset indices to process and save.
        local_annot_imgs_path: Directory path where images will be saved.

    Note:
        - Images are saved as PNG files named "annot_image_{idx}.png"
        - Uses promptBuilder's image_size, layout, scene_mode, align, and alpha settings
        - Creates the output directory if it doesn't exist
    """
    for img_idx in img_idxs:
        sc, gt, pr = promptBuilder.read_sc_gt_pr(img_idx, promptBuilder.image_size)
        formatted_image = _format_images(sc, gt, pr, img_idx, promptBuilder.layout, promptBuilder.scene_mode, promptBuilder.align, promptBuilder.alpha)[0]
        formatted_image.save(local_annot_imgs_path / f"annot_image_{img_idx}.png")

def make_synthetic_diff_text(
        templates_filepath: Path,
        gts: torch.Tensor,
        prs: torch.Tensor,
        pos_class_name: str
) -> str:
    """Generate synthetic difference text based on confusion matrix patterns.

    Analyzes the confusion between ground truth and prediction masks to determine which
    of the four cases (TP, TN, FP, FN) are present, then selects an appropriate template
    describing the differences.

    Confusion Matrix Cases:
        - TP (True Positive): Prediction = 1, GT = 1 (correctly predicted class)
        - TN (True Negative): Prediction = 0, GT = 0 (correctly predicted absence)
        - FP (False Positive): Prediction = 1, GT = 0 (incorrectly predicted presence)
        - FN (False Negative): Prediction = 0, GT = 1 (missed class)

    Args:
        templates_filepath: Path to JSON file containing templates for each case combination.
            Templates are keyed by case strings like "TP+FP" or "TP+TN+FP+FN".
        gts: Ground truth binary mask tensor (boolean or 0/1 values).
        prs: Prediction binary mask tensor (boolean or 0/1 values).
        pos_class_name: Name of the positive class to insert into template placeholders.

    Returns:
        Formatted text description of the differences, with {POS_CLASS} placeholders
        replaced by pos_class_name.

    Example:
        ```python
        templates = {
            "TP+FP": "The {POS_CLASS} is mostly correct but includes extra regions.",
            "TP+FN": "The {POS_CLASS} is present but incomplete.",
            ...
        }
        text = make_synthetic_diff_text(
            Path("templates.json"),
            gt_mask,
            pr_mask,
            "aeroplane"
        )
        # Returns: "The aeroplane is mostly correct but includes extra regions."
        ```

    Note:
        - Masks are automatically converted to boolean for logical operations
        - Uses torch.any() to check if each case type exists anywhere in the masks
        - Template file must contain entries for all possible case combinations
    """
    # Ensure tensors are boolean type for logical operations
    gts_bool = gts.bool()
    prs_bool = prs.bool()
    
    TP = torch.any(prs_bool * gts_bool) # True Positives (TP): Prediction is 1, and Target is 1
    TN = torch.any((~prs_bool) & (~gts_bool)) # True Negatives (TN): Prediction is 0, and Target is 0
    FP = torch.any(prs_bool * (gts_bool == False)) # False Positives (FP): Prediction is 1, and Target is 0
    FN = torch.any((prs_bool == False) * gts_bool) # False Negatives (FN): Prediction is 0, and Target is 1
    
    # populate computed cases string
    case = []
    if TP.item():
        case.append("TP")
    if TN.item():
        case.append("TN")
    if FP.item():
        case.append("FP")
    if FN.item():
        case.append("FN")
    case_str = "+".join(case)
    
    # populate template cases string
    templates_d: dict[str, str] = read_json(templates_filepath)
    template = templates_d[case_str].format(POS_CLASS=pos_class_name)

    return template

def main() -> None:
    """Main entry point for the module.
    
    Currently not implemented. This module is designed to be imported and used
    by other scripts rather than executed directly.
    """
    ...

if __name__ == '__main__':
    main()
