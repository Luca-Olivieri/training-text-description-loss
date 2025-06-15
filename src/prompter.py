from PIL import Image, ImageFont, ImageDraw, ImageOps
from pathlib import Path
import json
import pformat as pf

from config import *
from color_map import *
from utils import *
from data import *
from path import *
from copy import deepcopy

from torchvision.transforms.functional import to_pil_image

### Methods ###

def _concat_images_fn(images, titles, scene_mode, align):
    """
    Concatenates the images according to the values of the parameters 'scene_mode' and 'align' in various combinations.
    Some of them mark the images with titles given as parameters.
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

def _format_images(sc, gt, pr, idx, layout, scene_mode, align, alpha):
    """
    Returns 'sc', 'gt', 'pr' formatted for the prompt.
    Always return a list [sc, gt, pr].
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

### Prompt Modules ###

class PromptModule():
    """
    This is the base class for the prompt textual modules that, together, form the full prompts.
    It has to be implemented by the various modules according to their purpose and position in the full prompt.
    The attributes of the sub-classes can propagated to the class 'PromptBuilder', which uses them through out the building of the prompt.
    """
    prompts_path = None # variable shared among all sub-classes, it has to be initialized externally when building the prompt.
    def __init__(self, variation):
        """
        N.B. The PromperBuilder class inherits all and only the private attributes that are enclosed by '_' (e.g., self._value_ = ...)
        """
        self.full_path = Path(self.prompts_path / f"{variation}.txt") # full path complete of the prompt variation.
    def import_variation(self, variation_path):
        """
        This method returns the .txt file found at the path 'prompts_path/content_path'.
        This is not to be re-implemented.
        """
        return read_txt(self.full_path)
    def __call__(self):
        """
        This method should return the textual prompt module complete with items to display.
        It should not read images or re-format them, it should receive the images ready to be inserted in the prompt.
        Each sub-class re-implements this method accordingly.
        """
        return self.import_variation(self.full_path)
    def __to_dict__(self):
        """
        This method should return the module as well as the attributes of the class in a dict.
        It should not be re-implemented.
        """
        return {"class": self.__class__.__name__} | vars(self)
    
### 1. Context ###

class ContextModule(PromptModule):
    def __init__(self, variation):
        super().__init__(f"1_context/{variation}")
    def __call__(self):
        return super().__call__()

### 2. Color Map ###

class ColorMapModule(PromptModule):
    def __init__(self, variation):
        super().__init__(f"2_color_map/{variation}")
    def __call__(self, color_map_item):
        return super().__call__(), color_map_item
    
class Image_ColorMapModule(ColorMapModule):
    def __call__(self):
        return super().__call__(get_color_map_as("img"))

class RGB_ColorMapModule(ColorMapModule):
    def __call__(self):
        return super().__call__(get_color_map_as("rgb"))

class Names_ColorMapModule(ColorMapModule):
    def __call__(self):
        return super().__call__(get_color_map_as("names"))

class Patches_ColorMapModule(ColorMapModule):
    def __call__(self):
        return super().__call__(get_color_map_as("patches"))
    
class ClassSplitted_ColorMapModule(ColorMapModule):
    def __call__(self):
        text_prompt, _ = super().__call__(None) # no color map item needed
        return text_prompt

### 3. Input Format ###

class InputFormatModule(PromptModule):
    def __init__(self, variation):
        super().__init__(f"3_input_format/{variation}")
    def __call__(self):
        return super().__call__()

# Concatenated Images # 

# TODO: uniform the variation names to the class names (e.g. "concat_sc_hz" --> "ConcatMasks_Sc_Hz")
class ConcatMasks_Sc_Hz_InputFormatModule(InputFormatModule):
    def __init__(self, variation):
        super().__init__(variation=f"{variation}/concat_sc_hz")
        self._layout_ = "concat"
        self._scene_mode_ = "yes"
        self._align_ = "horizontal"

class ConcatMasks_Sc_Vr_InputFormatModule(InputFormatModule):
    def __init__(self, variation):
        super().__init__(variation=f"{variation}/concat_sc_vr")
        self._layout_ = "concat"
        self._scene_mode_ = "yes"
        self._align_ = "vertical"

class ConcatMasks_Ovr_Hz_InputFormatModule(InputFormatModule):
    def __init__(self, variation):
        super().__init__(variation=f"{variation}/concat_ovr_hz")
        self._layout_ = "concat"
        self._scene_mode_ = "overlay"
        self._align_ = "horizontal"

class ConcatMasks_Ovr_Vr_InputFormatModule(InputFormatModule):
    def __init__(self, variation):
        super().__init__(variation=f"{variation}/concat_ovr_vr")
        self._layout_ = "concat"
        self._scene_mode_ = "overlay"
        self._align_ = "vertical"

class ConcatMasks_NoSc_Hz_InputFormatModule(InputFormatModule):
    def __init__(self, variation):
        super().__init__(variation=f"{variation}/concat_noSc_hz")
        self._layout_ = "concat"
        self._scene_mode_ = "no"
        self._align_ = "horizontal"

class ConcatMasks_NoSc_Vr_InputFormatModule(InputFormatModule):
    def __init__(self, variation):
        super().__init__(variation=f"{variation}/concat_noSc_vr")
        self._layout_ = "concat"
        self._scene_mode_ = "no"
        self._align_ = "vertical"

# Separated Images # 

class SepMasks_NoSc_InputFormatModule(InputFormatModule):
    def __init__(self, variation):
        super().__init__(variation=f"{variation}/sep_noSc")
        self._layout_ = "separate"
        self._scene_mode_ = "no"
        self._align_ = None

class SepMasks_Ovr_InputFormatModule(InputFormatModule):
    def __init__(self, variation):
        super().__init__(variation=f"{variation}/sep_ovr")
        self._layout_ = "separate"
        self._scene_mode_ = "overlay"
        self._align_ = None

class SepMasks_Sc_InputFormatModule(InputFormatModule):
    def __init__(self, variation):
        super().__init__(variation=f"{variation}/sep_sc")
        self._layout_ = "separate"
        self._scene_mode_ = "yes"
        self._align_ = None

# Arrays # 

class ArrayMasks_InputFormatModule(InputFormatModule):
    def __init__(self, variation):
        super().__init__(variation=f"{variation}/array_noImgs")
        self._layout_ = "array"
        self._scene_mode_ = "no"
        self._align_ = None

class ArrayMasks_Imgs_InputFormatModule(InputFormatModule):
    def __init__(self, variation):
        super().__init__(variation=f"{variation}/array_imgs")
        self._layout_ = "array_with_imgs"
        self._scene_mode_ = "no"
        self._align_ = None

class ArrayMasks_Imgs_Ovr_InputFormatModule(InputFormatModule):
    def __init__(self, variation):
        super().__init__(variation=f"{variation}/array_imgs_ovr")
        self._layout_ = "array_with_imgs"
        self._scene_mode_ = "overlay"
        self._align_ = None

### 4. Task ###

class TaskModule(PromptModule):
    def __init__(self, variation):
        super().__init__(f"4_task/{variation}")
    def __call__(self):
        return super().__call__()
    
### 5. Output Format ###

class OutputFormatModule(PromptModule):
    def __init__(self, variation):
        super().__init__(f"5_output_format/{variation}")
    def __call__(self):
        return super().__call__()
    
### 6. Support Set ###

class SupportSetModule(PromptModule):
    def __init__(self, variation, sup_set_idxs):
        super().__init__(f"6_support_set/{variation}")
        self.__sup_set_idxs__ = sup_set_idxs
    def __call__(self, sup_set_items):
        prompt = []
        if len(sup_set_items) != 0:
            prompt.append(super().__call__())
            for i, item in enumerate(sup_set_items):
                prompt.append(f"EXAMPLE {i+1}.")
                prompt.append(item)
        return prompt
    
### 7. Query ###
    
class QueryModule(PromptModule):
    def __init__(self, variation):
        super().__init__(f"7_query/{variation}")
    def __call__(self, query_item):
        prompt = []
        prompt.append(super().__call__())
        prompt.append(query_item)
        return prompt

### 8. Evaluation ###

class EvalModule(PromptModule):
    def __init__(self, variation):
        super().__init__(f"8_eval/{variation}")
    def __call__(self, target, answer):
        return pf.pformat(super().__call__(), target=target, answer=answer)
    
### Prompts Logic ###

class PromptBuilder():

    # TODO: can we parallelise or speed up the methods that build the prompts?

    def __init__(self, by_model, alpha, split_by, image_size, array_size, class_map, color_map):
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

    def read_sc_gt_pr(self, idx, image_size_):
        """
        Reads the scene, ground truth and prediction masks from disk and formats them in RGB format ready to be visualised.
        The color map is applied to the masks.
        """
        prs_path = get_mask_prs_path(self.by_model)
        sc = to_pil_image(get_sc(SCS_PATH / (image_UIDs[idx] + ".jpg"), image_size_))
        gt = apply_colormap(get_gt(GTS_PATH / (image_UIDs[idx] + ".png"), self.class_map, image_size_), self.color_map)
        pr = apply_colormap(get_pr(prs_path / f"mask_pr_{idx}.png", self.class_map, image_size_), self.color_map)
        assert sc.size == gt.size == pr.size
        return sc, gt, pr

    def get_state(self):
        """
        Returns the current state of the prompter as a dictionary.
        """
        def to_dict(obj):
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
    
    def build_img_prompt(self, img_idx, with_answer_gt=False):
        """
        Receives an image idx and optionally its target, and builds the individual formatted image prompt.
        The query and few-shot modules make use of this method to serve the images.
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
                answer_gt = get_one_answer_gt(self.by_model, img_idx)[img_idx]
            elif self.split_by == "class-splitted":
                answer_gt = get_one_sup_set_answer_gt(self.by_model, img_idx)[img_idx]
            img_prompts.append(answer_gt) # add target answer if specified
        return flatten_list(img_prompts)

    def inherit_settings_from_modules(self):
        """
        Inherit variables enclosed by '_' from the modules in 'self.modules_dict'.
        """
        for m in self.modules_dict.values():
            for attr, value in vars(m).items():
                if attr.startswith('_') and attr.endswith('_'):
                    setattr(self, attr.strip('_'), value)
    
    def load_modules(self, context_module, color_map_module, input_format_module, task_module, output_format_module, support_set_module, query_module, eval_module):
        """
        Load the prompt modules (in a strict order) and inherits their shared variables.
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

    def build_inference_prompt(self, query_idx):
        """
        Builds the inference prompt (in which the VLM performs the differencing in textual form).
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
    
    def create_class_specific_promptBuilder(self, pos_class):
        """
        Creates a clone of this 'PromptBuilder' class, but changes the color map so that:
        - The positive class is white (255, 255, 255).
        - All the other classes are black (0, 0, 0).
        """
        class_specific_promptBuilder = deepcopy(self)
        class_specific_promptBuilder.color_map = {c: [255, 255, 255] if c == pos_class else [0, 0, 0] for c in range(len(CLASSES))}
        return class_specific_promptBuilder
    
    def build_class_splitted_support_set_items(self):
        """
        Generates the few-shot example items in the class-splitted scenario.
        """
        sup_set_items = []
        for idx in self.sup_set_idxs:
            sup_set_items.append(self.build_class_specific_support_set_item(idx))
        return sup_set_items
    
    def build_class_specific_support_set_item(self, img_idx):
        """
        Generates a single few-shot example in a class-specific scenario.
        In this one, since the items have to class-specific, each image is associated with a single class index, marking the class the few-shot is going to display.
        However, the actual evaluation text is hard-coded (can be adapted in future to select change dynamically the positive class).
        """
        img_idx_to_class_ = {
            2: 20,
            16: 1,
            18: 13,
        }
        class_specific_promptBuilder = self.create_class_specific_promptBuilder(img_idx_to_class_[img_idx])
        return class_specific_promptBuilder.build_img_prompt(img_idx, with_answer_gt=True)
    
    def build_class_specific_inference_prompt(self, query_idx, pos_class):
        """
        Builds the inference prompt (in which the VLM performs the differencing in textual form) in a class-specific scenario..
        """
        significant_class_name = CLASSES[pos_class]
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

    def build_eval_prompt(self, query_idx: int, answer_pr: str) -> Prompt:
        """
        Builds the evaluation prompt (in which the LLM-as-a-Judge evaluates the differencing in textual form).
        """
        answer_gt = get_one_answer_gt(self.by_model, query_idx)[query_idx]
        prompt = [self.modules_dict["eval"](answer_gt, answer_pr)]
        return prompt
    
    def build_class_splitted_inference_prompts(
            self,
            query_idx: int
    ) -> dict[int, list[str]]:
        """
        Builds a list of full inference prompts for a given 'query_idx' split by class masks. 
        Each inference prompt masks only consider one class at a time.
        """
        # TODO: if the masks only have BACKGROUND class, there might be an error when trying to build the prompt.
        significant_classes_gt = get_significant_classes(GTS_PATH / (image_UIDs[query_idx] + ".png"), self.image_size, self.class_map)
        significant_classes_pr = get_significant_classes(get_mask_prs_path(self.by_model) / (f"mask_pr_{query_idx}.png"), self.image_size, self.class_map)
        significant_classes = sorted(list(set(significant_classes_gt + significant_classes_pr))) # all appearing classes
        class_splitted_prompts = {}
        for pos_class in significant_classes:
            class_specific_prompt = self.build_class_specific_inference_prompt(query_idx, pos_class)
            class_splitted_prompts[pos_class] = class_specific_prompt
        return class_splitted_prompts
    
    # TODO implement this method to be as fast as possible when the segNet is training.
    def build_class_splitted_inference_prompts_fixed(
            self,
            query_idx: int
    ) -> dict[int, list[str]]:
        pass

    def build_class_splitted_eval_prompt(self, query_idx, pos_class_2_answer_pr) -> dict[int, str]:
        """
        Builds a list of full evaluation prompts for a given 'query_idx' split by class masks. 
        Each evaluation prompt masks only consider one class at a time.
        """
        pos_class_2_eval_prompt = {}
        significant_classes = pos_class_2_answer_pr.keys()
        class_splitted_answer_pr = pos_class_2_answer_pr.values()
        for pos_class, answer_pr in zip(significant_classes, class_splitted_answer_pr):
            pos_class = int(pos_class)
            pos_class_2_eval_prompt[pos_class] = [pf.pformat(self.build_eval_prompt(query_idx, answer_pr)[0], pos_class=CLASSES[pos_class])]
        return pos_class_2_eval_prompt
    
def save_formatted_images(promptBuilder: PromptBuilder, img_idxs: tuple[int]) -> None:
    for img_idx in img_idxs:
        sc, gt, pr = promptBuilder.read_sc_gt_pr(img_idx, promptBuilder.image_size)
        formatted_image = _format_images(sc, gt, pr, img_idx, promptBuilder.layout, promptBuilder.scene_mode, promptBuilder.align, promptBuilder.alpha)[0]
        formatted_image.save(LOCAL_ANNOT_IMGS_PATH / f"annot_image_{img_idx}.png") 

if __name__ == "__main__":

    PromptModule.prompts_path = get_prompts_path("non-splitted")

    promptBuilder = PromptBuilder(
        by_model            = "LRASPP_MobileNet_V3",
        alpha               = 0.8,
        split_by            = "class-splitted",
        image_size          = 520,
        array_size          = (32, 32),
        class_map           = CLASS_MAP, # imported from 'class_map.py'
        color_map           = COLOR_MAP_DICT,
    )
    
    promptBuilder.load_modules(
    context_module          = ContextModule(variation="default"),
    color_map_module        = ClassSplitted_ColorMapModule(variation="default"),
    input_format_module     = ArrayMasks_Imgs_Ovr_InputFormatModule("original"),
    task_module             = TaskModule(variation="default"),
    output_format_module    = OutputFormatModule(variation="default"),
    support_set_module      = SupportSetModule(variation="default", sup_set_idxs=(16, 2, 18)),
    query_module            = QueryModule(variation="default"),
    eval_module             = EvalModule(variation="3_specify_pos_class_recency")
    )

    print(promptBuilder.build_class_specific_inference_prompt(0, 15))
