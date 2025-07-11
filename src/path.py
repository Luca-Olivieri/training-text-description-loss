from config import *

from pathlib import Path

### Constant Paths ###

SRC_PATH =  BASE_PATH / "src"
PRIVATE_DATASETS_PATH = BASE_PATH / "data"
MISC_PATH = BASE_PATH / "misc"

VOC_PATH = PRIVATE_DATASETS_PATH / "VOCdevkit/VOC2012"
MY_DATA_PATH = BASE_PATH / "my_data"
SPLITS_PATH = VOC_PATH / "ImageSets/Segmentation"

SCS_PATH = VOC_PATH / "JPEGImages"
GTS_PATH = VOC_PATH / "SegmentationClass"

LOCAL_ANNOT_IMGS_PATH = BASE_PATH / "annot_images"

def get_selected_model_path(
        by_model: str
) -> Path:
    """
    Gets the path to the selected model directory for a given model.

    Args:
        by_model: The name of the model.

    Returns:
        The path to the selected model directory.
    """
    return MY_DATA_PATH / "by_model" / by_model

def get_selected_model_split_path(
        by_model: str,
        split_by: str
) -> Path:
    """
    Gets the path to the split directory for a given model and split type.

    Args:
        by_model: The name of the model.
        split_by: The type of split.

    Returns:
        The path to the split directory for the model."""
    return get_selected_model_path(by_model) / split_by

def get_answer_prs_root_path(
        by_model: str,
        split_by: str
) -> Path:
    """
    Gets the root path for answer predictions for a given model and split type.

    Args:
        by_model: The name of the model.
        split_by: The type of split.

    Returns:
        The root path for answer predictions.
    """
    return get_selected_model_path(by_model) / split_by / "answer_prs"

def get_eval_prs_root_path(
        by_model: str,
        split_by: str
) -> Path:
    """
    Gets the root path for evaluation predictions for a given model and split type.

    Args:
        by_model: The name of the model.
        split_by: The type of split.

    Returns:
        The root path for evaluation predictions.
    """
    return get_selected_model_path(by_model) / split_by / "eval_prs"

### Variable Paths ###

def get_prompts_path(split_by: str) -> Path:
    """
    Gets the path to the prompts directory for a given split type.

    Args:
        split_by: The type of split.

    Returns:
        The path to the prompts directory.
    """
    prompts_path = MY_DATA_PATH / "prompts" / split_by
    return prompts_path

def get_data_gen_prompts_path() -> Path:
    """
    Returns the path to the synthetic data generation prompts directory.
    """
    prompts_path = MY_DATA_PATH / "prompts"
    return prompts_path

def get_mask_prs_path(by_model: str) -> Path:
    """
    Gets the path to the mask predictions directory for a given model.

    Args:
        by_model: The name of the model.

    Returns:
        The path to the mask predictions directory.
    """
    prs_path = get_selected_model_path(by_model) / "_mask_prs_"
    return prs_path

def get_answer_gts_path(by_model: str) -> Path:
    """
    Gets the path to the answer ground truth file for a given model.

    Args:
        by_model: The name of the model.

    Returns:
        The path to the answer ground truth file.
    """
    answers_gts_path = get_selected_model_path(by_model) / "answer_gts.jsonl"
    return answers_gts_path

def get_sup_set_answer_gts_path(
        by_model,
        split_by="class-splitted"
) -> Path:
    """
    Gets the path to the supervised set answer ground truth file for a given model and split type.

    Args:
        by_model: The name of the model.
        split_by: The type of split. Defaults to "class-splitted".

    Returns:
        The path to the supervised set answer ground truth file.
    """
    answers_gts_path = get_selected_model_split_path(by_model, split_by) / "sup_set_answer_gts.jsonl"
    return answers_gts_path

def get_answer_prs_path(
        by_model: str,
        split_by: str,
        rel_path: str
) -> Path:
    """
    Gets the path to the answer predictions file for a given model, split type, and relative path.

    Args:
        by_model: The name of the model.
        split_by: The type of split.
        rel_path: The relative path for the predictions file.

    Returns:
        The path to the answer predictions file.
    """
    answers_prs_path = get_selected_model_split_path(by_model, split_by) / "answer_prs" / f"{rel_path}.jsonl"
    return answers_prs_path

def get_eval_gts_path(
        by_model,
        split_by
) -> Path:
    """
    Gets the path to the evaluation ground truth file for a given model and split type.

    Args:
        by_model: The name of the model.
        split_by: The type of split.

    Returns:
        The path to the evaluation ground truth file.
    """
    eval_gts_path = get_selected_model_split_path(by_model, split_by) / "eval_gts.jsonl"
    return eval_gts_path

def get_eval_prs_path(
        by_model: str,
        split_by: str,
        rel_path: str
) -> Path:
    """
    Gets the path to the evaluation predictions file for a given model, split type, and relative path.

    Args:
        by_model: The name of the model.
        split_by: The type of split.
        rel_path: The relative path for the predictions file.

    Returns:
        The path to the evaluation predictions file.
    """
    eval_prs_path = get_selected_model_split_path(by_model, split_by) / "eval_prs" / f"{rel_path}.jsonl"
    return eval_prs_path

def main() -> None:
    print(os.environ["HF_HOME"])
    print(os.environ["HF_HUB_CACHE"])

if __name__ == '__main__':
    main()

