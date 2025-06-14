from pathlib import Path
import torch

from config import *

### Constant Paths ###

SRC_PATH =  BASE_PATH / "src"
PRIVATE_DATASETS_PATH = BASE_PATH / "data"
MISC_PATH = BASE_PATH / "misc"

MODEL_WEIGHTS_ROOT = PRIVATE_DATASETS_PATH / "torch_weights"
torch.hub.set_dir(MODEL_WEIGHTS_ROOT) # set local model weights directory
# os.environ["HF_HOME"] = str(PRIVATE_DATASETS_PATH / "huggingface_hub") # HuggingFace Hub Directory

VOC_PATH = PRIVATE_DATASETS_PATH / "VOCdevkit/VOC2012"
MY_DATA_PATH = BASE_PATH / "my_data"
SPLITS_PATH = VOC_PATH / "ImageSets/Segmentation"

SCS_PATH = VOC_PATH / "JPEGImages"
GTS_PATH = VOC_PATH / "SegmentationClass"

LOCAL_ANNOT_IMGS_PATH = BASE_PATH / "annot_images"

def get_selected_model_path(by_model: str) -> Path:
    return MY_DATA_PATH / "by_model" / by_model

def get_selected_model_split_path(by_model: str, split_by: str) -> Path:
    return get_selected_model_path(by_model) / split_by

def get_answer_prs_root_path(by_model: str, split_by: str) -> Path:
    return get_selected_model_path(by_model) / split_by / "answer_prs"

def get_eval_prs_root_path(by_model: str, split_by: str) -> Path:
    return get_selected_model_path(by_model) / split_by / "eval_prs"

### Variable Paths ###

def get_prompts_path(split) -> Path:
    prompts_path = MY_DATA_PATH / "prompts" / split
    return prompts_path

def get_mask_prs_path(by_model):
    prs_path = get_selected_model_path(by_model) / "_mask_prs_"
    return prs_path

def get_answer_gts_path(by_model):
    answers_gts_path = get_selected_model_path(by_model) / "answer_gts.jsonl"
    return answers_gts_path

def get_sup_set_answer_gts_path(by_model, split_by="class-splitted"):
    answers_gts_path = get_selected_model_split_path(by_model, split_by) / "sup_set_answer_gts.jsonl"
    return answers_gts_path

def get_answer_prs_path(by_model, split_by, rel_path):
    answers_prs_path = get_selected_model_split_path(by_model, split_by) / "answer_prs" / f"{rel_path}.jsonl"
    return answers_prs_path

def get_eval_gts_path(by_model, split_by):
    eval_gts_path = get_selected_model_split_path(by_model, split_by) / "eval_gts.jsonl"
    return eval_gts_path

def get_eval_prs_path(by_model, split_by, rel_path):
    eval_prs_path = get_selected_model_split_path(by_model, split_by) / "eval_prs" / f"{rel_path}.jsonl"
    return eval_prs_path

def main() -> None:
    pass

if __name__ == '__main__':
    main()

