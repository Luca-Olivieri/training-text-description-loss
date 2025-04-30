from pathlib import Path
import torch

### Constant Paths ###

ROOT_PATH = Path("/home/olivieri/exp").resolve() # outer-most project path

SRC_PATH =  ROOT_PATH / "src"
CONFIG_PATH =  ROOT_PATH / "config"
PRIVATE_DATASETS_PATH = ROOT_PATH / "data"
MISC_PATH = ROOT_PATH / "misc"

MODEL_WEIGHTS_ROOT = PRIVATE_DATASETS_PATH / "torch_weights"
torch.hub.set_dir(MODEL_WEIGHTS_ROOT) # set local model weights directory

VOC_PATH = PRIVATE_DATASETS_PATH / "VOCdevkit/VOC2012"
MY_ANNOTS_PATH = VOC_PATH / "MyAnnotations"
SPLITS_PATH = VOC_PATH / "ImageSets/Segmentation"

SCS_PATH = VOC_PATH / "JPEGImages"
GTS_PATH = VOC_PATH / "SegmentationClass"

def get_selected_annots_path(by_model, split):
    return MY_ANNOTS_PATH / "by_model" / by_model / split

### Variable Paths ###

def get_prompts_path(split):
    prompts_path = MY_ANNOTS_PATH / "prompts" / split
    return prompts_path

def get_mask_prs_path(by_model):
    prs_path = MY_ANNOTS_PATH / "by_model" / by_model / "_mask_prs_"
    return prs_path

def get_answer_gts_path(by_model, split):
    answers_gts_path = get_selected_annots_path(by_model, split) / "answer_gts.jsonl"
    return answers_gts_path

def get_sup_set_answer_gts_path(by_model, split):
    answers_gts_path = get_selected_annots_path(by_model, split) / "sup_set_answer_gts.jsonl"
    return answers_gts_path

def get_answer_prs_path(by_model, split, variation):
    answers_prs_path = get_selected_annots_path(by_model, split) / "answer_prs" / f"{variation}.jsonl"
    return answers_prs_path

def get_eval_gts_path(by_model, split):
    eval_gts_path = get_selected_annots_path(by_model, split) / "eval_gts.jsonl"
    return eval_gts_path

def get_eval_prs_path(by_model, split, variation):
    eval_prs_path = get_selected_annots_path(by_model, split) / "eval_prs" / f"{variation}.jsonl"
    return eval_prs_path

if __name__ == "__main__":
    print(f"{ROOT_PATH=}")
    print(f"{PRIVATE_DATASETS_PATH=}")
    print(f"{MY_ANNOTS_PATH=}")
    print(f"{SCS_PATH=}")
    print(f"{GTS_PATH=}")
