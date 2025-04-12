from pathlib import Path

### Constant Paths ###

ROOT_PATH = Path("/home/olivieri/exp").resolve() # outer-most project path

SRC_PATH =  ROOT_PATH / "src"
CONFIG_PATH =  ROOT_PATH / "config"
PRIVATE_DATASETS_PATH = ROOT_PATH / "data"
MISC_PATH = ROOT_PATH / "misc"

VOC_PATH = PRIVATE_DATASETS_PATH / "VOCdevkit/VOC2012"
MY_ANNOTS_PATH = VOC_PATH / "MyAnnotations"
SPLITS_PATH = VOC_PATH / "ImageSets/Segmentation"

SCS_PATH = VOC_PATH / "JPEGImages"
GTS_PATH = VOC_PATH / "SegmentationClass"

def get_selected_annots_path(by_model, image_resizing_mode, output_mode):
    return MY_ANNOTS_PATH / "by_model" / by_model / image_resizing_mode / output_mode

### Variable Paths ###

def get_prompts_path(output_mode, image_resizing_mode):
    prompts_path = MY_ANNOTS_PATH / "prompts" / image_resizing_mode / output_mode
    return prompts_path

def get_mask_prs_path(by_model, image_resizing_mode):
    prs_path = MY_ANNOTS_PATH / "by_model" / by_model / image_resizing_mode / "_mask_prs_"
    return prs_path

def get_answer_gts_path(by_model, image_resizing_mode, output_mode):
    answers_gts_path = get_selected_annots_path(by_model, image_resizing_mode, output_mode) / "answer_gts.jsonl"
    return answers_gts_path

def get_answer_prs_path(by_model, image_resizing_mode, output_mode, variation):
    answers_prs_path = get_selected_annots_path(by_model, image_resizing_mode, output_mode) / "answer_prs" / f"{variation}.jsonl"
    return answers_prs_path

def get_eval_gts_path(by_model, image_resizing_mode, output_mode):
    eval_gts_path = get_selected_annots_path(by_model, image_resizing_mode, output_mode) / "eval_gts.jsonl"
    return eval_gts_path

def get_eval_prs_path(by_model, image_resizing_mode, output_mode, variation):
    eval_prs_path = get_selected_annots_path(by_model, image_resizing_mode, output_mode) / "eval_prs" / f"{variation}.jsonl"
    return eval_prs_path

if __name__ == "__main__":
    print(f"{ROOT_PATH=}")
    print(f"{PRIVATE_DATASETS_PATH=}")
    print(f"{MY_ANNOTS_PATH=}")
    print(f"{SCS_PATH=}")
    print(f"{GTS_PATH=}")
