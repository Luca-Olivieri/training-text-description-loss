from config import *
from models.vl_models import GenParams, OllamaMLLM
from prompter import FastPromptBuilder
from data import CLASS_MAP, append_many_to_jsonl, get_image_UIDs, SegDataset, crop_augment_preprocess_batch, apply_classmap
from color_map import COLOR_MAP_DICT
from path import SPLITS_PATH, get_mask_prs_path
from utils import blend_tensors

from pathlib import Path
from torchvision.models import segmentation as segmodels
from functools import partial
import torchvision
from torchvision.transforms._presets import SemanticSegmentation
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from collections import OrderedDict

import asyncio

import torch

def create_diff_mask(
        mask1: torch.Tensor,
        mask2: torch.Tensor
) -> torch.Tensor:
    """
    Creates a binary difference mask from two integer-based segmentation masks.

    The operation is fully vectorized and runs efficiently on CUDA devices.

    Args:
        mask1 (torch.Tensor): The first segmentation mask with class indices.
                              Expected dtype: torch.long, torch.int, etc.
        mask2 (torch.Tensor): The second segmentation mask with class indices.
                              Must have the same shape and device as mask1.

    Returns:
        torch.Tensor: A mask with value 255 (uint8) where pixels in mask1 and mask2
                      are different, and 0 where they are the same.
    """
    # 1. Ensure the masks have the same shape for element-wise comparison.
    assert mask1.shape == mask2.shape, f"Input masks must have the same shape, but got {mask1.shape} and {mask2.shape}"
    
    # 2. Perform element-wise comparison. This creates a boolean tensor.
    #    'True' where elements are not equal, 'False' where they are equal.
    #    This is the functional equivalent of `torch.ne(mask1, mask2)`.
    diff = (mask1 != mask2)
    
    # 3. Convert the boolean tensor to an 8-bit unsigned integer tensor and scale.
    #    In PyTorch, casting a boolean to a numeric type converts True -> 1 and False -> 0.
    #    We then multiply by 255 to get the desired output range.
    diff_mask = diff.to(torch.uint8) * 255
    
    return diff_mask

async def main() -> None:

    exp_path = Path(CONFIG['data_gen']['root']) / f"{CONFIG['data_gen']['exp_name']}"
    captions_path = exp_path / "captions.jsonl"
    images_path = exp_path / "images"

    model_name = "gemma3:12b-it-qat"

    vlm = OllamaMLLM(model_name)

    by_model = "LRASPP_MobileNet_V3"

    gen_params = GenParams(
        seed=CONFIG["seed"],
        temperature=0.1
    )

    prompt_blueprint={
            "context": "default",
            "color_map": "default",
            "input_format": "sep_ovr_original",
            "task": "default",
            "output_format": "default",
            "support_set_intro": "default",
            "support_set_item": "default",
            "query": "default",
    }

    fast_prompt_builder = FastPromptBuilder(
        seed=CONFIG["seed"],
        prompt_blueprint=prompt_blueprint,
        by_model=by_model,
        alpha=0.6,
        class_map=CLASS_MAP,
        color_map=COLOR_MAP_DICT,
        image_size=CONFIG['vlm']['image_size'],
        sup_set_img_idxs=[16],
        str_formats=None,
        prs_mask_paths=get_mask_prs_path(by_model=by_model)
    )

    append_many_to_jsonl(captions_path, [{"state": fast_prompt_builder.get_state()} | {"vlm": f"{vlm.__class__.__name__}:{vlm.model}"}])

    train_image_UIDs_ = get_image_UIDs(SPLITS_PATH, split='train', shuffle=False, uids_to_exclude=['2007_000256'])
    val_image_UIDs_ = get_image_UIDs(SPLITS_PATH, split='val', shuffle=False, uids_to_exclude=['2007_000256'])
    len(train_image_UIDs_), len(val_image_UIDs_)

    image_uids = train_image_UIDs_

    offset = 32

    ds = SegDataset(image_uids[offset:], CONFIG['seg']['image_size'], CLASS_MAP)
    print(len(ds))

    segnet = segmodels.lraspp_mobilenet_v3_large(weights=None, weights_backbone=None).to(CONFIG["device"])
    segnet.load_state_dict(torch.load(TORCH_WEIGHTS_CHECKPOINTS / ("lraspp_mobilenet_v3_large-full-pt" + ".pth")))
    segnet.requires_grad_(False)
    segnet.eval()

    preprocess_fn = partial(SemanticSegmentation, resize_size=CONFIG['seg']['image_size'])() # same as original one, but with custom resizing

    center_crop_fn = T.CenterCrop(CONFIG['seg']['image_size'])
    random_crop_fn = T.RandomCrop(CONFIG['seg']['image_size'])

    # augmentations
    augment_fn = None

    collate_fn = partial(
        crop_augment_preprocess_batch,
        crop_fn=center_crop_fn,
        augment_fn=None,
        preprocess_fn=None)

    dl = DataLoader(
            ds,
            batch_size=CONFIG["data_gen"]["batch_size"],
            shuffle=False,
            generator=TORCH_GEN.clone_state(),
            collate_fn=collate_fn,
        )

    # Generation

    with torch.inference_mode():
        for step, (scs_img, gts) in enumerate(dl):

            if augment_fn: scs_img, gts = augment_fn(scs_img, tv_tensors.Mask(gts))

            if preprocess_fn: scs = preprocess_fn(scs_img)

            scs = scs.to(CONFIG["device"])
            gts = gts.to(CONFIG["device"]) # shape [N, H, W]

            logits = segnet(scs)
            logits: torch.Tensor = logits["out"] if isinstance(logits, OrderedDict) else logits # shape [N, C, H, W]
            prs = logits.argmax(dim=1, keepdim=True)
            scs_img = (scs_img*255).to(torch.uint8)

            gts = gts.unsqueeze(1)

            # Both VLM and VLE receive the images in the same size.
            gts = TF.resize(gts, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
            prs = TF.resize(prs, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
            scs_img = TF.resize(scs_img, fast_prompt_builder.image_size, TF.InterpolationMode.BILINEAR)
            
            cs_prompts = fast_prompt_builder.build_cs_inference_prompts(gts, prs, scs_img)

            batch_idxs = [offset + dl.batch_size*step + i for i in range(dl.batch_size)]
            batch_image_uids = image_uids[batch_idxs]

            cs_answer_list = await vlm.predict_many_class_splitted(
                cs_prompts,
                batch_idxs,
                gen_params=gen_params,
                jsonl_save_path=None,
                only_text=True,
                splits_in_parallel=False,
                batch_size=None,
                use_tqdm=False
            )

            for img_idx, img_uid, cs_ans, sc_img, gt, pr in zip(batch_idxs, batch_image_uids, cs_answer_list, scs_img, gts, prs):
                sign_classes = fast_prompt_builder.extract_significant_classes(gt, pr)
                
                for pos_c in sign_classes:

                    append_many_to_jsonl(captions_path, [{"img_uid": img_uid, "pos_class": pos_c, "content": cs_ans["content"][pos_c]}])

                    pos_class_gt = (gt == pos_c)
                    pos_class_pr = (pr == pos_c)

                    diff_mask = create_diff_mask(pos_class_gt, pos_class_pr)

                    ovr_diff_mask = blend_tensors(sc_img, diff_mask, CONFIG['data_gen']['alpha'])
                    
                    torchvision.utils.save_image(ovr_diff_mask/255., images_path / f"{img_uid}-{pos_c}.png", normalize=True)


if __name__ == '__main__':
    asyncio.run(main())
