from config import *
from models.vl_models import GenParams, OllamaMLLM
from models.seg_models import SegModelWrapper, SEGMODELS_REGISTRY
from prompter import FastPromptBuilder
from data import VOC2012SegDataset, COCO2017SegDataset, append_many_to_jsonl, crop_augment_preprocess_batch, apply_classmap
from color_map import apply_colormap
from path import get_mask_prs_path
from utils import blend_tensors, create_directory
from viz import create_diff_mask

from pathlib import Path
from torchvision.models import segmentation as segmodels
from functools import partial
import torchvision
from torchvision.transforms._presets import SemanticSegmentation
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from collections import OrderedDict

import asyncio

import torch

DATA_GEN_CONFIG = CONFIG['data_gen']

async def main() -> None:

    exp_path = create_directory(Path(DATA_GEN_CONFIG['data_root']), DATA_GEN_CONFIG['exp_name'])
    images_RB_path = create_directory(exp_path,  "images_RB")
    images_L_path = create_directory(exp_path,  "images_L")
    captions_path = exp_path / "captions.jsonl"

    model_name = "gemma3:12b-it-qat"

    vlm = OllamaMLLM(model_name, container_name=CONFIG['ollama_container_name'])

    by_model = "LRASPP_MobileNet_V3"

    gen_params = GenParams(
        seed=CONFIG["seed"],
        temperature=DATA_GEN_CONFIG['temperature'],
        top_p=DATA_GEN_CONFIG['top_p']
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
    
    offset = DATA_GEN_CONFIG['offset']

    seg_dataset = COCO2017SegDataset(
        root_path=Path(CONFIG['datasets']['COCO2017_root_path']),
        split='train',
        img_idxs=slice(offset, None, None),
        resize_size=CONFIG['seg']['image_size'],
        center_crop=True,
        only_VOC_labels=True
    )
    print(f"{len(seg_dataset)} unique images.")

    sup_set_seg_dataset = VOC2012SegDataset(
        root_path=Path("/home/olivieri/exp/data/VOCdevkit"),
        split='prompts_split',
        resize_size=CONFIG['seg']['image_size'],
        center_crop=True,
        with_unlabelled=False,
        mask_prs_path=get_mask_prs_path(by_model=by_model)
    )

    fast_prompt_builder = FastPromptBuilder(
        seg_dataset=seg_dataset,
        seed=CONFIG["seed"],
        prompt_blueprint=prompt_blueprint,
        by_model=by_model,
        alpha=0.6,
        class_map=seg_dataset.get_class_map(with_unlabelled=False),
        color_map=seg_dataset.get_color_map_dict(with_unlabelled=False),
        image_size=CONFIG['vlm']['image_size'],
        sup_set_img_idxs=[16],
        sup_set_seg_dataset=sup_set_seg_dataset,
        str_formats=None,
    )

    append_many_to_jsonl(captions_path, [{"state": fast_prompt_builder.get_state()} | {"vlm": f"{vlm.__class__.__name__}:{vlm.model}"}])

    segnet = segmodels.lraspp_mobilenet_v3_large(weights=None, weights_backbone=None).to(CONFIG["device"])
    segnet.load_state_dict(torch.load(Path(DATA_GEN_CONFIG['pretrained_weights_path'])))
    segnet.requires_grad_(False)
    segnet = segnet.eval()

    preprocess_fn = partial(SemanticSegmentation, resize_size=CONFIG['seg']['image_size'])() # same as original one, but with custom resizing

    collate_fn = partial(
        crop_augment_preprocess_batch,
        crop_fn=T.CenterCrop(CONFIG['seg']['image_size']),
        augment_fn=None,
        preprocess_fn=None
    )

    dl = DataLoader(
        seg_dataset,
        batch_size=CONFIG["data_gen"]["batch_size"],
        shuffle=False,
        generator=get_torch_gen(),
        collate_fn=collate_fn,
    )

    # Generation

    with torch.inference_mode():

        img_idx_count = 0
        cs_img_idx_count = 0

        for step, (scs_img, gts) in enumerate(dl):

            if preprocess_fn: scs = preprocess_fn(scs_img)

            scs = scs.to(CONFIG["device"])
            gts = gts.to(CONFIG["device"]) # shape [N, H, W]

            logits = segnet(scs)
            logits: torch.Tensor = logits["out"] if isinstance(logits, OrderedDict) else logits # shape [N, C, H, W]
            prs = logits.argmax(dim=1, keepdim=True)

            # NOTE only for COCO
            if hasattr(seg_dataset, 'only_VOC_labels') and seg_dataset.only_VOC_labels:
                prs = apply_classmap(prs, seg_dataset.voc_idx_to_coco_idx)
                prs = apply_classmap(prs, seg_dataset.get_class_map())

            scs_img = (scs_img*255).to(torch.uint8)

            gts = gts.unsqueeze(1)

            # Both VLM and VLE receive the images in the same size.
            gts_down = TF.resize(gts, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
            prs_down = TF.resize(prs, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
            scs_down = TF.resize(scs_img, fast_prompt_builder.image_size, TF.InterpolationMode.BILINEAR)
            
            cs_prompts = fast_prompt_builder.build_cs_inference_prompts(gts_down, prs_down, scs_down)

            batch_idxs = [dl.batch_size*step + i for i in range(len(scs_down))]
            batch_image_uids = seg_dataset.image_UIDs[batch_idxs]

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
                # VLEs and VLMs are fed usually fed downsampled images, therefore they might not see some classes which appear in the saved images.
                # The classes that do not appear in the downsampled classes are skipped (the corresponding splits are not generated).
                sign_classes = fast_prompt_builder.extract_significant_classes(
                    TF.resize(gt, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST),
                    TF.resize(pr, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
                )

                sign_classes = sorted(sign_classes, key=lambda pos_c: str(pos_c))
                
                for pos_c in sign_classes:

                    append_many_to_jsonl(captions_path, [{'img_idx': img_idx + offset, "img_uid": str(img_uid), "pos_class": pos_c, "content": cs_ans["content"][pos_c]}])

                    pos_class_gt = (gt == pos_c)
                    pos_class_pr = (pr == pos_c)

                    diff_mask = create_diff_mask(pos_class_gt, pos_class_pr)

                    # L overlay image
                    ovr_diff_mask_L = blend_tensors(sc_img, diff_mask*255, DATA_GEN_CONFIG['alpha'])
                    torchvision.utils.save_image(ovr_diff_mask_L/255., images_L_path / f"{img_uid}-{pos_c}.png", normalize=True)

                    # RB overlay image
                    # diff_mask += (diff_mask*pos_class_gt) # sets to 2 the false negatives
                    # diff_mask_col_RB = apply_colormap([diff_mask], {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 0, 255)})
                    # ovr_diff_mask_RB = blend_tensors(sc_img, diff_mask_col_RB, DATA_GEN_CONFIG['alpha'])
                    # torchvision.utils.save_image(ovr_diff_mask_RB/255., images_RB_path / f"{img_uid}-{pos_c}.png", normalize=True)

                    cs_img_idx_count += 1

                img_idx_count += 1

                if img_idx_count % DATA_GEN_CONFIG['print_every'] == 0:
                    print(f"Generated {img_idx_count} unique photos ({cs_img_idx_count} class-splitted photos).")
                    

if __name__ == '__main__':
    asyncio.run(main())
