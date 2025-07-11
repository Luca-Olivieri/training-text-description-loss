from config import *
from models.vl_models import GenParams, OllamaMLLM
from prompter import FastPromptBuilder
from data import CLASS_MAP, append_many_to_jsonl, get_image_UIDs, image_UIDs, SegDataset, crop_augment_preprocess_batch
from color_map import COLOR_MAP_DICT
from path import SPLITS_PATH, get_mask_prs_path

from pathlib import Path
from torchvision.models import segmentation as segmodels
from functools import partial
from torchvision.transforms._presets import SemanticSegmentation
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from collections import OrderedDict

import asyncio

async def main() -> None:

    var_name = 'train_no_aug'

    answer_path = Path('/home/olivieri/exp/data/data_gen/VOC2012') / f"{var_name}.jsonl"

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
        image_size=224,
        sup_set_img_idxs=[16],
        str_formats=None,
        prs_mask_paths=get_mask_prs_path(by_model=by_model)
    )

    append_many_to_jsonl(answer_path, [{"state": fast_prompt_builder.get_state()} | {"vlm": f"{vlm.__class__.__name__}:{vlm.model}"}])

    train_image_UIDs_ = np.array([uid for uid in sorted(get_image_UIDs(SPLITS_PATH, split='train', shuffle=False)) if uid != image_UIDs[16]])
    val_image_UIDs_ = np.array(sorted(get_image_UIDs(SPLITS_PATH, split='val', shuffle=False)))
    len(train_image_UIDs_), len(val_image_UIDs_)

    # train_ds = SegDataset(train_image_UIDs_, CONFIG['seg']['image_size'], CLASS_MAP)
    # val_ds = SegDataset(val_image_UIDs_, CONFIG['seg']['image_size'], CLASS_MAP)

    offset = 384

    ds = SegDataset(train_image_UIDs_[offset:], CONFIG['seg']['image_size'], CLASS_MAP)

    segnet = segmodels.lraspp_mobilenet_v3_large(weights=None, weights_backbone=None).to(CONFIG["device"])
    segnet.load_state_dict(torch.load(TORCH_WEIGHTS_CHECKPOINTS / ("lraspp_mobilenet_v3_large-full-pt" + ".pth")))
    segnet.requires_grad_(False)
    segnet.eval()

    preprocess_fn = partial(SemanticSegmentation, resize_size=CONFIG['seg']['image_size'])() # same as original one, but with custom resizing

    center_crop_fn = T.CenterCrop(CONFIG['seg']['image_size'])
    random_crop_fn = T.RandomCrop(CONFIG['seg']['image_size'])

    # augmentations
    augment_fn = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=0, scale=(0.5, 2)), # Zooms in and out of the image.
    ])
    augment_fn = None

    collate_fn = partial(
        crop_augment_preprocess_batch,
        crop_fn=center_crop_fn,
        augment_fn=None,
        preprocess_fn=None)

    dl = DataLoader(
            ds,
            batch_size=CONFIG["seg"]["batch_size"],
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
            
            cs_prompts = fast_prompt_builder.build_cs_inference_prompts(gts.unsqueeze(1), prs, scs_img)

            epoch_idxs = [offset + dl.batch_size*step + i for i in range(dl.batch_size)]

            await vlm.predict_many_class_splitted(
                cs_prompts,
                epoch_idxs,
                gen_params=gen_params,
                jsonl_save_path=answer_path,
                only_text=True,
                splits_in_parallel=False,
                batch_size=None,
                use_tqdm=False
            )

if __name__ == '__main__':
    asyncio.run(main())
