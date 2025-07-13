from config import *
from data import JSONLDataset, SegDataset, ImageCaptionDataset, CLASS_MAP, get_image_UIDs, crop_augment_preprocess_batch
from path import SPLITS_PATH
from models.vl_encoders import VLE_REGISTRY, VLEncoder
from viz import print_layer_numel

from torch import nn
from torchvision.models import segmentation as segmodels
from functools import partial
from torchvision.transforms._presets import SemanticSegmentation
import torchvision.transforms as T
from torch.utils.data import DataLoader
from collections import OrderedDict

def main() -> None:

    offset = ...

    captions_ds = JSONLDataset(Path("/home/olivieri/exp/data/data_gen/VOC2012/flat/train_no_aug_flat.jsonl"))
    masks_ds = SegDataset(get_image_UIDs(SPLITS_PATH, split="train", shuffle=False, uids_to_exclude=['2007_000256']), CONFIG['seg']['image_size'], CLASS_MAP)

    image_caption_ds = ImageCaptionDataset(masks_ds, captions_ds)

    vle: VLEncoder = VLE_REGISTRY.get("flair", device=CONFIG['device'])
    vle.set_vision_trainable_params('visual_proj')
    print_layer_numel(vle, print_only_total=True, only_trainable=True)

    segnet = segmodels.lraspp_mobilenet_v3_large(weights=None, weights_backbone=None).to(CONFIG["device"])
    segnet.load_state_dict(torch.load(TORCH_WEIGHTS_CHECKPOINTS / ("lraspp_mobilenet_v3_large-full-pt" + ".pth")))
    segnet.requires_grad_(False)
    segnet.eval()

    center_crop_fn = T.CenterCrop(CONFIG['seg']['image_size'])
    
    segnet_preprocess_fn = partial(SemanticSegmentation, resize_size=CONFIG['seg']['image_size'])()

    collate_fn = partial(
        crop_augment_preprocess_batch,
        crop_fn=center_crop_fn,
        augment_fn=None,
        preprocess_fn=None
    )

    image_caption_dl = DataLoader(
        image_caption_ds,
        batch_size=CONFIG["seg"]["batch_size"],
        shuffle=True,
        generator=TORCH_GEN.clone_state(),
        collate_fn=collate_fn,
    )

    criterion = ...

    lr = 1e-4
    optimizer = torch.optim.AdamW(vle.model.parameters(), lr=lr)

    for step, (scs_img, gts, text) in enumerate(image_caption_dl):

            if segnet_preprocess_fn: scs = segnet_preprocess_fn(scs_img)

            scs = scs.to(CONFIG["device"])
            gts = gts.to(CONFIG["device"]) # shape [N, H, W]

            optimizer.zero_grad()

            vle.encode_and_project()

            logits = segnet(scs)
            logits: torch.Tensor = logits["out"] if isinstance(logits, OrderedDict) else logits # shape [N, C, H, W]
            prs = logits.argmax(dim=1, keepdim=True)
            scs_img = (scs_img*255).to(torch.uint8)

if __name__ == '__main__':
    main()
