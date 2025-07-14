from config import *
from data import JSONLDataset, ImageDataset, ImageCaptionDataset, CLASS_MAP, get_image_UIDs, crop_image_preprocess_image_text_batch
from path import SPLITS_PATH
from models.vl_encoders import VLE_REGISTRY, VLEncoder
from viz import get_layer_numel_str
from utils import get_compute_capability
from logger import get_logger, log_intro, log_title

from torch import nn
from torchvision.models import segmentation as segmodels
from functools import partial
from torchvision.transforms._presets import SemanticSegmentation
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from collections import OrderedDict
import math

from typing import Callable

from vendors.flair.src.flair.train import backward, unwrap_model

def train_loop(
        vle: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        criterion: Callable,
) -> None:
    
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(vle.model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": 1e-2}, # authors use 0.5
        ],
        lr=5e-4,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    lr = 1e-4
    optimizer = torch.optim.AdamW(vle.model.parameters(), lr=lr)

    if get_compute_capability() >= 7.0:
        vle.model = torch.compile(vle.model)
    
    for epoch in range(CONFIG["vle"]['train']["num_epochs"]):

        for step, (images, texts) in enumerate(train_dl):

                # TODO handle AMP
                
                vle.model.train()

                vle_output = vle.encode_and_project(images, texts, broadcast=False)

                optimizer.zero_grad()

                losses = criterion(
                        image_features=vle_output.global_image_token,
                        image_tokens=vle_output.local_image_tokens.clone(),
                        text_features=vle_output.global_text_token.squeeze(1),
                        logit_scale=vle.model.logit_scale,
                        visual_proj=vle.model.visual_proj,
                        logit_bias=vle.model.logit_bias,
                        output_dict=True
                )
                total_loss = sum(losses.values())

                scaler = None # for AMP
                backward(total_loss, scaler)

                grad_clip_norm = None
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(vle.model.parameters(), grad_clip_norm, norm_type=2.0)
                
                optimizer.step()

                if (step+1) % 10 == 0:
                    print(f"step {step+1}/{len(train_dl)}, {total_loss=}")

                with torch.no_grad():
                    unwrap_model(vle.model).logit_scale.clamp_(0, math.log(100))

        print(f"Epoch {epoch+1}/{CONFIG['vle']['train']['num_epochs']}, {total_loss=}")

def main() -> None:

    VLE_CONFIG = CONFIG['vle']

    logger = get_logger(VLE_CONFIG['train']['log_dir_path'], VLE_CONFIG['train']['exp_name'])
    # tb_writer = get_tb_logger(CONFIG["tb_dir"], CONFIG["exp_name"])

    # Datasets
    train_image_text_ds = ImageCaptionDataset(
        ImageDataset(Path('/home/olivieri/exp/data/data_gen/VOC2012/train_no_aug/images')),
        JSONLDataset(Path('/home/olivieri/exp/data/data_gen/VOC2012/train_no_aug/captions.jsonl'))
    )
    val_image_text_ds = ImageCaptionDataset(
        ImageDataset(Path('/home/olivieri/exp/data/data_gen/VOC2012/val_no_aug/images')),
        JSONLDataset(Path('/home/olivieri/exp/data/data_gen/VOC2012/val_no_aug/captions.jsonl'))
    )

    # Vision-Language Encoder
    vle: VLEncoder = VLE_REGISTRY.get("flair", device=CONFIG['device'])
    # vle.set_vision_trainable_params('visual_proj')
    vle.set_vision_trainable_params('proj+visual_proj')
    layer_numel_str = get_layer_numel_str(vle.model, print_only_total=False, only_trainable=True).split('\n')

    # Log trainable parameters
    log_title(logger, "Trainable Params")
    [logger.info(t) for t in layer_numel_str]

    # DataLoaders
    train_collate_fn = partial(
        crop_image_preprocess_image_text_batch,
        crop_fn=T.CenterCrop(VLE_CONFIG['image_size']),
        preprocess_images_fn=vle.preprocess_images,
        preprocess_texts_fn=vle.preprocess_texts
    )
    val_collate_fn = partial(
        crop_image_preprocess_image_text_batch,
        crop_fn=T.CenterCrop(VLE_CONFIG['image_size']),
        preprocess_images_fn=vle.preprocess_images,
        preprocess_texts_fn=vle.preprocess_texts
    )

    train_image_text_dl = DataLoader(
        train_image_text_ds,
        batch_size=VLE_CONFIG['train']["batch_size"],
        shuffle=False,
        generator=TORCH_GEN.clone_state(),
        collate_fn=train_collate_fn,
    )
    
    val_image_text_dl = DataLoader(
        val_image_text_ds,
        batch_size=VLE_CONFIG['train']["batch_size"],
        shuffle=False,
        generator=TORCH_GEN.clone_state(),
        collate_fn=val_collate_fn,
    )

    # TODO investigate what this rank and world_size is.
    criterion = vle.create_loss(
        add_mps_loss = True,
        rank = 0,
        world_size = 1,
        num_caps_per_img = 1
    )

    log_intro(
        logger=logger,
        exp_name=VLE_CONFIG['train']['exp_name'],
        exp_desc=VLE_CONFIG['train']['exp_desc'],
        config=CONFIG,
        train_ds=train_image_text_ds,
        val_ds=val_image_text_ds,
        train_dl=train_image_text_dl,
        val_dl=val_image_text_dl
    )

    log_title(logger, "Training Start")

    try:
        train_loop(
            vle,
            train_image_text_dl,
            val_image_text_dl,
            criterion,
    )
    except KeyboardInterrupt:
        log_title(logger, "Training Interrupted")

if __name__ == '__main__':
    main()
