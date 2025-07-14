from config import *
from data import JSONLDataset, DiffMaskDataset, DiffImageTextDataset, CLASS_MAP, get_image_UIDs, crop_image_preprocess_image_text_batch
from path import SPLITS_PATH
from models.vl_encoders import VLE_REGISTRY, VLEncoder
from viz import get_layer_numel_str
from utils import compile_torch_model
from logger import get_logger, log_intro, log_title, log_scores

from torch import nn
from torchvision.models import segmentation as segmodels
from functools import partial
from torchvision.transforms._presets import SemanticSegmentation
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from collections import OrderedDict
import math

from typing import Callable

from vendors.flair.src.flair.train import backward

VLE_CONFIG = CONFIG['vle']
VLE_TRAIN_CONFIG = VLE_CONFIG['train']

def train_loop(
        vle: VLEncoder,
        train_dl: DataLoader,
        val_dl: DataLoader,
        criterion: Callable,
) -> torch.Tensor:
    
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

    vle.model = compile_torch_model(vle.model) # effective only with GPUs with compute capability >= 7.0

    val_loss = vle.evaluate(val_dl, criterion)
    log_scores(f"Before any weight update, VALIDATION", val_loss, None, 0, "val", None, "val_")
    
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
            total_loss = sum(losses.values()) # in our case, we only have losses has only the 'constrastive loss' key.

            scaler = None # for AMP
            backward(total_loss, scaler)

            grad_clip_norm = None
            grad_norm = None
            if grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(vle.model.parameters(), grad_clip_norm, norm_type=2.0)
            
            optimizer.step()

            if (step+1) % CONFIG['vle']['train']['log_every'] == 0:
                # print(f"step {step+1}/{len(train_dl)}, {total_loss=}")
                log_scores(f"epoch: {epoch+1}/{VLE_TRAIN_CONFIG['num_epochs']}, step: {step+1}/{len(train_dl)}", total_loss, None, None, "train", f", lr: {lr:.2e}, grad_norm: {grad_norm:.2f}" ,"batch_")

            with torch.no_grad():
                vle.model.logit_scale.clamp_(0, math.log(100))

            torch.cuda.synchronize() if CONFIG["device"] == "cuda" else None
            
        val_loss = vle.evaluate(val_dl, criterion)

        log_scores(f"epoch: {epoch+1}/{CONFIG['seg']['num_epochs']}, VALIDATION", val_loss, None, epoch+1, "val", None,"val_")

    return val_loss

def main() -> None:

    logger = get_logger(VLE_CONFIG['train']['log_dir_path'], VLE_CONFIG['train']['exp_name'])
    # tb_writer = get_tb_logger(CONFIG["tb_dir"], CONFIG["exp_name"])

    # Datasets
    train_image_text_ds = DiffImageTextDataset(
        DiffMaskDataset(Path('/home/olivieri/exp/data/data_gen/VOC2012/train_no_aug/images')),
        JSONLDataset(Path('/home/olivieri/exp/data/data_gen/VOC2012/train_no_aug/captions.jsonl'))
    )
    val_image_text_ds = DiffImageTextDataset(
        DiffMaskDataset(Path('/home/olivieri/exp/data/data_gen/VOC2012/val_no_aug/images')),
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
        generator=get_torch_gen(),
        collate_fn=train_collate_fn,
    )
    
    val_image_text_dl = DataLoader(
        val_image_text_ds,
        batch_size=VLE_CONFIG['train']["batch_size"],
        shuffle=False,
        generator=get_torch_gen(),
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
        val_loss = train_loop(
            vle,
            train_image_text_dl,
            val_image_text_dl,
            criterion,
    )
    except KeyboardInterrupt:
        log_title(logger, "Training Interrupted")

    log_title("Training Finished")
    
    # final train. logs to file
    train_loss, train_metrics_score = vle.evaluate(train_image_text_dl, criterion, None)
    log_scores(f"After {VLE_TRAIN_CONFIG['num_epochs']} epochs of training, TRAINING", train_loss, train_metrics_score, None, "train", None, "train_")
    log_scores(f"After {VLE_TRAIN_CONFIG['num_epochs']} epochs of training, VALIDATION", val_loss, None, None, None, None, "val_")

    # Save the weights

if __name__ == '__main__':
    main()
