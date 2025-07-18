from config import *
from data import JSONLDataset, ImageDataset, ImageCaptionDataset, crop_image_preprocess_image_text_batch
from models.vl_encoders import VLE_REGISTRY, VLEncoder
from viz import get_layer_numel_str
from utils import compile_torch_model
from logger import LogManager

from torch import nn
from functools import partial
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import math
import torchmetrics as tm
from open_clip_train.precision import get_autocast # for AMP
from torch.amp import GradScaler # for AMP

from typing import Callable, Optional

from vendors.flair.src.flair.train import backward

VLE_CONFIG = CONFIG['seg']
VLE_TRAIN_CONFIG = VLE_CONFIG['train']

if VLE_TRAIN_CONFIG['log_only_to_stdout']:
    log_manager = LogManager(
        exp_name=VLE_TRAIN_CONFIG['exp_name']
    )
else:
    log_manager = LogManager(
        exp_name=VLE_TRAIN_CONFIG['exp_name'],
        file_logs_dir_path=VLE_TRAIN_CONFIG['file_logs_dir_path'],
        tb_logs_dir_path=VLE_TRAIN_CONFIG['tb_logs_dir_path']
    )

def train_loop(
        vle: VLEncoder,
        train_dl: DataLoader,
        val_dl: DataLoader,
        criterion: Callable,
        metrics_dict: dict[dict, tm.Metric],
        checkpoint_dict: Optional[dict] = None
) -> torch.Tensor:

    # optimizer
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(vle.model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": 1e-2}, # authors use 0.5, seems to high
        ],
        lr=5e-4,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    optimizer = torch.optim.AdamW(vle.model.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict']) if VLE_TRAIN_CONFIG['resume'] else None
    
    # AMP scaler
    scaler = GradScaler() if VLE_TRAIN_CONFIG['precision'] == "amp" else None
    scaler.load_state_dict(checkpoint_dict["scaler_state_dict"]) if (VLE_TRAIN_CONFIG['resume'] and VLE_TRAIN_CONFIG['precision'] == "amp") else None

    vle.model = compile_torch_model(vle.model) # effective only with GPUs with compute capability >= 7.0

    log_manager.log_title("Initial Validation")
    val_loss, val_metrics_score = vle.evaluate(val_dl, criterion, metrics_dict)
    log_manager.log_scores(f"Before any weight update, VALIDATION", val_loss, val_metrics_score, 0, "val", None, "val_")
    log_manager.log_title("Training Start")

    autocast = get_autocast(VLE_TRAIN_CONFIG['precision'])
    
    for epoch in range(VLE_TRAIN_CONFIG["num_epochs"]):

        global_epoch = checkpoint_dict['global_epoch'] + epoch if VLE_TRAIN_CONFIG['resume'] else epoch

        for step, (images, texts) in enumerate(train_dl):
            
            vle.model.train()

            optimizer.zero_grad()

            with autocast():
                vle_output = vle.encode_and_project(images, texts, broadcast=False)

                losses = criterion(
                        image_features=vle_output.global_image_token,
                        image_tokens=vle_output.local_image_tokens.clone(),
                        text_features=vle_output.global_text_token.squeeze(1),
                        logit_scale=vle.model.logit_scale.exp(),
                        visual_proj=vle.model.visual_proj,
                        logit_bias=vle.model.logit_bias,
                        output_dict=True
                )
                total_loss = sum(losses.values()) # in our case, we only have losses has only the 'constrastive loss' key.

            backward(total_loss, scaler)

            grad_clip_norm = 5
            grad_norm = 0.0
            if grad_clip_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(vle.model.parameters(), grad_clip_norm, norm_type=2.0)
            
            optimizer.step()

            step_counter = epoch*len(train_dl) + step + 1
            global_step_counter = checkpoint_dict['global_step_counter'] + step_counter if VLE_TRAIN_CONFIG['resume'] else step_counter

            if (step+1) % CONFIG['vle']['train']['log_every'] == 0:
                log_manager.log_scores(f"epoch: {epoch+1}/{VLE_TRAIN_CONFIG['num_epochs']} ({global_epoch}), step: {step+1}/{len(train_dl)} ({global_step_counter})", total_loss, train_metrics_score, global_step_counter, "train", f", lr: {optimizer.param_groups[0]['lr']:.2e}, grad_norm: {grad_norm:.2f}" ,"batch_")

            with torch.no_grad():
                vle.model.logit_scale.clamp_(0, math.log(100))

            torch.cuda.synchronize() if CONFIG["device"] == "cuda" else None
            
        val_loss = vle.evaluate(val_dl, criterion)

        log_manager.log_scores(f"epoch: {epoch+1}/{VLE_TRAIN_CONFIG['num_epochs']} ({global_epoch}), VALIDATION", val_loss, val_metrics_score, global_epoch+1, "val", None,"val_")

    log_manager.log_title("Training Finished")

    # checkpoint saving
    if VLE_TRAIN_CONFIG['save_weights']:    
        new_checkpoint_dict = {
                'global_epoch': global_epoch,
                'global_step': global_step_counter,
                'model_state_dict': vle.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
        }
        new_checkpoint_dict |= {'scaler_state_dict': scaler.state_dict()} if scaler else None
        # 'scheduler_state_dict': scheduler.state_dict(),

        ckp_filename = f"flair-{vle.checkpoint}-{VLE_TRAIN_CONFIG['exp_name']}"
        torch.save(vle.model.state_dict(), TORCH_WEIGHTS_CHECKPOINTS / 'vle' / f"{ckp_filename}.pth")
        log_manager.log_line(f"Model '{ckp_filename}.pth' successfully saved.")

    # final logs
    log_manager.log_title("Final Evaluation")

    train_loss, train_metrics_score = vle.evaluate(train_dl, criterion)
    log_manager.log_scores(f"After {VLE_TRAIN_CONFIG['num_epochs']} epochs of training, TRAINING", train_loss, train_metrics_score, step_counter, "train", None, "train_")
    log_manager.log_scores(f"After {VLE_TRAIN_CONFIG['num_epochs']} epochs of training, VALIDATION", val_loss, val_metrics_score, None, None, None, "val_")


def main() -> None:

    mask_color: str = VLE_TRAIN_CONFIG['mask_color'] # can be 'RB' or 'L'

    # Datasets
    train_image_text_ds = ImageCaptionDataset(
        ImageDataset(Path(f'/home/olivieri/exp/data/data_gen/VOC2012/train/images_{mask_color}')),
        JSONLDataset(Path('/home/olivieri/exp/data/data_gen/VOC2012/train/captions.jsonl'))
    )
    val_image_text_ds = ImageCaptionDataset(
        ImageDataset(Path(f'/home/olivieri/exp/data/data_gen/VOC2012/train/images_{mask_color}')),
        JSONLDataset(Path('/home/olivieri/exp/data/data_gen/VOC2012/train/captions.jsonl'))
    )

    # Vision-Language Encoder
    vle: VLEncoder = VLE_REGISTRY.get("flair", device=CONFIG['device'], vision_adapter=True)

    checkpoint_dict = torch.load(TORCH_WEIGHTS_CHECKPOINTS / 'vle' / (... + ".pth"))
    # vle.model.load_state_dict(checkpoint_dict['model_state_dict']) # load my weights
    vle.set_vision_trainable_params(['proj', 'visual_proj', 'vision_adapter'])

    # TODO try to set the weights of 'vision_adapter' to 0 and see what happens.

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
        batch_size=VLE_TRAIN_CONFIG["batch_size"],
        shuffle=False,
        generator=get_torch_gen(),
        collate_fn=train_collate_fn,
    )
    
    val_image_text_dl = DataLoader(
        val_image_text_ds,
        batch_size=VLE_TRAIN_CONFIG["batch_size"],
        shuffle=False,
        generator=get_torch_gen(),
        collate_fn=val_collate_fn,
    )

    criterion = vle.create_loss(
        add_mps_loss=True, # as suggested by the authors (depending by the trainable params, this might not affect the training)
        num_caps_per_img=1
    )

    log_manager.log_intro(
        exp_name=VLE_TRAIN_CONFIG['exp_name'],
        exp_desc=VLE_TRAIN_CONFIG['exp_desc'],
        config=CONFIG,
        train_ds=train_image_text_ds,
        val_ds=val_image_text_ds,
        train_dl=train_image_text_dl,
        val_dl=val_image_text_dl
    )
    
    # Log trainable parameters
    log_manager.log_title("Trainable Params")
    [log_manager.log_line(t) for t in get_layer_numel_str(vle.model, print_only_total=False, only_trainable=True).split('\n')]

    try:
        train_loop(
            vle,
            train_image_text_dl,
            val_image_text_dl,
            criterion,
            checkpoint_dict
        )
    except KeyboardInterrupt:
        log_manager.log_title("Training Interrupted", pad_symbol='~')

    log_manager.close_loggers()

if __name__ == '__main__':
    main()
