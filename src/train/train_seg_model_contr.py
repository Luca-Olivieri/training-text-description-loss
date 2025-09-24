from config import *
from data import VOC2012SegDataset, crop_augment_preprocess_batch, apply_classmap
from models.seg_models import SegModelWrapper, SEGMODELS_REGISTRY
from models.vl_models import GenParams, OllamaMLLM
from models.vl_encoders import VLE_REGISTRY, VLEncoder
from loss import SigLipLossMultiText
from prompter import FastPromptBuilder
from logger import LogManager
from path import get_mask_prs_path
from viz import get_layer_numel_str
from utils import clear_memory, compile_torch_model, subsample_sign_classes

from functools import partial
from collections import OrderedDict
from torch import nn
from torch.utils.data import DataLoader
from open_clip_train.precision import get_autocast # for AMP
from torch.amp import GradScaler # for AMP
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from torchvision.transforms._presets import SemanticSegmentation
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
import torchmetrics as tm
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from vendors.flair.src.flair.train import backward
import math

import asyncio

from typing import Optional, Callable
from torch.nn.modules.loss import _Loss

SEG_CONFIG = CONFIG['seg']
SEG_TRAIN_CONFIG = SEG_CONFIG['train']
SEG_WITH_TEXT_CONFIG = SEG_TRAIN_CONFIG['with_text']
SEG_CONTR_CONFIG = SEG_WITH_TEXT_CONFIG['contr']

VLE_CONFIG = CONFIG['vle']
VLE_TRAIN_CONFIG = VLE_CONFIG['train']

if SEG_TRAIN_CONFIG['log_only_to_stdout']:
    log_manager = LogManager(
        exp_name=SEG_TRAIN_CONFIG['exp_name'],
        exp_desc=SEG_TRAIN_CONFIG['exp_desc'],
    )
else:
    log_manager = LogManager(
        exp_name=SEG_TRAIN_CONFIG['exp_name'],
        exp_desc=SEG_TRAIN_CONFIG['exp_desc'],
        file_logs_dir_path=SEG_TRAIN_CONFIG['file_logs_dir_path'],
        tb_logs_dir_path=SEG_TRAIN_CONFIG['tb_logs_dir_path']
    )

async def train_loop(
        segmodel: SegModelWrapper,
        vlm: OllamaMLLM,
        vle: VLEncoder,
        train_dl: DataLoader,
        val_dl: DataLoader,
        fast_prompt_builder: FastPromptBuilder,
        seg_preprocess_fn: nn.Module,
        gen_params: GenParams,
        criterion: _Loss,
        aux_criterion: nn.Module,
        metrics_dict: dict[dict, tm.Metric],
        checkpoint_dict: Optional[dict] = None,
        sign_classes_filter: Optional[Callable[[list[int]], list[int]]] = None,
) -> None:
    
    # --- 1. Initialization and State Restoration ---
    start_epoch = 0
    global_step = 0
    if checkpoint_dict:
        start_epoch = checkpoint_dict['epoch'] + 1
        global_step = checkpoint_dict['global_step']
        log_manager.log_line(f"Resuming training from epoch {start_epoch}, global step {global_step}.")

    grad_accum_steps = SEG_TRAIN_CONFIG['grad_accum_steps']
    num_batches_per_epoch = len(train_dl)
    # The number of optimizer steps per epoch
    num_steps_per_epoch = math.ceil(num_batches_per_epoch / grad_accum_steps)

    # --- 2. Optimizer Setup ---
    lr=SEG_TRAIN_CONFIG['lr_schedule']['base_lr']
    optimizer = torch.optim.AdamW(segmodel.model.parameters(), lr=lr)
    if checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

    # --- 3. Scheduler Setup ---
    total_steps = num_steps_per_epoch * SEG_TRAIN_CONFIG['num_epochs']
    sched_config = SEG_TRAIN_CONFIG['lr_schedule']
    
    scheduler = None
    if sched_config['policy'] == 'const':
        scheduler = const_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)
    elif sched_config['policy'] == 'const-cooldown':
        cooldown_steps = num_steps_per_epoch * sched_config['epochs_cooldown']
        scheduler = const_lr_cooldown(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps, cooldown_steps, sched_config['lr_cooldown_power'], sched_config['lr_cooldown_end'])
    elif sched_config['policy'] == 'cosine':
        scheduler = cosine_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)

    # --- 4. AMP and Model Compilation Setup ---
    autocast = get_autocast(SEG_TRAIN_CONFIG['precision'])
    scaler = GradScaler() if SEG_TRAIN_CONFIG['precision'] == "amp" else None
    if scaler and checkpoint_dict:
        scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])

    # --- 5. Initial Validation ---
    log_manager.log_title("Initial Validation")
    val_loss, val_metrics_score = segmodel.evaluate(val_dl, criterion, metrics_dict)
    log_manager.log_scores(f"Before any weight update, VALIDATION", val_loss, val_metrics_score, start_epoch, "val", None, "val_")
    best_val_mIoU = val_metrics_score['mIoU']

    log_manager.log_title("Training Start")
    
    # --- 6. Main Training Loop ---
    train_metrics = tm.MetricCollection(metrics_dict)
    for epoch in range(start_epoch, SEG_TRAIN_CONFIG["num_epochs"]):
        
        train_metrics.reset() # in theory, this can be removed
        segmodel.model.train()

        accum_img_features, accum_cs_global_text_tokens = [], []
        seg_batch_loss = None
        cs_counter = 0

        for step, (scs_img, gts) in enumerate(train_dl):

            # --- Seg --- #

            scs = seg_preprocess_fn(scs_img)

            scs = scs.to(CONFIG["device"])
            gts = gts.to(CONFIG["device"]) # shape [B, H, W]
            
            with autocast():
                logits = segmodel.model(scs)
                logits: torch.Tensor = logits["out"] if isinstance(logits, OrderedDict) else logits # shape [N, C, H, W]
            
                if seg_batch_loss is None:
                    seg_batch_loss: torch.Tensor = criterion(logits, gts) / grad_accum_steps
                else:
                    seg_batch_loss += criterion(logits, gts) / grad_accum_steps

            train_metrics.update(logits.detach().argmax(dim=1), gts)

            if SEG_WITH_TEXT_CONFIG['with_text']:

                # --- VLM --- #

                scs_img = (scs_img*255).to(torch.uint8)
                gts = gts.unsqueeze(1)
                prs = logits.argmax(dim=1, keepdim=True)
                # Both VLM and VLE receive the images in the same downsampled size.
                gts_down = TF.resize(gts, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
                prs_down = TF.resize(prs, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
                scs_down = TF.resize(scs_img, fast_prompt_builder.image_size, TF.InterpolationMode.BILINEAR)
                cs_prompts = fast_prompt_builder.build_cs_inference_prompts(apply_classmap(gts_down, fast_prompt_builder.class_map), apply_classmap(prs_down, fast_prompt_builder.class_map), scs_down, sign_classes_filter)

                batch_idxs = [train_dl.batch_size*step + i for i in range(len(scs_down))]
                
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

                # --- VLE --- #

                with autocast():

                    # aggregate the class-splitted global text tokens
                    for i, img_idx in enumerate(batch_idxs):
                        cs_texts = list(cs_answer_list[i]['content'].values()) # gather the text for each pos. class of this image
                        cs_texts = vle.preprocess_texts(cs_texts)
                        cs_vle_output = vle.encode_and_project(images=None, texts=cs_texts, broadcast=False)

                        cs_global_text_token = cs_vle_output.global_text_token # [N_cs, D]
                        cs_counter += len(cs_global_text_token)

                        # aggr_global_text_token = global_text_token.mean(dim=0) # aggregating the class-splitted text vectors by averaging.
                        accum_cs_global_text_tokens.append(cs_global_text_token)
                    # global_text_tokens = torch.stack(global_text_tokens)
                    # global_text_tokens = segmodel.model.bottleneck_adapter.mlp(global_text_tokens)
                    
                    bottleneck_out: torch.Tensor = segmodel.activations['bottleneck']
                    bottleneck_out = segmodel.adapt_tensor(bottleneck_out)
                    accum_img_features.append(bottleneck_out)

            # del scs_img, scs, gts, prs, logits, bottleneck_out, global_text_tokens

            # seg_batch_loss.backward()
            
            is_last_batch = (step + 1) == num_batches_per_epoch
            is_accum_step = (step + 1) % grad_accum_steps == 0

            # --- Optimizer Step and Scheduler Update ---
            if is_accum_step or is_last_batch:

                if SEG_WITH_TEXT_CONFIG['with_text']:

                    bottleneck_out = torch.concat(accum_img_features, dim=0)

                    with autocast():

                        aux_batch_loss = torch.tensor(0.0, device=CONFIG['device'])
                        for i, cs_global_text_token in enumerate(accum_cs_global_text_tokens):

                            cs_global_text_token = segmodel.model.bottleneck_adapter.mlp(cs_global_text_token)

                            aux_batch_loss += aux_criterion(
                                image_features=bottleneck_out,
                                text_features=cs_global_text_token,
                                positive_image_idx=i,
                                logit_scale=vle.model.logit_scale/vle.model.logit_scale,
                                logit_bias=vle.model.logit_bias*0,
                                output_dict=False
                            )/(len(cs_global_text_token)*train_dl.batch_size*grad_accum_steps)

                        batch_loss = seg_batch_loss*(1. - SEG_CONTR_CONFIG['loss_lam']) + aux_batch_loss*SEG_CONTR_CONFIG['loss_lam'] # lambda coefficient
                        
                    cs_mult = cs_counter/(train_dl.batch_size*grad_accum_steps)
                else:
                    aux_batch_loss = torch.tensor(-1.0, device=CONFIG['device'])
                    batch_loss = seg_batch_loss
                    cs_mult = -1.0

                # batch_loss.backward()
                backward(batch_loss, scaler)

                if scheduler:
                    scheduler(global_step)

                max_grad_norm = SEG_TRAIN_CONFIG['grad_clip_norm']
                if scaler:
                    if SEG_TRAIN_CONFIG['grad_clip_norm']:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            segmodel.model.parameters(),
                            max_grad_norm if max_grad_norm else float('inf'),
                            norm_type=2.0
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if SEG_TRAIN_CONFIG['grad_clip_norm']:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            segmodel.model.parameters(),
                            max_grad_norm if max_grad_norm else float('inf'),
                            norm_type=2.0
                        )
                    optimizer.step()

                #grad_norm = torch.nn.utils.clip_grad_norm_(
                #    segmodel.model.parameters(),
                #    max_grad_norm if max_grad_norm else float('inf'),
                #    norm_type=2.0
                #)
                
                # optimizer.step()
                optimizer.zero_grad()

                global_step += 1 # Increment global step *only* after an optimizer step

                # --- Logging ---
                if global_step % SEG_TRAIN_CONFIG['log_every'] == 0:
                    train_metrics_score = train_metrics.compute()                    
                    train_metrics_score = {'aux_loss': aux_batch_loss} | train_metrics_score # add the aux. loss to the logged metrics
                    current_lr = optimizer.param_groups[0]['lr']
                    step_in_epoch = (step // grad_accum_steps) + 1
                    log_manager.log_scores(
                        f"epoch: {epoch+1}/{SEG_TRAIN_CONFIG['num_epochs']}, step: {step_in_epoch}/{num_steps_per_epoch} (global_step: {global_step})",
                        seg_batch_loss, train_metrics_score, global_step, "train",
                        f", lr: {current_lr:.2e}, grad_norm: {grad_norm:.2f}, cs_mult: {cs_mult:.2f}", "batch_"
                    )
                
                # reset accumulated values
                if SEG_WITH_TEXT_CONFIG['with_text']:
                    accum_img_features, accum_cs_global_text_tokens = [], []
                seg_batch_loss = None
                cs_counter = 0

                train_metrics.reset() # only the batch metrics are logged

        # --- End of Epoch Validation and Checkpointing ---
        val_loss, val_metrics_score = segmodel.evaluate(val_dl, criterion, metrics_dict)
        log_manager.log_scores(f"epoch: {epoch+1}/{SEG_TRAIN_CONFIG['num_epochs']}, VALIDATION", val_loss, val_metrics_score, epoch+1, "val", None, "val_")

        if val_metrics_score['mIoU'] > best_val_mIoU:
            best_val_mIoU = val_metrics_score['mIoU']
            
            if SEG_TRAIN_CONFIG['save_weights_root_path']:
                # Note: 'epoch' is saved, so on resume we start from 'epoch + 1'
                new_checkpoint_dict = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': segmodel.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if scaler:
                    new_checkpoint_dict['scaler_state_dict'] = scaler.state_dict()

                save_dir = Path(SEG_TRAIN_CONFIG['save_weights_root_path'])
                save_dir.mkdir(parents=True, exist_ok=True)
                ckp_filename = f"lraspp_mobilenet_v3_large_{SEG_TRAIN_CONFIG['exp_name']}.pth"
                full_ckp_path = save_dir / ckp_filename
                torch.save(new_checkpoint_dict, full_ckp_path)
                log_manager.log_line(f"New best model saved to {full_ckp_path} with validation mIoU: {best_val_mIoU:.4f}")
    
    log_manager.log_title("Training Finished")

async def main() -> None:
    train_ds = VOC2012SegDataset(
        root_path=Path("/home/olivieri/exp/data/VOCdevkit"),
        split='train',
        resize_size=SEG_CONFIG['image_size'],
        center_crop=False,
        with_unlabelled=True,
    )

    val_ds = VOC2012SegDataset(
        root_path=Path("/home/olivieri/exp/data/VOCdevkit"),
        split='val',
        resize_size=SEG_CONFIG['image_size'],
        center_crop=False,
        with_unlabelled=True,
    )

    # Segmentation Model
    segmodel: SegModelWrapper = SEGMODELS_REGISTRY.get(
        'lraspp_mobilenet_v3_large',
        pretrained_weights_path=Path(SEG_CONFIG['pretrained_weights_path']),
        adaptation='contrastive_global',
        device=CONFIG['device']
    ) # TODO to modify with the actual segnet intermediate checkpoint

    segmodel.adapt()

    if True:
        state_dict: OrderedDict = torch.load('/home/olivieri/exp/data/torch_weights/seg/lraspp_mobilenet_v3_large/with_text/synth_contr/lraspp_mobilenet_v3_large_phase_2_synth_text_e5_250903_1711.pth')
        model_state_dict = state_dict.get('model_state_dict', state_dict)
        segmodel.model.load_state_dict(model_state_dict)

    segmodel.set_trainable_params(train_decoder_only=False)

    checkpoint_dict = None
    if SEG_TRAIN_CONFIG['resume_path']:
        seg_weights_path = Path(SEG_TRAIN_CONFIG['resume_path'])
        if seg_weights_path.exists():
            checkpoint_dict = torch.load(seg_weights_path, map_location=CONFIG['device'])
            segmodel.model.load_state_dict(checkpoint_dict['model_state_dict'])
        else:
            raise AttributeError(f"ERROR: Resume path '{seg_weights_path}' not found. ")

    # Vision-Language Model
    model_name = "gemma3:12b-it-qat"
    vlm = OllamaMLLM(model_name, container_name=CONFIG['ollama_container_name'])

    by_model = "LRASPP_MobileNet_V3"

    gen_params = GenParams(
        seed=CONFIG["seed"],
        temperature=SEG_WITH_TEXT_CONFIG['vlm_temperature']
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

    # NOTE when used in this pipeline, the dataset is useful only to access class maps and color maps, the actual data is not retrieved from here.
    seg_dataset = VOC2012SegDataset(
        root_path=Path(CONFIG['datasets']['VOC2012_root_path']),
        split='train',
        resize_size=CONFIG['seg']['image_size'],
        center_crop=True,
        with_unlabelled=False,
    )

    sup_set_seg_dataset = VOC2012SegDataset(
        root_path=Path(CONFIG['datasets']['VOC2012_root_path']),
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

    # Vision-Language Encoder
    vle: VLEncoder = VLE_REGISTRY.get("flair", version='flair-cc3m-recap.pt', device=CONFIG['device'], vision_adapter=False, text_adapter=False)
    vle_weights_path = Path(SEG_WITH_TEXT_CONFIG['vle_weights_path'])
    if vle_weights_path.exists():
        vle.model.load_state_dict(torch.load(vle_weights_path, map_location=CONFIG['device'])['model_state_dict'])
    else:
        raise AttributeError(f"ERROR: VLE weights path '{vle_weights_path}' not found.")
    
    vle.set_vision_trainable_params(None)

    # NOTE deleting vision layers only if encoding text only.
    del vle.model.visual, vle.model.visual_proj, vle.model.image_post
    clear_memory()
    vle.model = compile_torch_model(vle.model)

    seg_preprocess_fn = partial(SemanticSegmentation, resize_size=SEG_CONFIG['image_size'])() # same as original one, but with custom resizing

    # training cropping functions
    center_crop_fn = T.CenterCrop(SEG_CONFIG['image_size'])
    random_crop_fn = T.RandomCrop(SEG_CONFIG['image_size'])

    # augmentations
    augment_fn = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        # T.RandomAffine(degrees=0, scale=(0.5, 2)), # Zooms in and out of the image.
        # T.RandomAffine(degrees=[-30, 30], translate=[0.2, 0.2], scale=(0.5, 2), shear=15), # Full affine transform.
        # T.RandomPerspective(p=0.5, distortion_scale=0.2) # Shears the image
    ])

    train_collate_fn = partial(
        crop_augment_preprocess_batch,
        crop_fn=random_crop_fn,
        augment_fn=augment_fn,
        preprocess_fn=None
    )

    val_collate_fn = partial(
        crop_augment_preprocess_batch,
        crop_fn=T.CenterCrop(SEG_CONFIG['image_size']),
        augment_fn=None,
        preprocess_fn=seg_preprocess_fn
    )

    criterion = nn.CrossEntropyLoss(ignore_index=21)
    # aux_criterion = SigLipLoss()
    aux_criterion = SigLipLossMultiText()

    # sign_classes_filter = partial(subsample_sign_classes, k=0)
    sign_classes_filter = None

    train_dl = DataLoader(
        train_ds,
        batch_size=SEG_TRAIN_CONFIG["batch_size"],
        shuffle=True,
        generator=get_torch_gen(),
        collate_fn=train_collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=SEG_TRAIN_CONFIG["batch_size"],
        shuffle=False,
        generator=get_torch_gen(),
        collate_fn=val_collate_fn,
    )

    metrics_dict = {
        "acc": MulticlassAccuracy(num_classes=train_ds.get_num_classes(with_unlabelled=True), top_k=1, average="micro", multidim_average="global", ignore_index=21).to(CONFIG["device"]),
        "mIoU": MulticlassJaccardIndex(num_classes=train_ds.get_num_classes(with_unlabelled=True), average="macro", ignore_index=21).to(CONFIG["device"]),
    }

    log_manager.log_intro(
        config=CONFIG,
        train_ds=train_ds,
        val_ds=val_ds,
        train_dl=train_dl,
        val_dl=val_dl
    )

    # Log trainable parameters
    log_manager.log_title("Trainable Params")
    [log_manager.log_line(t) for t in get_layer_numel_str(segmodel.model, print_only_total=False, only_trainable=True).split('\n')]

    try:
        await train_loop(
            segmodel,
            vlm,
            vle,
            train_dl,
            val_dl,
            fast_prompt_builder,
            seg_preprocess_fn,
            gen_params,
            criterion,
            aux_criterion,
            metrics_dict,
            checkpoint_dict,
            sign_classes_filter
    )
    except KeyboardInterrupt:
        log_manager.log_title("Training Interrupted", pad_symbol='~')

    segmodel.remove_handles()

    log_manager.close_loggers()


if __name__ == '__main__':
    asyncio.run(main())
