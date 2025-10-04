from core.config import *
from data import VOC2012SegDataset, crop_augment_preprocess_batch, apply_classmap
from models.seg import SegModelWrapper, SEGMODELS_REGISTRY
from models.mllm import GenParams, OllamaMLLMAdapter
from models.vle import VLE_REGISTRY, VLEncoder
from train.loss import GroupedPairedNegativeSigLipLoss
from core.prompter import FastPromptBuilder
from core.logger import LogManager
from core.path import get_mask_prs_path
from core.viz import get_layer_numel_str, create_cs_ovr_masks
from core.utils import clear_memory, compile_torch_model, subsample_sign_classes, nanstd, flatten_tensor_list, unflatten_tensor_list, flatten_list_of_lists, unflatten_list_of_lists, NegativeTextGenerator, diff_text_word_pools
from cache.cache import Cache, PercentilePolicy, MaskTextCache, Identity

from functools import partial
from collections import OrderedDict
from torch import nn
from torch.utils.data import DataLoader
from open_clip_train.precision import get_autocast # for AMP
from torch.amp import GradScaler # for AMP
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from torchvision.transforms._presets import SemanticSegmentation
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex, BinaryJaccardIndex
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
        vlm: OllamaMLLMAdapter,
        vle: VLEncoder,
        train_dl: DataLoader,
        val_dl: DataLoader,
        fast_prompt_builder: FastPromptBuilder,
        seg_preprocess_fn: nn.Module,
        gen_params: GenParams,
        criterion: _Loss,
        aux_criterion: GroupedPairedNegativeSigLipLoss,
        metrics_dict: dict[str, tm.Metric],
        mask_text_cache: MaskTextCache,
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

    neg_text_gen = NegativeTextGenerator(diff_text_word_pools)

    # --- 2. Optimizer Setup ---
    lr=SEG_TRAIN_CONFIG['lr_schedule']['base_lr']
    optimizer = torch.optim.AdamW(segmodel.model.parameters(), lr=lr, weight_decay=math.pow(10, -0.5))
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

        accum_img_features, accum_cs_global_text_tokens = [], {}
        seg_batch_loss = None
        cs_counter = 0
        filtered_cs_counter = 0

        print(mask_text_cache)

        for step, (uids, scs_img, gts) in enumerate(train_dl):

            # --- Seg --- #

            scs = seg_preprocess_fn(scs_img)

            scs = scs.to(CONFIG["device"])
            gts = gts.to(CONFIG["device"]) # shape [B, H, W]
            
            with autocast():
                # segmodel.model.eval() # in eval mode, no dropout
                logits = segmodel.model(scs)
                segmodel.model.train()
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

                cs_prompts_by_uid = {uid: cs_p for uid, cs_p in zip(uids, cs_prompts)}
                prs_by_uid = {uid: pr for uid, pr in zip(uids, prs_down)}

                cs_dict: dict[str, list[int]] = {uid: list(cs_p.keys()) for uid, cs_p in zip(uids, cs_prompts)}

                cs_counter += sum([1 for uid, sign_classes in cs_dict.items() for pos_c in sign_classes])

                cs_splitted_bool_masks = {uid: {pos_c: prs_by_uid[uid] == pos_c for pos_c in sign_classes} for uid, sign_classes in cs_dict.items()}
                cs_dict_to_update: dict[str, list[int]] = mask_text_cache.get_cs_keys_to_update(cs_splitted_bool_masks)

                filtered_cs_prompts_dict = {uid: {pos_c: cs_prompts_by_uid[uid][pos_c] for pos_c in sign_classes} for uid, sign_classes in cs_dict_to_update.items()}

                batch_idxs = [train_dl.batch_size*step + i for i in range(len(scs_down))]

                cs_answer_list = await vlm.predict_cs_many(
                    list(filtered_cs_prompts_dict.values()),
                    batch_idxs,
                    gen_params=gen_params,
                    jsonl_save_path=None,
                    only_text=True,
                    splits_in_parallel=False,
                    batch_size=None,
                    use_tqdm=False
                )

                cs_answer_dict: dict[str, dict[int, str]] = {}
                for uid, cs_a in zip(cs_dict_to_update.keys(), cs_answer_list):
                    cs_answer_dict |= {uid: cs_a["content"]}
                
                # images and texts to update
                cs_splitted_bool_masks_to_update = {uid: {pos_c: prs_by_uid[uid] == pos_c for pos_c in sign_classes} for uid, sign_classes in cs_dict_to_update.items()}
                batch = {f"{uid}-{pos_c}": (bool_mask, cs_answer_dict[uid][pos_c]) for uid, cs_bool_mask in cs_splitted_bool_masks_to_update.items() for pos_c, bool_mask in cs_bool_mask.items()}
                
                # --- VLE --- #

                with autocast():

                    uids_to_update = list(cs_dict_to_update.keys())
                    sign_classes_to_update = list(cs_dict_to_update.values())

                    # B is the batch size
                    # L is the number of UIDs involved in the contrastive loss.
                    # M is the number of negative texts for image.
                    # P is the total number of positive pairs (P) involved in the contrastive loss.

                    cs_texts = [list(cs_a.values()) for cs_a in cs_answer_dict.values()]
                    cs_neg_texts = [[neg_text_gen.generate(pos_txt, num_negatives=train_dl.batch_size-1, change_probability=0.5) for pos_txt in cs_txts] for cs_txts in cs_texts]
                    cs_texts = [vle.preprocess_texts(cs_txt) for cs_txt in cs_texts] # list of P tensors (., tokens)
                    cs_neg_texts = torch.stack([torch.stack([vle.preprocess_texts(neg_txts) for neg_txts in cs_txts], dim=0) for cs_txts in cs_neg_texts], dim=0) # (P, M, D_t)
                    
                    flat_cs_texts, cs_text_struct_info = flatten_tensor_list(cs_texts) # (P, tokens)
                    P = len(flat_cs_texts)
                    flat_cs_neg_texts = cs_neg_texts.view(-1, cs_neg_texts.shape[-1]) # (P*M, tokens)
                    flat_cs_concat_texts = torch.cat([flat_cs_texts, flat_cs_neg_texts], dim=0) # (P + P*M, tokens)

                    filtered_scs_down = torch.stack([sc for sc, uid in zip(scs_down, cs_dict.keys()) if uid in uids_to_update])
                    filtered_gts_down = torch.stack([gt for gt, uid in zip(gts_down, cs_dict.keys()) if uid in uids_to_update])
                    cs_ovr_masks_gt = [torch.stack(list(cs_ovr_mask.values())) for cs_ovr_mask in create_cs_ovr_masks(filtered_scs_down, filtered_gts_down.squeeze(1), sign_classes_to_update, alpha=0.55)] # list of n tensors (., 3, H, W)
                    flat_cs_ovr_masks_gt = torch.cat(cs_ovr_masks_gt, dim=0)# (P, 3, H, W)
                    flat_cs_ovr_masks_gt = vle.preprocess_images(flat_cs_ovr_masks_gt/255.)

                    flat_cs_vle_output = vle.encode_and_project(images=flat_cs_ovr_masks_gt, texts=flat_cs_concat_texts, broadcast=False, pool=False)

                    cs_global_text_token = flat_cs_vle_output.global_text_token.squeeze(1) # (P + P*M, D)
                    pos_cs_global_text_token = cs_global_text_token[:P] # (P, D)
                    neg_cs_global_text_token = cs_global_text_token[P:] # (P*M, D)
                    neg_cs_global_text_token = neg_cs_global_text_token.view(cs_neg_texts.shape[0], cs_neg_texts.shape[1], neg_cs_global_text_token.shape[-1])
                    
                    gt_global_image_token = flat_cs_vle_output.global_image_token # (P, D)
                    
                    bottleneck_out: torch.Tensor = segmodel.activations['bottleneck'] # (B, 32, 32, 960)
                    bottleneck_out: torch.Tensor = segmodel.adapt_tensor(bottleneck_out) # (B, 960)
                    pr_global_image_token: torch.Tensor = segmodel.model.bottleneck_adapter.mlp(bottleneck_out) # (B, D)

                    pr_global_image_token = torch.cat([b.unsqueeze(0).expand(len(cs_dict_to_update[uid]), -1) for b, uid in zip(pr_global_image_token, cs_dict.keys()) if uid in uids_to_update]) # (P, D)

                    lhs: torch.Tensor = vle.model.concat_adapter(torch.cat([gt_global_image_token, pr_global_image_token], dim=-1)) # (P, D)
                    rhs = pos_cs_global_text_token # (P, D)

                    image_indices_for_pos_texts = torch.tensor(list(range(len(lhs))), device=rhs.device) # (P,)

                    group_indices_for_pos_pairs = [[i]*len(pos_classes) for i, pos_classes in enumerate(cs_dict_to_update.values())]
                    group_indices_for_pos_pairs, _ = flatten_list_of_lists(group_indices_for_pos_pairs)
                    group_indices_for_pos_pairs = torch.tensor(group_indices_for_pos_pairs, device=rhs.device) # (P,)

            if SEG_WITH_TEXT_CONFIG['with_text']:

                with autocast():

                    aux_batch_loss = aux_criterion(
                        image_features=lhs,
                        positive_text_features=rhs,
                        image_indices_for_pos_texts=image_indices_for_pos_texts,
                        group_indices_for_pos_pairs=group_indices_for_pos_pairs,
                        negative_text_features=neg_cs_global_text_token,
                        logit_scale=vle.model.logit_scale,
                        logit_bias=vle.model.logit_bias,
                        output_dict=False,
                    )

                    batch_loss: torch.Tensor = seg_batch_loss*(1. - SEG_CONTR_CONFIG['loss_lam']) + aux_batch_loss*SEG_CONTR_CONFIG['loss_lam'] # lambda coefficient
                    
                cs_mult = filtered_cs_counter/(train_dl.batch_size*grad_accum_steps)
                filtered_perc = filtered_cs_counter/cs_counter
            else:
                aux_batch_loss = torch.tensor(-1.0, device=CONFIG['device'])
                batch_loss = seg_batch_loss
                cs_mult = -1.0
                filtered_perc = -1.0

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


            optimizer.zero_grad()

            global_step += 1 # Increment global step *only* after an optimizer step

            # --- Logging ---
            if global_step % SEG_TRAIN_CONFIG['log_every'] == 0:
                train_metrics_score = train_metrics.compute()
                if SEG_WITH_TEXT_CONFIG['with_text']:
                    metric_diffs_values = torch.stack(list(mask_text_cache.get_metric_diffs(batch).values()))
                    train_metrics_score |= {'metric_diff_mean': metric_diffs_values.nanmean(), 'metric_diff_std': nanstd(metric_diffs_values, dim=0)}
                train_metrics_score = {'aux_loss': aux_batch_loss} | train_metrics_score
                train_metrics_score |= {"cs_mult": torch.tensor(cs_mult)}
                train_metrics_score |= {"filtered_perc": torch.tensor(filtered_perc)}
                # train_metrics_score |= {"quantile": torch.tensor(mask_text_cache.update_policy.trend.get_current_value())}
                current_lr = optimizer.param_groups[0]['lr']
                step_in_epoch = (step // grad_accum_steps) + 1
                log_manager.log_scores(
                    f"epoch: {epoch+1}/{SEG_TRAIN_CONFIG['num_epochs']}, step: {step_in_epoch}/{num_steps_per_epoch} (global_step: {global_step})",
                    seg_batch_loss, train_metrics_score, global_step, "train",
                    f", lr: {current_lr:.2e}, grad_norm: {grad_norm:.2f}", "batch_"
                )
            
            if SEG_WITH_TEXT_CONFIG['with_text']:
                mask_text_cache.update(batch)
            
            # reset accumulated values
            if SEG_WITH_TEXT_CONFIG['with_text']:
                accum_img_features, accum_cs_global_text_tokens = [], {}
            seg_batch_loss = None
            cs_counter = 0
            filtered_cs_counter = 0

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
        output_uids=True,
        img_idxs=slice(0, 64, 1)
    )

    val_ds = VOC2012SegDataset(
        root_path=Path("/home/olivieri/exp/data/VOCdevkit"),
        split='val',
        resize_size=SEG_CONFIG['image_size'],
        center_crop=False,
        with_unlabelled=True,
        img_idxs=slice(0, 64, 1)
    )

    # Segmentation Model
    segmodel: SegModelWrapper = SEGMODELS_REGISTRY.get(
        'lraspp_mobilenet_v3_large',
        pretrained_weights_path=Path(SEG_CONFIG['pretrained_weights_path']),
        adaptation='contrastive_diff',
        device=CONFIG['device']
    )

    segmodel.adapt()
    
    cache = Cache(storage_device="cpu", memory_device="cuda")
    update_policy = PercentilePolicy(
        metric=BinaryJaccardIndex().to(CONFIG["device"]),
        percentile=0.4,
        trend=Identity()
    )
    mask_text_cache = MaskTextCache(cache, update_policy)

    if False:
        state_dict: OrderedDict = torch.load('/home/olivieri/exp/data/torch_weights/seg/lraspp_mobilenet_v3_large/with_text/phases_cache_contr/lraspp_mobilenet_v3_large_phase_2_synth_text_e5_FIXED_250930_1021.pth')
        model_state_dict = state_dict.get('model_state_dict', state_dict)
        segmodel.model.load_state_dict(model_state_dict)
        mask_text_cache.cache = Cache.load(
            directory_path=Path("/home/olivieri/exp/data/torch_weights/cache/phase_3_prefill_cache_FIXED_250930_1105"),
            storage_device="cpu",
            memory_device="cuda"
        )
    else:
        state_dict: OrderedDict = torch.load('/home/olivieri/exp/data/torch_weights/seg/lraspp_mobilenet_v3_large/no_text/lraspp_mobilenet_v3_large_start_b64_250930_0742.pth')
        model_state_dict = state_dict.get('model_state_dict', state_dict)
        segmodel.model.load_state_dict(model_state_dict, strict=False)
        mask_text_cache.cache = Cache.load(
            directory_path=Path("/home/olivieri/exp/data/torch_weights/cache/phase_3_prefill_cache"), # NOTE wrong cache
            storage_device="cpu",
            memory_device="cuda"
        )

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
    vlm = OllamaMLLMAdapter(model_name, ollama_container_name=CONFIG['ollama_container_name'])

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
        with_unlabelled=False
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
    vle: VLEncoder = VLE_REGISTRY.get("flair", version='flair-cc3m-recap.pt', device=CONFIG['device'], vision_adapter=False, text_adapter=False, concat_adapter=True)
    vle_weights_path = Path(SEG_WITH_TEXT_CONFIG['vle_weights_path'])
    # vle_weights_path = Path('/home/olivieri/exp/data/torch_weights/vle/flair/full_training/flair-flair-cc3m-recap.pt-diff_concat_adapter_gt-pr__t_L_b128_250930_1351.pth')
    if vle_weights_path.exists():
        vle.model.load_state_dict(torch.load(vle_weights_path, map_location=CONFIG['device'])['model_state_dict'], strict=False)
    else:
        raise AttributeError(f"ERROR: VLE weights path '{vle_weights_path}' not found.")

    vle.model.logit_scale = nn.Parameter(torch.tensor(1.0, device=CONFIG['device']))
    vle.model.logit_bias = nn.Parameter(torch.tensor(0.0, device=CONFIG['device']))
    
    vle.set_vision_trainable_params(None)

    del vle.model.visual_proj
    clear_memory()
    vle.model = compile_torch_model(vle.model)

    seg_preprocess_fn = partial(SemanticSegmentation, resize_size=SEG_CONFIG['image_size'])() # same as original one, but with custom resizing

    # training cropping functions
    center_crop_fn = T.CenterCrop(SEG_CONFIG['image_size'])
    random_crop_fn = T.RandomCrop(SEG_CONFIG['image_size'])

    # augmentations
    augment_fn = T.Compose([
        T.Identity()
        # T.RandomHorizontalFlip(p=0.5),
        # T.RandomAffine(degrees=0, scale=(0.5, 2)), # Zooms in and out of the image.
        # T.RandomAffine(degrees=[-30, 30], translate=[0.2, 0.2], scale=(0.5, 2), shear=15), # Full affine transform.
        # T.RandomPerspective(p=0.5, distortion_scale=0.2) # Shears the image
    ])

    train_collate_fn = partial(
        partial(crop_augment_preprocess_batch, output_uids=True),
        crop_fn=center_crop_fn,
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
    aux_criterion = GroupedPairedNegativeSigLipLoss()

    sign_classes_filter = partial(subsample_sign_classes, k=0)
    # sign_classes_filter = None

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
            mask_text_cache,
            checkpoint_dict,
            sign_classes_filter
    )
    except KeyboardInterrupt:
        log_manager.log_title("Training Interrupted", pad_symbol='~')

    segmodel.remove_handles()

    log_manager.close_loggers()


if __name__ == '__main__':
    asyncio.run(main())
