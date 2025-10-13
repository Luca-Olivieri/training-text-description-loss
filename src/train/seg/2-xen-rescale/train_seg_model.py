from core.config import *
from core.datasets import VOC2012SegDataset
from core.data import crop_augment_preprocess_batch
from models.seg import SegModelWrapper, SEGMODELS_REGISTRY
from models.mllm import MLLMGenParams, OllamaMLLMAdapter, MLLM_REGISTRY
from models.vle import VLE_REGISTRY, VLEncoder, NewLayer, MapComputeMode
from core.prompter import FastPromptBuilder
from core.logger import LogManager
from core.viz import get_layer_numel_str, create_diff_mask
from core.torch_utils import compile_torch_model, nanstd, map_tensors, unprefix_state_dict, blend_tensors
from core.utils import subsample_sign_classes
from cache.cache import Cache, PercentilePolicy, MaskTextCache, Identity
from core.data_utils import flatten_cs_dicts, unflatten_cs_dicts
from core.pipeline import extract_content_from_mllm_responses
from train.seg.loss import xen_rescaler, pow_rescale_fn

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
from torch.nn.modules.loss import _Loss
import math

import asyncio

from core._types import Optional, Callable

config = setup_config(BASE_CONFIG, Path('/home/olivieri/exp/src/train/seg/2-xen-rescale/config.yml'))

seg_config = config['seg']
seg_train_config = seg_config['train']
seg_train_with_text_config = seg_train_config['with_text']

vlm_config = config['vlm']

vle_config = config['vle']

config['var_name'] += f'-{config["timestamp"]}'

exp_path: Path = config['root_exp_path'] / config['exp_name'] / config['var_name']

logs_path: Path = exp_path/'logs'

save_weights_root_path: Path = exp_path/'weights'

if seg_train_with_text_config['with_text'] is False:
    seg_train_with_text_config['with_cache'] = False # no cache if text is not involved

if seg_train_config['log_only_to_stdout']:
    log_manager = LogManager(
        exp_name=f'{config["exp_name"]}-{config["var_name"]}',
        exp_desc=config['exp_desc'],
    )
else:
    log_manager = LogManager(
        exp_name=f'{config["exp_name"]}-{config["var_name"]}',
        exp_desc=config['exp_desc'],
        file_logs_dir_path=logs_path,
        tb_logs_dir_path=logs_path
    )

async def train_loop(
        segmodel: SegModelWrapper,
        vlm: OllamaMLLMAdapter,
        vle: VLEncoder,
        train_dl: DataLoader,
        val_dl: DataLoader,
        fast_prompt_builder: FastPromptBuilder,
        seg_preprocess_fn: nn.Module,
        gen_params: MLLMGenParams,
        train_criterion: _Loss,
        val_criterion: _Loss,
        weights_trend_fn: Callable[[torch.Tensor], torch.Tensor],
        metrics_dict: dict[str, tm.Metric],
        mask_text_cache: Optional[MaskTextCache],
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

    num_batches_per_epoch = len(train_dl)
    # The number of optimizer steps per epoch
    num_steps_per_epoch = math.ceil(num_batches_per_epoch)

    # --- 2. Optimizer Setup ---
    lr=seg_train_config['lr_schedule']['base_lr']
    optimizer = torch.optim.AdamW(segmodel.model.parameters(), lr=lr, weight_decay=math.pow(10, -0.5))
    if checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

    # --- 3. Scheduler Setup ---
    total_steps = num_steps_per_epoch * seg_train_config['num_epochs']
    sched_config = seg_train_config['lr_schedule']
    
    scheduler = None
    if sched_config['policy'] == 'const':
        scheduler = const_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)
    elif sched_config['policy'] == 'const-cooldown':
        cooldown_steps = num_steps_per_epoch * sched_config['epochs_cooldown']
        scheduler = const_lr_cooldown(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps, cooldown_steps, sched_config['lr_cooldown_power'], sched_config['lr_cooldown_end'])
    elif sched_config['policy'] == 'cosine':
        scheduler = cosine_lr(optimizer, sched_config['base_lr'], sched_config['warmup_length'], total_steps)

    # --- 4. AMP and Model Compilation Setup ---
    autocast = get_autocast(seg_train_config['precision'])
    scaler = GradScaler() if seg_train_config['precision'] == "amp" else None
    if scaler and checkpoint_dict:
        scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])

    # --- 5. Initial Validation ---
    log_manager.log_title("Initial Validation")
    val_loss, val_metrics_score = segmodel.evaluate(val_dl, val_criterion, metrics_dict)
    log_manager.log_scores(
        title=f"Before any weight update, VALIDATION",
        loss=val_loss,
        metrics_score=val_metrics_score,
        tb_log_counter=start_epoch,
        tb_phase="val",
        suffix=None,
        metrics_prefix="val_"
    )
    best_val_mIoU = val_metrics_score['mIoU']

    log_manager.log_title("Training Start")
    
    if seg_train_with_text_config['with_cache']:
        log_manager.log_line(mask_text_cache)
    
    # --- 6. Main Training Loop ---
    train_metrics = tm.MetricCollection(metrics_dict)
    for epoch in range(start_epoch, seg_train_config["num_epochs"]):
        
        train_metrics.reset() # in theory, this can be removed
        segmodel.model.train()

        for step, (uids, scs_img, gts) in enumerate(train_dl):

            cs_counter = 0
            filtered_cs_counter = 0

            # --- Seg --- #

            scs: torch.Tensor = seg_preprocess_fn(scs_img)

            scs: torch.Tensor = scs.to(config['device'])
            gts: torch.Tensor = gts.to(config['device']) # shape [B, H, W]
            
            with autocast():
                if not 'dropout' in seg_train_config['regularizers']:
                    segmodel.model.eval() # in eval mode, no dropout
                logits = segmodel.model(scs)
                segmodel.model.train()
                logits: torch.Tensor = logits["out"] if isinstance(logits, OrderedDict) else logits # shape [N, C, H, W]

                batch_loss: torch.Tensor = train_criterion(logits, gts)

            train_metrics.update(logits.detach().argmax(dim=1), gts)

            if seg_train_with_text_config['with_text']:

                # --- VLM --- #

                scs_img = (scs_img*255).to(torch.uint8)
                gts = gts.unsqueeze(1)
                prs = logits.argmax(dim=1, keepdim=True)
                # Both VLM and VLE receive the images in the same downsampled size.
                gts_down = TF.resize(gts, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
                prs_down = TF.resize(prs, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
                scs_down = TF.resize(scs_img, fast_prompt_builder.image_size, TF.InterpolationMode.BILINEAR)
                cs_prompts = fast_prompt_builder.build_cs_inference_prompts(map_tensors(gts_down, fast_prompt_builder.class_map), map_tensors(prs_down, fast_prompt_builder.class_map), scs_down, sign_classes_filter)

                cs_prompts_by_uid = {uid: cs_p for uid, cs_p in zip(uids, cs_prompts)}
                prs_by_uid = {uid: pr for uid, pr in zip(uids, prs_down)}

                cs_dict: dict[str, list[int]] = {uid: list(cs_p.keys()) for uid, cs_p in zip(uids, cs_prompts)}

                cs_counter += sum([1 for uid, sign_classes in cs_dict.items() for pos_c in sign_classes])

                cs_splitted_bool_masks = {uid: {pos_c: prs_by_uid[uid] == pos_c for pos_c in sign_classes} for uid, sign_classes in cs_dict.items()}
                
                if seg_train_with_text_config['with_cache']:
                    cs_dict_to_update: dict[str, list[int]] = mask_text_cache.get_cs_keys_to_update(cs_splitted_bool_masks)
                else:
                    cs_dict_to_update = cs_dict

                filtered_cs_prompts_dict = {uid: {pos_c: cs_prompts_by_uid[uid][pos_c] for pos_c in sign_classes} for uid, sign_classes in cs_dict_to_update.items()}
                
                flat_cs_inference_prompts, flat_pos_classes, batch_indices = flatten_cs_dicts(list(filtered_cs_prompts_dict.values()))
                
                flat_answers = await vlm.predict_batch(flat_cs_inference_prompts, gen_params=gen_params)
                flat_answers = extract_content_from_mllm_responses(flat_answers)

                cs_answer_list = unflatten_cs_dicts(flat_answers, flat_pos_classes, batch_indices, original_batch_size=len(cs_dict_to_update))
                cs_answer_dict_to_update = {uid: cs_answer for uid, cs_answer in zip(list(cs_dict_to_update.keys()), cs_answer_list)}
                cs_answer_dict = {uid: cs_answer_dict_to_update.get(uid, None) for uid in cs_dict.keys()}
                
                # images and texts to update
                cs_splitted_bool_masks_to_update = {uid: {pos_c: prs_by_uid[uid] == pos_c for pos_c in sign_classes} for uid, sign_classes in cs_dict_to_update.items()}
                batch = {f"{uid}-{pos_c}": (bool_mask, cs_answer_dict_to_update[uid][pos_c]) for uid, cs_bool_mask in cs_splitted_bool_masks_to_update.items() for pos_c, bool_mask in cs_bool_mask.items()}
                
                # --- VLE --- #

                with autocast():

                    batch_maps = []

                    # aggregate the class-splitted global text tokens
                    for i, (uid, sc, gt, pr) in enumerate(zip(uids, scs_down, gts_down, prs_down)):

                        if uid in cs_answer_dict_to_update.keys():

                            cs_answers = cs_answer_dict[uid] # gather the text for each pos. class of this image
                            cs_maps = []

                            filtered_cs_counter += len(cs_answers.keys())

                            for pos_c, ans in cs_answers.items():

                                pos_class_gt = (gt == pos_c)
                                pos_class_pr = (pr == pos_c)

                                diff_mask = create_diff_mask(pos_class_gt, pos_class_pr)

                                # L overlay image
                                ovr_diff_mask_L = blend_tensors(sc, diff_mask*255, alpha=0.55)
                                
                                img_tensor = vle.preprocess_images([ovr_diff_mask_L])
                                text_tensor = vle.preprocess_texts([ans])
                                map, min_value, max_value = vle.get_maps(
                                    img_tensor,
                                    text_tensor,
                                    map_compute_mode=MapComputeMode.ATTENTION,
                                    upsample_size=seg_config['image_size'],
                                    upsample_mode=TF.InterpolationMode.BILINEAR,
                                    attn_heads_idx=[0, 3, 5, 7] # as done by the authors
                                ) # [1, 1, H, W], m, M
                                map = map.squeeze(0).squeeze(0) # [H, W]

                                cs_maps.append(map)
                        else:
                            cs_maps = []
                            
                            if isinstance(seg_config['image_size'], int):
                                img_size = [seg_config['image_size']]*2
                            
                            # default maps are uniformly valued
                            map = torch.full(
                                size=img_size,
                                fill_value=1/(img_size[0]*img_size[1]),
                                device=config['device']
                            )
                            cs_maps.append(map)

                        reduced_cs_maps = torch.stack(cs_maps, dim=0).mean(dim=0) # [H, W]
                    
                        batch_maps.append(reduced_cs_maps)

                    batch_maps = torch.stack(batch_maps) # [B, H, W]
                    batch_maps = weights_trend_fn(batch_maps) # [B, H, W]

                    old_batch_loss = batch_loss.clone().detach()
                    batch_loss = xen_rescaler(
                        loss=batch_loss,
                        weights=batch_maps,
                        alpha=seg_train_with_text_config['rescale_alpha'],
                        normalise=True
                    )
                    rescale_mult = batch_loss.detach().mean()/old_batch_loss.detach().mean()
                    del old_batch_loss

                    cs_mult = filtered_cs_counter/(train_dl.batch_size)
                    filtered_perc = filtered_cs_counter/cs_counter
            else:
                batch_loss = batch_loss.mean()
                cs_mult = -1.0
                filtered_perc = -1.0
                rescale_mult = -1.0

            backward(batch_loss, scaler)

            if scheduler:
                scheduler(global_step)

            max_grad_norm = seg_train_config['grad_clip_norm']
            if scaler:
                if seg_train_config['grad_clip_norm']:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        segmodel.model.parameters(),
                        max_grad_norm if max_grad_norm else float('inf'),
                        norm_type=2.0
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                if seg_train_config['grad_clip_norm']:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        segmodel.model.parameters(),
                        max_grad_norm if max_grad_norm else float('inf'),
                        norm_type=2.0
                    )
                optimizer.step()

            optimizer.zero_grad()

            global_step += 1 # Increment global step *only* after an optimizer step

            # --- Logging ---
            if global_step % seg_train_config['log_every'] == 0:
                train_metrics_score = train_metrics.compute()
                if seg_train_with_text_config['with_cache']:
                    metric_diffs_values = torch.stack(list(mask_text_cache.get_metric_diffs(batch).values()))
                    train_metrics_score |= {'metric_diff_mean': metric_diffs_values.nanmean(), 'metric_diff_std': nanstd(metric_diffs_values, dim=0)}
                train_metrics_score['cs_mult'] = torch.tensor(cs_mult)
                train_metrics_score['filtered_perc'] = torch.tensor(filtered_perc)
                train_metrics_score['rescale_mult'] = torch.tensor(rescale_mult)
                train_metrics_score['lr'] = torch.tensor(optimizer.param_groups[0]['lr'])
                train_metrics_score['grad_norm'] = grad_norm
                # train_metrics_score |= {"quantile": torch.tensor(mask_text_cache.update_policy.trend.get_current_value())}
                step_in_epoch = (step) + 1
                log_manager.log_scores(
                    title=f"epoch: {epoch+1}/{seg_train_config['num_epochs']}, step: {step_in_epoch}/{num_steps_per_epoch} (global_step: {global_step})",
                    loss=batch_loss,
                    metrics_score=train_metrics_score,
                    tb_log_counter=global_step,
                    tb_phase="train",
                    suffix=None,
                    # suffix=f", lr: {current_lr:.2e}, grad_norm: {grad_norm:.2f}",
                    metrics_prefix="batch_"
                )
            
            if seg_train_with_text_config['with_cache']:
                mask_text_cache.update(batch)

            train_metrics.reset() # only the batch metrics are logged

        # --- End of Epoch Validation and Checkpointing ---
        val_loss, val_metrics_score = segmodel.evaluate(val_dl, val_criterion, metrics_dict)
        log_manager.log_scores(
            title=f"epoch: {epoch+1}/{seg_train_config['num_epochs']}, VALIDATION",
            loss=val_loss,
            metrics_score=val_metrics_score,
            tb_log_counter=epoch+1,
            tb_phase="val",
            suffix=None,
            metrics_prefix="val_"
        )

        if val_metrics_score['mIoU'] > best_val_mIoU:
            best_val_mIoU = val_metrics_score['mIoU']
            
            if save_weights_root_path:
                # NOTE on resume we start from 'epoch + 1'
                new_checkpoint_dict = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': segmodel.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if scaler:
                    new_checkpoint_dict['scaler_state_dict'] = scaler.state_dict()

                save_weights_root_path.mkdir(parents=False, exist_ok=True)
                ckp_filename = f'lraspp_mobilenet_v3_large-{config["exp_name"]}-{config["var_name"]}.pth'
                full_ckp_path = save_weights_root_path / ckp_filename
                torch.save(new_checkpoint_dict, full_ckp_path)
                log_manager.log_line(f"New best model saved to {full_ckp_path} with validation mIoU: {best_val_mIoU:.4f}")
    
    log_manager.log_title("Training Finished")

async def main() -> None:
    img_idxs = None

    train_ds = VOC2012SegDataset(
        root_path=config['datasets']['VOC2012_root_path'],
        split='train',
        device=config['device'],
        resize_size=seg_config['image_size'],
        center_crop=False,
        with_unlabelled=True,
        output_uids=True,
        img_idxs=img_idxs
    )
    
    val_ds = VOC2012SegDataset(
        root_path=config['datasets']['VOC2012_root_path'],
        split='val',
        device=config['device'],
        resize_size=seg_config['image_size'],
        center_crop=False,
        with_unlabelled=True,
        img_idxs=img_idxs
    )

    # Segmentation Model
    segmodel = SEGMODELS_REGISTRY.get(
        'lraspp_mobilenet_v3_large',
        pretrained_weights_path=seg_config['pretrained_weights_path'],
        device=config['device'],
        adaptation=seg_config['adaptation']
    )

    segmodel.adapt()

    if seg_train_with_text_config['with_cache']:
        cache = Cache(storage_device='cpu', memory_device='cuda')
        update_policy = PercentilePolicy(
            metric=BinaryJaccardIndex().to(config['device']),
            percentile=seg_train_with_text_config['cache_update_policy_percentile'],
            trend=Identity()
        )
        mask_text_cache = MaskTextCache(cache, update_policy)
        
        if seg_train_with_text_config['cache_path']:
            mask_text_cache.cache = Cache.load(
                directory_path=seg_train_with_text_config['cache_path'],
                storage_device="cpu",
                memory_device="cuda"
            )
    else:
        mask_text_cache = None

    if seg_config['checkpoint_path']:
        state_dict: OrderedDict = torch.load(seg_config['checkpoint_path'])
        model_state_dict = state_dict.get('model_state_dict', state_dict)
        segmodel.model.load_state_dict(model_state_dict)

    segmodel.set_trainable_params(train_decoder_only=False)

    checkpoint_dict = None
    if seg_train_config['resume_path']:
        seg_weights_path = Path(seg_train_config['resume_path'])
        if seg_weights_path.exists():
            checkpoint_dict = torch.load(seg_weights_path, map_location=config['device'])
            segmodel.model.load_state_dict(checkpoint_dict['model_state_dict'])
        else:
            raise AttributeError(f"ERROR: Resume path '{seg_weights_path}' not found. ")

    # Vision-Language Model
    model_name = 'gemma3:12b-it-qat'
    vlm = MLLM_REGISTRY.get(model_name, http_endpoint=config['ollama_http_endpoint'])

    gen_params = MLLMGenParams(seed=config['seed'], **vlm_config['MLLMGenParams'])

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
        root_path=config['datasets']['VOC2012_root_path'],
        split='train',
        device=config['device'],
        resize_size=seg_config['image_size'],
        center_crop=True,
        with_unlabelled=False,
    )

    sup_set_seg_dataset = VOC2012SegDataset(
        root_path=config['datasets']['VOC2012_root_path'],
        split='prompts_split',
        device=config['device'],
        resize_size=seg_config['image_size'],
        center_crop=True,
        with_unlabelled=False,
        mask_prs_path=config['mask_prs_path']
    )

    fast_prompt_builder = FastPromptBuilder(
        seg_dataset=seg_dataset,
        prompts_file_path=config['prompts_path'] / 'fast_cs_prompt.json',
        prompt_blueprint=prompt_blueprint,
        by_model=config['by_model'],
        alpha=vlm_config['alpha'],
        class_map=seg_dataset.get_class_map(with_unlabelled=False),
        color_map=seg_dataset.get_color_map_dict(with_unlabelled=False),
        image_size=vlm_config['image_size'],
        sup_set_img_idxs=vlm_config['sup_set_img_idxs'],
        sup_set_gt_path=config['sup_set_gt_path'],
        sup_set_seg_dataset=sup_set_seg_dataset,
        str_formats=None,
        seed=config["seed"],
    )

    # Vision-Language Encoder

    new_layers = [new_layer for new_layer in NewLayer if new_layer.value in vle_config['new_layers']]

    vle: VLEncoder = VLE_REGISTRY.get(
        "flair",
        version='flair-cc3m-recap.pt',
        pretrained_weights_root_path=vle_config['pretrained_weights_root_path'],
        new_layers=new_layers,
        device=config['device']
    )

    if vle_config['checkpoint_path']:
        vle_checkpoint_path = Path(vle_config['checkpoint_path'])
        if vle_checkpoint_path.exists():
            vle.model.load_state_dict(unprefix_state_dict(torch.load(vle_checkpoint_path, map_location=config['device'])['model_state_dict'], prefix='_orig_mod'))
        else:
            raise AttributeError(f"ERROR: VLE weights path '{vle_checkpoint_path}' not found.")
    
    vle.set_trainable_params(None)

    vle.model = compile_torch_model(vle.model)

    seg_preprocess_fn = partial(SemanticSegmentation, resize_size=seg_config['image_size'])() # same as original one, but with custom resizing

    # training cropping functions
    if 'random_crop' in seg_train_config['regularizers']:
        crop_fn = T.RandomCrop(seg_config['image_size'])
    else:
        crop_fn = T.CenterCrop(seg_config['image_size'])

    # augmentations
    augment_fn = T.Compose([
        T.Identity(),
    ])
    if 'random_horizontal_flip' in seg_train_config['regularizers']:
        augment_fn.transforms.append(T.RandomHorizontalFlip(p=0.5))

    train_collate_fn = partial(
        partial(crop_augment_preprocess_batch, output_uids=True),
        crop_fn=crop_fn,
        augment_fn=augment_fn,
        preprocess_fn=None
    )

    val_collate_fn = partial(
        crop_augment_preprocess_batch,
        crop_fn=T.CenterCrop(seg_config['image_size']),
        augment_fn=None,
        preprocess_fn=seg_preprocess_fn
    )

    train_criterion = nn.CrossEntropyLoss(ignore_index=21, reduction='none')
    val_criterion = nn.CrossEntropyLoss(ignore_index=21)

    weights_trend_fn = partial(pow_rescale_fn, exp=0.8)

    if seg_train_with_text_config['with_text']:
        if seg_train_with_text_config['sign_classes_filter_k'] is not None:
            sign_classes_filter = partial(
                subsample_sign_classes,
                k=seg_train_with_text_config['sign_classes_filter_k']
            ) # only take K PR classes
        else:
            sign_classes_filter = None # take PR all classes
    else:
        sign_classes_filter = None

    train_dl = DataLoader(
        train_ds,
        batch_size=seg_train_config['batch_size'],
        shuffle=True,
        generator=get_torch_gen(),
        collate_fn=train_collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=seg_train_config['batch_size'],
        shuffle=False,
        generator=get_torch_gen(),
        collate_fn=val_collate_fn,
    )

    metrics_dict = {
        "acc": MulticlassAccuracy(num_classes=train_ds.get_num_classes(with_unlabelled=True), top_k=1, average='micro', multidim_average='global', ignore_index=21).to(config['device']),
        "mIoU": MulticlassJaccardIndex(num_classes=train_ds.get_num_classes(with_unlabelled=True), average='macro', ignore_index=21).to(config['device']),
    }

    log_manager.log_intro(
        config=config,
        train_ds=train_ds,
        val_ds=val_ds,
        train_dl=train_dl,
        val_dl=val_dl
    )

    # Log trainable parameters
    log_manager.log_title("Trainable Params")
    [log_manager.log_line(t) for t in get_layer_numel_str(segmodel.model, print_only_total=False, only_trainable=True).split('\n')]

    if seg_train_with_text_config['with_text']:
        log_manager.log_line("loading Ollama VLM...")
        await vlm.load_model()
        log_manager.log_line("Done loading Ollama VLM.")

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
            train_criterion,
            val_criterion,
            weights_trend_fn,
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
