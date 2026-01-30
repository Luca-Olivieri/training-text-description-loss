from core.config import *
from core.data_utils import flatten_cs_dicts, unflatten_cs_dicts, flatten_list_of_lists
from core.torch_utils import blend_tensors, map_tensors, flatten_tensor_list
from core.prompter import FastPromptBuilder

from models.seg import SegModelWrapper
from models.mllm import MLLMResponse, MLLMAdapter, MLLMGenParams
from models.vle import VLEncoder

from train.seg.loss import GroupedSigLipLoss

from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torchmetrics as tm
from torchmetrics.metric import Metric

from core._types import Optional, Callable

def convert_mllm_responses_into_jsonl_objects(
        img_idxs: list[int],
        img_uids: list[str],
        pos_classes: list[int],
        responses: list[MLLMResponse],
) -> list[dict]:
    return [{'img_idx': idx, 'img_uid': str(uid), 'pos_class': pos_c, 'content': resp.text} for idx, uid, pos_c, resp in zip(img_idxs, img_uids, pos_classes, responses, strict=True)]

def extract_content_from_mllm_responses(
        responses: list[MLLMResponse],
) -> list[str]:
    return [resp.text for resp in responses]

# TODO the method should explicitly handle separately gts and prs, is is not clear that you should provide 2x masks.
def format_pos_ovr_masks(
        scs_img: torch.Tensor,
        mask: torch.Tensor,
        pos_classes: list[int],
        mask_alpha: float = 0.55
) -> torch.Tensor:
    
    pos_classes = torch.tensor(pos_classes).to(mask.device).view(-1, 1, 1, 1).repeat(2, 1, 1, 1)

    pos_class_masks = (mask == pos_classes).to(torch.uint8)*255

    ovr_masks = (blend_tensors(scs_img.repeat(2, 1, 1, 1), pos_class_masks.expand(-1, 3, -1, -1), mask_alpha)/255.).float()

    return ovr_masks

async def evaluate_with_contr_loss(
            segmodel: SegModelWrapper,
            dl: DataLoader,
            criterion: nn.modules.loss._Loss,
            aux_criterion: GroupedSigLipLoss,
            metrics_dict: dict[str, Metric],
            fast_prompt_builder: FastPromptBuilder,
            sign_classes_filter: Optional[Callable[[list[int]], list[int]]],
            vlm: MLLMAdapter,
            gen_params: MLLMGenParams,
            autocast: Callable,
            vle: VLEncoder
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        running_seg_loss = 0.0
        running_seg_supcount = 0
        
        running_aux_loss = 0.0
        running_aux_supcount = 0

        metrics = tm.MetricCollection(metrics_dict)
        metrics.reset()

        segmodel.model.eval()

        with torch.inference_mode():
            for step, (uids, scs_img, gts) in enumerate(dl):
                scs: torch.Tensor = segmodel.preprocess_images(scs_img)
                gts: torch.Tensor

                scs = scs.to(segmodel.device)
                gts = gts.to(segmodel.device) # [B, H, W]

                with autocast():
                    logits = segmodel.model(scs)
                    logits = logits["out"] if isinstance(logits, OrderedDict) else logits # [B, C, H, W]
                    seg_batch_loss: torch.Tensor = criterion(logits, gts)
                
                running_seg_loss += seg_batch_loss.item() * gts.size(0)
                running_seg_supcount += gts.size(0)

                metrics.update(logits.argmax(dim=1), gts)

                # contr. loss part

                scs_img = (scs_img*255).to(torch.uint8)
                gts = gts.unsqueeze(1)
                prs = logits.argmax(dim=1, keepdim=True)
                # Both VLM and VLE receive the images in the same downsampled size.
                gts_down = TF.resize(gts, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
                prs_down = TF.resize(prs, fast_prompt_builder.image_size, TF.InterpolationMode.NEAREST)
                scs_down = TF.resize(scs_img, fast_prompt_builder.image_size, TF.InterpolationMode.BILINEAR)
                cs_prompts = fast_prompt_builder.build_cs_inference_prompts(map_tensors(gts_down, fast_prompt_builder.class_map), map_tensors(prs_down, fast_prompt_builder.class_map), scs_down, sign_classes_filter)

                cs_prompts_by_uid = {uid: cs_p for uid, cs_p in zip(uids, cs_prompts)}

                cs_dict: dict[str, list[int]] = {uid: list(cs_p.keys()) for uid, cs_p in zip(uids, cs_prompts)}
                
                cs_dict_to_update = cs_dict

                filtered_cs_prompts_dict = {uid: {pos_c: cs_prompts_by_uid[uid][pos_c] for pos_c in sign_classes} for uid, sign_classes in cs_dict_to_update.items()}
                
                flat_cs_inference_prompts, flat_pos_classes, batch_indices = flatten_cs_dicts(list(filtered_cs_prompts_dict.values()))
                
                flat_answers = await vlm.predict_batch(flat_cs_inference_prompts, gen_params=gen_params)
                flat_answers = extract_content_from_mllm_responses(flat_answers)

                cs_answer_list = unflatten_cs_dicts(flat_answers, flat_pos_classes, batch_indices, original_batch_size=len(cs_dict_to_update))
                cs_answer_dict_to_update = {uid: cs_answer for uid, cs_answer in zip(list(cs_dict_to_update.keys()), cs_answer_list)}
                
                # --- VLE --- #

                with autocast():
                    # B is the batch size
                    # P is the total (unrolled) number of positive pairs (P) involved in the contrastive loss.

                    cs_texts = [list(cs_a.values()) for cs_a in cs_answer_dict_to_update.values()]
                    cs_texts = [vle.preprocess_texts(cs_txt) for cs_txt in cs_texts] # list of class-splitted tensors (., n_t)
                    
                    flat_cs_texts, cs_text_struct_info = flatten_tensor_list(cs_texts) # (P, n_t)
                    P = len(flat_cs_texts)

                    with torch.no_grad():
                        flat_cs_vle_txt_output = vle.encode_and_project_texts(flat_cs_texts) # (P, D)
                    
                    cs_global_text_token = flat_cs_vle_txt_output.global_text_token # (P, D)
                    
                    bottleneck_out: torch.Tensor = segmodel.activations['bottleneck'] # (B, D_bn, H_bn, W_bn)

                    if not segmodel.model.bottleneck_adapter.needs_query:
                        bottleneck_out: torch.Tensor = segmodel.adapt_tensor(bottleneck_out) # (B, D)
                        b_global_image_token_dict: dict[str, torch.Tensor] = dict(zip(uids, bottleneck_out, strict=True))

                        b_global_image_token = torch.cat([b_global_image_token_dict[uid].unsqueeze(0).expand(len(pos_classes), -1) for uid, pos_classes in cs_dict_to_update.items()]) # (P, D)
                    else:
                        b_global_image_token_dict: dict[str, torch.Tensor] = dict(zip(uids, bottleneck_out, strict=True))

                        b_global_image_token = torch.cat([b_global_image_token_dict[uid].unsqueeze(0).expand(len(pos_classes), -1, -1, -1) for uid, pos_classes in cs_dict_to_update.items()]) # (P, D_bn, H, W)
                        b_global_image_token: torch.Tensor = segmodel.model.bottleneck_adapter(b_global_image_token, cs_global_text_token) # (P, D)

                    lhs: torch.Tensor = b_global_image_token # (P, D)
                    rhs: torch.Tensor = cs_global_text_token # (P, D)
                    
                    image_indices_for_pos_texts = torch.tensor(list(range(len(lhs))), device=rhs.device) # (P,)

                    group_indices_for_pos_pairs = [[i]*len(pos_classes) for i, pos_classes in enumerate(cs_dict_to_update.values())]
                    group_indices_for_pos_pairs, _ = flatten_list_of_lists(group_indices_for_pos_pairs)
                    group_indices_for_pos_pairs = torch.tensor(group_indices_for_pos_pairs, device=rhs.device) # (P,)

                    aux_batch_loss: torch.Tensor = aux_criterion(
                        image_features=lhs,
                        positive_text_features=rhs,
                        image_indices_for_pos_texts=image_indices_for_pos_texts,
                        group_indices_for_pos_pairs=group_indices_for_pos_pairs,
                        logit_scale=vle.logit_scale.exp(),
                        logit_bias=vle.logit_bias,
                        output_dict=False,
                    )

                running_aux_loss += aux_batch_loss.item() * P
                running_aux_supcount += P
                
                del scs, gts, logits

        seg_loss = torch.tensor(running_seg_loss / running_seg_supcount)
        aux_loss = torch.tensor(running_aux_loss / running_aux_supcount)
        metrics_score: dict[str, torch.Tensor] = metrics.compute()

        return seg_loss, aux_loss, metrics_score
