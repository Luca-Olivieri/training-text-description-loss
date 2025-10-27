from core.config import *
from models.mllm import MLLMResponse

from core.torch_utils import blend_tensors

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
