from core.config import *
from models.mllm import MLLMResponse

def convert_mllm_responses_into_jsonl_objects(
        img_idxs: list[int],
        img_uids: list[str],
        pos_classes: list[int],
        responses: list[MLLMResponse],
) -> list[dict]:
    return [{'img_idxs': idx, 'img_uid': str(uid), 'pos_class': pos_c, 'content': resp.text} for idx, uid, pos_c, resp in zip(img_idxs, img_uids, pos_classes, responses)]
