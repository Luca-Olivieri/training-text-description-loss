from config import *
from data import ImageCaptionDataset
from models.vl_encoders import VLE_REGISTRY, VLEncoder
from viz import print_layer_numel


def main() -> None:
    
    answers_ds = ImageCaptionDataset(Path("/home/olivieri/exp/data/data_gen/VOC2012/flat/train_no_aug_flat.jsonl"))

    vle: VLEncoder = VLE_REGISTRY.get("flair", device=CONFIG['device'])

    print(vle)

    vle.set_vision_trainable_params('visual_proj')

    print_layer_numel(vle, print_only_total=True, only_trainable=True)



if __name__ == '__main__':
    main()
