import os
from dotenv import load_dotenv
import random
from pathlib import Path
from typing import Any
import yaml
import numpy as np
import torch
from datetime import datetime
from PIL import Image

def load_config(config_filepath: Path) -> dict[str, Any]:
    with open(config_filepath, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict

# Base Paths #

BASE_PATH = Path("/home/olivieri/exp").resolve() # outer-most project path
CONFIG_PATH =  BASE_PATH / "config"

CONFIG = load_config(CONFIG_PATH / "config.yml")

for module in ['segnet', 'vle']:
    CONFIG[module]['train']["exp_name"] += f'_{datetime.now().strftime("%y%m%d_%H%M")}'

# PyTorch Hub
TORCH_WEIGHTS_ROOT = Path('/home/olivieri/exp/data/torch_weights')
TORCH_WEIGHTS_CHECKPOINTS = TORCH_WEIGHTS_ROOT / 'checkpoints'
torch.hub.set_dir(TORCH_WEIGHTS_ROOT) # set local model weights directory

# HuggingFace Hub
os.environ["HF_HOME"] = '/home/olivieri/exp/data/huggingface/datasets'
os.environ["HF_HUB_CACHE"] = '/home/olivieri/exp/data/huggingface/hub'

# Reproducibility #

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

_torch_gen = torch.Generator()
_torch_gen.manual_seed(CONFIG["seed"])

def get_torch_gen() -> torch.Generator:
    return _torch_gen.clone_state()

# Environmental Variables #

load_dotenv(str(CONFIG_PATH / ".env"), override=True)

GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY_3")
GDRIVE_ANNOT_IMGS_PATH = os.getenv("GDRIVE_ANNOT_IMGS_PATH")

# Custom Representations

# BEWARE! this are applied everywhere in the code.
Image.Image.__repr__ = lambda obj: f"<PIL.Image.Image image mode={obj.mode} size={obj.size}>"

# PyTorch

torch.backends.cudnn.benchmark = True # if True, can speeds up computation at the cost of reproducibility.

def main() -> None:
    # c = load_config(CONFIG_PATH / "config.yml")
    # print(c["seed"])
    print(os.environ['HF_HOME'])
    print(os.environ['HF_HUB_CACHE'])

if __name__ == '__main__':
    main()
