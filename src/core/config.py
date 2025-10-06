import os
from dotenv import load_dotenv
import random
from pathlib import Path
import yaml
import numpy as np
import torch
from datetime import datetime
from PIL import Image

from core._types import Any

def load_config(config_filepath: Path) -> dict[str, Any]:
    with open(config_filepath, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict

def setup_config(
        base_config: dict[str, Any],
        exp_config_filepath: Path
) -> dict[str, Any]:
    exp_config = load_config(exp_config_filepath)
    if set(base_config) & set(exp_config):
        raise ValueError(f'Base config and exp config at "{exp_config_filepath}" have some keys in common. They should be disjoint.')
    config = base_config | exp_config
    # Ollama configs
    config |= {'ollama_container_name': os.getenv("OLLAMA_CONTAINER_NAME")} # 'ollama_container_name' is passed to the Docker environment
    config |= {'ollama_http_endpoint': f'http://{config["ollama_container_name"]}:11434'}
    if 'exp_name' in list(config.keys()): 
        config['exp_name'] += f'_{datetime.now().strftime("%y%m%d_%H%M")}' # add datetime to exp name
    config = convert_paths_in_dict(config)
    return config

def convert_paths_in_dict(data: Any) -> Any:
    """
    Recursively traverses a nested data structure (dictionaries and lists)
    and converts string values to pathlib.Path objects if their
    corresponding key contains the word "path".

    The modification is done in-place on the original data structure.

    Args:
        data: The dictionary, list, or other data to process.

    Returns:
        The same data structure, with paths converted.
    """
    # If the data is a dictionary, iterate through its items
    if isinstance(data, dict):
        for key, value in data.items():
            # Check if the key is a string before calling .lower()
            # And check if the value is a string to be converted
            if isinstance(key, str) and "path" in key.lower() and isinstance(value, str):
                data[key] = Path(value)
            else:
                # If the value is a nested structure, recurse into it
                convert_paths_in_dict(value)

    # If the data is a list, iterate through its items and recurse
    elif isinstance(data, list):
        for item in data:
            convert_paths_in_dict(item)
    
    # Base case: if data is not a dict or list, do nothing and return
    return data

# Base Config #

base_path = Path(__file__).resolve().parent.parent.parent # project path
BASE_CONFIG = load_config(base_path / 'config' / 'base_config.yml')
BASE_CONFIG['device'] = torch.device(BASE_CONFIG['device']) # cast device string to torch.device object

# Environmental Variables #

load_dotenv(str(base_path / 'config' / '.env'), override=True)

BASE_CONFIG |= {'_google_AI_key': os.getenv("GOOGLE_AI_KEY")}

# Reproducibility #

random.seed(BASE_CONFIG['seed'])
np.random.seed(BASE_CONFIG['seed'])
torch.manual_seed(BASE_CONFIG['seed'])

_torch_gen = torch.Generator()
_torch_gen.manual_seed(BASE_CONFIG['seed'])

def get_torch_gen() -> torch.Generator:
    return _torch_gen.clone_state()

# Custom Representations

# NOTE this are applied everywhere in the code.
Image.Image.__repr__ = lambda obj: f"<PIL.Image.Image image mode={obj.mode} size={obj.size}>"

# PyTorch

torch.backends.cudnn.benchmark = True # if True, can speeds up computation at the cost of reproducibility.

def main() -> None:
    # print(setup_config(BASE_CONFIG, Path('/home/olivieri/exp/config/exp_config.yml')))
    print(Path(__file__).resolve().parent.parent.parent)
    print(base_path)

if __name__ == '__main__':
    main()
