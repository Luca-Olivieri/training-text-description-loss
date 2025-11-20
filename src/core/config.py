"""
Configuration Management Module

This module handles all configuration management for the project, including:
- Loading and merging configuration files (YAML)
- Setting up reproducibility (random seeds)
- Managing environment variables
- Converting path strings to Path objects
- Configuring PyTorch and other libraries
- Providing a global BASE_CONFIG dictionary

The module automatically initializes the base configuration on import and sets
random seeds for reproducibility across numpy, random, and PyTorch.

Global Variables:
    BASE_CONFIG (dict): The base configuration dictionary loaded from base_config.yml
    base_path (Path): The root path of the project (3 levels up from this file)

Functions:
    load_config: Load a YAML configuration file
    setup_config: Merge base and experiment configurations
    convert_paths_in_dict: Convert string paths to Path objects recursively
    get_torch_gen: Get a cloned PyTorch random generator
"""

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
    """
    Load a YAML configuration file and return it as a dictionary.
    
    Args:
        config_filepath: Path to the YAML configuration file to load.
    
    Returns:
        A dictionary containing the parsed YAML configuration.
    
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
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
    # HuggingFace
    # Timestamp to univocally define config
    config['timestamp'] = f'{datetime.now().strftime("%y%m%d_%H%M")}' # add datetime to exp name
    config = convert_paths_in_dict(config)
    return config

def convert_paths_in_dict(data: Any) -> Any:
    """
    Recursively convert string values to Path objects in nested data structures.
    
    This function traverses a nested data structure (dictionaries and lists)
    and converts string values to pathlib.Path objects if their corresponding
    key, or any of its ancestor keys, contains the word "path" (case-insensitive).
    
    The modification is done in-place on the original data structure.
    
    Example:
        >>> data = {
        ...     'some_path': '/path/to/file',  # Converted (direct key)
        ...     'config': {
        ...         'output_path': {            # Converted (ancestor key)
        ...             'log_file': 'run.log'   # Also converted (ancestor has 'path')
        ...         }
        ...     },
        ...     'name': 'experiment'            # Not converted (no 'path' key)
        ... }
        >>> result = convert_paths_in_dict(data)
        >>> isinstance(result['some_path'], Path)
        True
        >>> isinstance(result['config']['output_path']['log_file'], Path)
        True
    
    Args:
        data: The dictionary, list, or other data structure to process.
              Can be arbitrarily nested with dicts and lists.
    
    Returns:
        The same data structure (modified in-place), with string values
        converted to Path objects where appropriate.
    
    Note:
        - The conversion is case-insensitive (checks for "path", "Path", "PATH", etc.)
        - Only string values are converted; other types remain unchanged
        - Once a "path" key is encountered, all string descendants are converted
    """
    def _convert_recursive(data: Any, is_under_path_key: bool):
        """
        Internal recursive helper function.

        Args:
            data: The current data segment (dict, list, etc.).
            is_under_path_key: True if an ancestor key contained "path".
        """
        # If the data is a dictionary, iterate through its items
        if isinstance(data, dict):
            for key, value in data.items():
                # The new context for children is True if the current context is already
                # True, OR if the current key contains "path".
                new_context = is_under_path_key or (isinstance(key, str) and "path" in key.lower())

                # If the current value is a string and we are in a "path" context,
                # convert it to a Path object.
                if new_context and isinstance(value, str):
                    data[key] = Path(value)
                else:
                    # Otherwise, recurse deeper, passing the new context.
                    _convert_recursive(value, new_context)

        # If the data is a list, the context doesn't change for its items.
        # We apply the same context we received to each item.
        elif isinstance(data, list):
            for item in data:
                _convert_recursive(item, is_under_path_key)
                
        # Base case: if data is not a dict or list, do nothing.
        # The recursion stops.
    
    _convert_recursive(data, is_under_path_key=False)
    return data


# Base Config #

# Project root path (3 levels up from this file: src/core/config.py -> project_root)
base_path = Path(__file__).resolve().parent.parent.parent # project path
# Load the base configuration from YAML
BASE_CONFIG = load_config(base_path / 'config' / 'base_config.yml')
# Convert device string (e.g., 'cuda', 'cpu') to torch.device object
BASE_CONFIG['device'] = torch.device(BASE_CONFIG['device']) # cast device string to torch.device object

# Environmental Variables #

load_dotenv(str(base_path / 'config' / '.env'), override=True)

BASE_CONFIG |= {'_google_AI_key': os.getenv("GOOGLE_AI_KEY")}

# HuggingFace

os.environ["HF_HOME"] = BASE_CONFIG['HF_home']

# PyTorch

torch.hub.set_dir('/home/olivieri/exp/data/private/torch_weights')
torch.backends.cudnn.benchmark = True # if True, can speeds up computation at the cost of reproducibility.

# Reproducibility #

random.seed(BASE_CONFIG['seed'])
np.random.seed(BASE_CONFIG['seed'])
torch.manual_seed(BASE_CONFIG['seed'])

_torch_gen = torch.Generator()
_torch_gen.manual_seed(BASE_CONFIG['seed'])

def get_torch_gen() -> torch.Generator:
    """
    Get a cloned PyTorch random generator with the configured seed.
    
    This function returns a clone of the global PyTorch generator that was
    initialized with the seed from BASE_CONFIG. Using this function ensures
    reproducible random number generation in PyTorch operations while
    maintaining independence between different uses.
    
    Returns:
        A PyTorch Generator object with the state cloned from the global generator.
        Each call returns an independent generator with the same initial state.
    
    Example:
        >>> gen1 = get_torch_gen()
        >>> gen2 = get_torch_gen()
        >>> # gen1 and gen2 have the same initial state but are independent
        >>> torch.randn(3, generator=gen1)
        >>> torch.randn(3, generator=gen2)  # Will produce the same sequence
    
    Note:
        This is useful for ensuring reproducibility in DataLoaders and other
        PyTorch operations that accept a generator parameter.
    """
    return _torch_gen.clone_state()

# Custom Representations

# NOTE this are applied everywhere in the code.
Image.Image.__repr__ = lambda obj: f"<PIL.Image.Image image mode={obj.mode} size={obj.size}>"

def main() -> None:
    # print(setup_config(BASE_CONFIG, Path('/home/olivieri/exp/config/exp_config.yml')))
    print(Path(__file__).resolve().parent.parent.parent)
    print(base_path)

if __name__ == '__main__':
    main()
