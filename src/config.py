import os
from dotenv import load_dotenv
import random
from pathlib import Path
from typing import Any
import yaml

def load_config(config_filepath: Path) -> dict[str, Any]:
    with open(config_filepath, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict

# Base Paths #

BASE_PATH = Path("/home/olivieri/exp").resolve() # outer-most project path
CONFIG_PATH =  BASE_PATH / "config"

CONFIG = load_config(CONFIG_PATH / "config.yml")

# Reproducibility #

random.seed(CONFIG["seed"])

# Environmental Variables #

load_dotenv(str(CONFIG_PATH / ".env"), override=True)

GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY_2")
GDRIVE_ANNOT_IMGS_PATH = os.getenv("GDRIVE_ANNOT_IMGS_PATH")

def main() -> None:
    c = load_config(CONFIG_PATH / "config.yml")
    print(c["seed"])

if __name__ == '__main__':
    main()
