import os
from dotenv import load_dotenv
import random
from pathlib import Path

# Reproducibility #

# TODO: get seed into the config.yml

def get_seed() -> int:
    return 42

random.seed(get_seed())

# Path #

BASE_PATH = Path("/home/olivieri/exp").resolve() # outer-most project path
config_path =  BASE_PATH / "config"

# Environmental Variables #

load_dotenv(str(config_path / ".env"), override=True)

GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY_2")
GDRIVE_ANNOT_IMGS_PATH = os.getenv("GDRIVE_ANNOT_IMGS_PATH")

if __name__ == "__main__":
    pass
