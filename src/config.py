import os
from dotenv import load_dotenv
import random
from pathlib import Path

# Reproducibility #

def get_seed() -> int:
    return 42

random.seed(get_seed())

# Path #

ROOT_PATH = Path("/home/olivieri/exp").resolve() # outer-most project path
CONFIG_PATH =  ROOT_PATH / "config"

# Environmental Variables #

load_dotenv(str(CONFIG_PATH / ".env"), override=True)

# TODO assigning a secret value to a regular variable would not be secure
os.environ["GOOGLE_AI_KEY"] = os.getenv("GOOGLE_AI_KEY_2")
GDRIVE_ANNOT_IMGS_PATH = os.getenv("GDRIVE_ANNOT_IMGS_PATH")

if __name__ == "__main__":
    print(os.environ["GOOGLE_AI_KEY"])
