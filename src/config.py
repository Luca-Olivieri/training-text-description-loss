import os
from dotenv import load_dotenv
import sys

from path import CONFIG_PATH

# Reproducibility #

SEED = 42

# Environmental Variables #

load_dotenv(str(CONFIG_PATH / ".env"), override=True)

os.environ["GOOGLE_AI_KEY"] = os.getenv("GOOGLE_AI_KEY_3")

if __name__ == "__main__":
    print(os.environ["GOOGLE_AI_KEY"])
