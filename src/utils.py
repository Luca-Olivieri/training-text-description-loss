from IPython.display import display

import torch
from torchvision.datasets import VOCSegmentation
from PIL import Image
from IPython.display import Markdown
import re
import ast

from path import PRIVATE_DATASETS_PATH

def download_VOC2012():
    VOCSegmentation(root=PRIVATE_DATASETS_PATH, image_set='trainval', download=True)
    
### Devices ###

DEVICE = torch.device('cpu')

### Utility Methods ###

def flatten_list(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list) or isinstance(item, tuple): result.extend(flatten_list(item))
        else: result.append(item)
    return result

def partition_list(list_, length):
    """
    Partitions a  list into a list of sub-lists maximum given size
    E.g. [1, 2, 3, 4, 5] -> [1, 2], [3, 4], [5]
    """
    return [list_[i:i + length] for i in range(0, len(list_), length)]

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None

def parse_eval_str_to_dict(eval_str):
    eval_str = extract_json(eval_str)
    try:
        eval_dict = ast.literal_eval(eval_str)
        return eval_dict
    except:
        print("Wrong parsing to dict!")
        return eval_str

def display_prompt(full_prompt):
    for prompt in full_prompt:
        if isinstance(prompt, Image.Image):
            display(prompt)
        else:
            display(Markdown(prompt))
