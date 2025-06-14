from IPython.display import display

import torch
from torchvision.datasets import VOCSegmentation
from PIL import Image
from IPython.display import Markdown
import re
import ast
import time
import functools
import tracemalloc
import asyncio
from typing import Callable, Sequence, TypeVar, Any, Tuple
import numpy as np
import xarray as xr
import pandas as pd
from abc import ABC
from tqdm import tqdm
from typing import Any, TypeVar, Optional
from PIL import Image
from PIL.Image import Image as PILImage

from path import PRIVATE_DATASETS_PATH
from config import *

# Type annotations
from abc import ABC, abstractmethod
Prompt = list[str | PILImage]
Conversation = list[dict[str, str]]
GenericClient = TypeVar
GenericResponse = TypeVar

def download_VOC2012():
    VOCSegmentation(root=PRIVATE_DATASETS_PATH, image_set='trainval', download=True)
    
### Devices ###

DEVICE = torch.device('cuda')

### Utility Methods ###

def flatten_list(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list) or isinstance(item, tuple): result.extend(flatten_list(item))
        else: result.append(item)
    return result

def batch_list(list_: Sequence[Any], batch_size: int):
    """
    Partitions a  list into a list of sub-lists maximum given size
    E.g. [None, 2, "Hello", 4, "World"] -> [None, 2], ["Hello", 4], ["World"]
    """
    if not isinstance(list_, list):
        list_ = list(list_)
    return [list_[i:i + batch_size] for i in range(0, len(list_), batch_size)]

def get_batch_keys_amount(list_of_dicts: list[dict]) -> int:
    return sum([len(d.keys()) for d in list_of_dicts])

def batch_class_splitted_list(list_of_dicts: list[dict], max_keys_per_batch: int) -> list[list[dict]]:

    batches = []
    current_batch = []
    current_key_count = 0

    for d in list_of_dicts:
        num_keys_in_dict = len(d)

        if current_key_count + num_keys_in_dict > max_keys_per_batch:
            if current_batch:
                batches.append(current_batch)
            current_batch = [d]
            current_key_count = num_keys_in_dict
        else:
            current_batch.append(d)
            current_key_count += num_keys_in_dict

    if current_batch:
        batches.append(current_batch)

    return batches

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None

def extract_uppercase_words(text):
    """
    Returns a list of all unique uppercase words from the given text.
    """
    return sorted(set(re.findall(r'\b[A-Z]+\b', text)))

def parse_eval_text_to_dict(eval_str):
    eval_str = extract_json(eval_str)
    try:
        eval_dict = ast.literal_eval(eval_str)
        return eval_dict
    except:
        print("Wrong parsing to dict!")
        return eval_str

def display_prompt(full_prompt):
    if isinstance(full_prompt, str):
        display(Markdown(full_prompt))
    else:
        for prompt in full_prompt:
            if isinstance(prompt, Image.Image):
                display(prompt)
            else:
                display(Markdown(prompt))

def create_empty_dataarray(dims_to_coords: dict[str, Sequence[Any]]) -> xr.DataArray:
    dims, coords = zip(*dims_to_coords.items())
    shape = [len(c) for c in coords]
    da = xr.DataArray(np.empty(shape, dtype=object), coords=coords, dims=dims)
    return da

def track_performance(n_trials: int = 10) -> Callable:
    def decorator(func: Callable) -> Callable:
        """
        A decorator that tracks the execution time, current memory usage,
        and peak memory usage of the decorated function.
        """
        def shared_prologue() -> None:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            start_time = time.perf_counter()
            return start_time
        
        def shared_epilogue(t_0: float) -> Tuple[float]:
            end_time = time.perf_counter()
            elapsed_time = end_time - t_0 # seconds
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            # Stop tracemalloc if it was started by this decorator
            # This is important to avoid interfering with other tracemalloc usage
            # or leaving it running unnecessarily.
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            return elapsed_time, current_memory / (1024 * 1024), peak_memory / (1024 * 1024)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Tuple[Any, pd.DataFrame]:
            perf_da = create_empty_dataarray({"trial": range(n_trials), "metric": ["exec_time", "curr_mem", "peak_mem"]})
            for trial in range(n_trials):
                t_0 = shared_prologue()
                result = await func(*args, **kwargs)
                exec_time, curr_mem, peak_mem = shared_epilogue(t_0)
                perf_da.loc[trial, "exec_time"] = exec_time
                perf_da.loc[trial, "curr_mem"] = curr_mem
                perf_da.loc[trial, "peak_mem"] = peak_mem
            return result, perf_da

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Tuple[Any, pd.DataFrame]:
            perf_da = create_empty_dataarray({"trial": range(n_trials), "metric": ["exec_time", "curr_mem", "peak_mem"]})
            for trial in range(n_trials):
                t_0 = shared_prologue()
                result = func(*args, **kwargs)
                exec_time, curr_mem, peak_mem = shared_epilogue(t_0)
                
            return result, perf_da

        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
        

def retry(
        max_retries: int,
        cooldown_seconds: int | float,
        exceptions_to_catch: list[Exception]
) -> Callable:
    """
    A decorator that retries a function when it encounters a specified set of exceptions.

    Args:
        max_retries (int): The maximum number of times to retry the function.
        cooldown_seconds (int or float): The time in seconds to wait between retries.
        exceptions_to_catch (tuple): A tuple of exception types to catch and trigger a retry.
                                     For example: (ValueError, TypeError).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts <= max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions_to_catch as e:
                    attempts += 1
                    if attempts > max_retries:
                        print(f"Function '{func.__name__}' failed after {max_retries + 1} attempts due to: {e}")
                        raise  # Re-raise the last exception if max retries are exceeded
                    else:
                        print(f"Attempt {attempts}/{max_retries + 1} failed for '{func.__name__}' due to: {e}. Retrying in {cooldown_seconds} seconds...")
                        time.sleep(cooldown_seconds)
                except Exception as e:
                    # Catch any other unexpected exceptions immediately
                    print(f"Function '{func.__name__}' encountered an unhandled exception: {e}. Not retrying.")
                    raise
        return wrapper
    return decorator
    
    return decorator

class DictObject(ABC):
    def __repr__(self) -> str:
        return " ".join([object.__repr__(self), str(self.__dict__)])

    def __getitem__(self, key: str) -> Any:
        try:
            value = self.__dict__[key]
            return value
        except KeyError:
            raise KeyError(f"'GenParams' object has no attribute '{key}'")

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.__dict__.keys():
            self.__dict__[key] = value
        else:
            raise KeyError(f"'GenParams' does not have key '{key}'.")

    def get_assigned_keys(self) -> Any:
        """
        Returns a list of keys that are valued 'None'.
        """
        return [key for key, value in self.__dict__.items() if value is not None]
    
# Create a tqdm progress bar with Hugging Face-like styling
def my_tqdm(data, desc: str = ""):
    if not isinstance(data, list):
        data = list(data)
    return tqdm(
        enumerate(data),
        total=len(data),
        desc=desc,  # Add a description
        unit="item",               # Specify the unit of progress
        colour="#67ad5b",            # Set a vibrant color,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

def main() -> None:
    @track_performance(2)
    def compute():
        list(range(1_000_000))
        return "Ciao"
    
    r = asyncio.run(compute())
    print(r)

if __name__ == "__main__":
    main()
