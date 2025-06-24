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
from typing import Callable, TypeVar, Any, Iterable
import numpy as np
import xarray as xr
import pandas as pd
from abc import ABC
from tqdm import tqdm
from typing import Any, TypeVar, Optional
from PIL import Image
from PIL.Image import Image as PILImage

from config import *
from path import PRIVATE_DATASETS_PATH

# Type annotations
from abc import ABC
Prompt = list[str | PILImage]
GenericClient = TypeVar
GenericResponse = TypeVar

def download_VOC2012() -> None:
    """Downloads the VOC2012 dataset using torchvision's VOCSegmentation utility."""
    VOCSegmentation(root=PRIVATE_DATASETS_PATH, image_set='trainval', download=True)

### Utility Methods ###

def flatten_list(nested_list: list[Any]) -> list[list[Any]]:
    """Flattens a nested list or tuple into a single list.

    Args:
        nested_list (list[Any]): The nested list or tuple to flatten.

    Returns:
        list[Any]: The flattened list.
    """
    result = []
    for item in nested_list:
        if isinstance(item, list) or isinstance(item, tuple): result.extend(flatten_list(item))
        else: result.append(item)
    return result

def batch_list(
        list_,
        batch_size
) -> list[list[Any]]:
    """Partitions a list into sub-lists of maximum given size.

    Args:
        list_ (list[Any]): The list to partition.
        batch_size (int): The maximum size of each batch.

    Returns:
        list[list[Any]]: The list of batches.
    """
    if not isinstance(list_, list):
        list_ = list(list_)
    return [list_[i:i + batch_size] for i in range(0, len(list_), batch_size)]

def get_batch_keys_amount(list_of_dicts: list[dict]) -> int:
    """Returns the total number of keys in a list of dictionaries.

    Args:
        list_of_dicts (list[dict]): The list of dictionaries.

    Returns:
        int: The total number of keys.
    """
    return sum([len(d.keys()) for d in list_of_dicts])

def batch_class_splitted_list(
        list_of_dicts,
        max_keys_per_batch
) -> list[list[dict]]:
    """Splits a list of dictionaries into batches, each batch not exceeding a maximum number of keys.

    Args:
        list_of_dicts (list[dict]): The list of dictionaries to batch.
        max_keys_per_batch (int): The maximum number of keys per batch.

    Returns:
        list[list[dict]]: The list of batches.
    """
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

def extract_json(text: str) -> str:
    """Extracts the first JSON object found in a string.

    Args:
        text (str): The string to search.

    Returns:
        str: The extracted JSON string, or None if not found.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None

def extract_uppercase_words(text: str) -> list[str]:
    """Returns a list of all unique uppercase words from the given text.

    Args:
        text (str): The text to search.

    Returns:
        list[str]: The sorted list of unique uppercase words.
    """
    return sorted(set(re.findall(r'\b[A-Z]+\b', text)))

def parse_eval_text_to_dict(eval_str: str) -> dict | str:
    """Parses a string containing a dictionary representation and returns the dictionary.

    Args:
        eval_str (str): The string to parse.

    Returns:
        dict | str: The parsed dictionary, or the original string if parsing fails.
    """
    eval_str = extract_json(eval_str)
    try:
        eval_dict = ast.literal_eval(eval_str)
        return eval_dict
    except:
        print("Wrong parsing to dict!")
        return eval_str

def display_prompt(full_prompt: str | Prompt) -> None:
    """Displays a prompt, which can be a string or a list of strings and images, using IPython display utilities.

    Args:
        full_prompt (str | Prompt): The prompt to display.
    """
    if isinstance(full_prompt, str):
        display(Markdown(full_prompt))
    else:
        for prompt in full_prompt:
            if isinstance(prompt, Image.Image):
                display(prompt)
            else:
                display(Markdown(prompt))

def create_empty_dataarray(dims_to_coords: dict[str, list[Any]]) -> xr.DataArray:
    """Creates an empty xarray DataArray with the specified dimensions and coordinates.

    Args:
        dims_to_coords (dict[str, list[Any]]): A mapping from dimension names to coordinate lists.

    Returns:
        xr.DataArray: The created empty DataArray.
    """
    dims, coords = zip(*dims_to_coords.items())
    shape = [len(c) for c in coords]
    da = xr.DataArray(np.empty(shape, dtype=object), coords=coords, dims=dims)
    return da

def track_performance(n_trials: int = 10) -> Callable:
    """A decorator that tracks execution time and memory usage of a function over multiple trials.

    Args:
        n_trials (int): The number of trials to run.

    Returns:
        Callable: The decorated function with performance tracking.
    """
    def decorator(func: Callable) -> Callable:
        """
        A decorator that tracks the execution time, current memory usage,
        and peak memory usage of the decorated function.
        """
        def shared_prologue() -> float:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            start_time = time.perf_counter()
            return start_time
        
        def shared_epilogue(t_0: float) -> tuple[float, float, float]:
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
        async def async_wrapper(*args, **kwargs) -> tuple[Any, pd.DataFrame]:
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
        def sync_wrapper(*args, **kwargs) -> tuple[Any, pd.DataFrame]:
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
        
# TODO Fix this, it does not work, it does not actually retry!
def retry(
        max_retries: int,
        cooldown_seconds: int | float,
        exceptions_to_catch: list[Exception]
) -> Callable:
    """A decorator that retries a function when it encounters a specified set of exceptions.

    Args:
        max_retries (int): The maximum number of times to retry the function.
        cooldown_seconds (int or float): The time in seconds to wait between retries.
        exceptions_to_catch (list[Exception]): The list of exception types to catch and trigger a retry.

    Returns:
        Callable: The decorated function with retry logic.
    """
    def decorator(func) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Callable:
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
        """Returns a string representation of the object, including its dictionary attributes."""
        return " ".join([object.__repr__(self), str(self.__dict__)])

    def __getitem__(self, key: str) -> Any:
        """Gets the value associated with a key from the object's dictionary.

        Args:
            key (str): The key to retrieve.

        Returns:
            Any: The value associated with the key.

        Raises:
            KeyError: If the key does not exist.
        """
        try:
            value = self.__dict__[key]
            return value
        except KeyError:
            raise KeyError(f"'GenParams' object has no attribute '{key}'")

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets the value for a key in the object's dictionary if the key exists.

        Args:
            key (str): The key to set.
            value (Any): The value to assign.

        Raises:
            KeyError: If the key does not exist.
        """
        if key in self.__dict__.keys():
            self.__dict__[key] = value
        else:
            raise KeyError(f"'GenParams' does not have key '{key}'.")

    def get_assigned_keys(self) -> Any:
        """Returns a list of keys that are valued 'None'.

        Returns:
            list: The list of keys with non-None values.
        """
        return [key for key, value in self.__dict__.items() if value is not None]

def my_tqdm(
        data: Iterable,
        desc: str = ""
) -> tqdm:
    """Wraps an iterable with tqdm progress bar, converting to list if needed.

    Args:
        data (Iterable): The data to wrap.
        desc (str): The description for the progress bar.

    Returns:
        tqdm: The tqdm progress bar iterator.
    """
    if not isinstance(data, list):
        data = list(data)
    return tqdm(
        enumerate(data),
        total=len(data),
        desc=desc, # Add a description
        unit="item", # Specify the unit of progress
        colour="#67ad5b", # Set a vibrant color,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

def get_compute_capability() -> float:
    compute_capability = torch.cuda.get_device_capability()
    compute_capability = compute_capability[0] + 0.1*compute_capability[1]
    return compute_capability

def main() -> None:
    pass

if __name__ == "__main__":
    main()
