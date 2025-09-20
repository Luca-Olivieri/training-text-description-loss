from config import *
from path import PRIVATE_DATASETS_PATH

import torch
from torch import nn
from torchvision.datasets import VOCSegmentation
import re
import ast
import time
import functools
import tracemalloc
import asyncio
import numpy as np
import xarray as xr
import pandas as pd
from abc import ABC
from tqdm import tqdm
from typing import Any, TypeVar
import torchmetrics as tm
import json
import gc

from typing import Callable, TypeVar, Any, Iterable, Self
from abc import ABC

GenericClient = TypeVar

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

def flatten_list_to_depth(
    nested_list: list[Any] | tuple[Any],
    depth: float = float('inf')
) -> list[Any]:
    """
    Flattens a nested list or tuple to a specified depth.

    By default, it flattens the list completely.

    Args:
        nested_list (list[Any] | tuple[Any]): The nested list or tuple to flatten.
        depth (float): The maximum number of levels to flatten.
                       - `float('inf')` (default): Flattens completely.
                       - `1`: Flattens only the first level.
                       - `0`: Returns a shallow copy of the list (no flattening).

    Returns:
        list[Any]: The flattened list.

    Examples:
        >>> data = [[[1, 2], [3, 4]], [5, [6, 7]]]
        >>> flatten(data)  # Default: full flattening
        [1, 2, 3, 4, 5, 6, 7]
        >>> flatten(data, depth=1)
        [[1, 2], [3, 4], 5, [6, 7]]
        >>> flatten(data, depth=2)
        [1, 2, 3, 4, 5, [6, 7]]
        >>> flatten(data, depth=0)
        [[[1, 2], [3, 4]], [5, [6, 7]]]
    """
    result = []
    for item in nested_list:
        # Check if the item is a list/tuple and we still have depth to flatten
        if isinstance(item, (list, tuple)) and depth > 0:
            # Recurse, but with one less level of depth allowed
            result.extend(flatten_list_to_depth(item, depth - 1))
        else:
            # Append the item as is, either because it's not a list/tuple
            # or because we've reached our desired depth
            result.append(item)
    return result

def flatten_cs_jsonl(
        input_path: Path,
        output_path: Path
) -> None:
    """
    Reads a JSONL file, flattens its records, and writes to a new JSONL file.

    - The first line of the input is copied directly to the output.
    - Each subsequent line is flattened from:
      {"img_idx": int, "content": {str_key: str_val, ...}}
    - To multiple lines of:
      {"img_idx": int, "pos_class": int, "content": str_val}
    """
    print(f"--- Processing '{input_path}' and writing to '{output_path}' ---")
    
    try:
        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            # 1. Read the first line (state) and copy it as is.
            first_line = infile.readline()
            if not first_line:
                print("Warning: Input file is empty.")
                return

            outfile.write(first_line)
            
            # 2. Process the rest of the lines in the file.
            lines_processed = 0
            records_generated = 0
            for line in infile:
                # Skip any blank lines
                line = line.strip()
                if not line:
                    continue
                
                # Load the JSON object from the line
                data = json.loads(line)
                
                # Extract the common data
                img_idx = data.get("img_idx")
                content_dict = data.get("content", {}) # Use .get with default for safety
                
                # 3. Iterate through the inner 'content' dictionary to create flattened records.
                if img_idx is not None and content_dict:
                    for pos_class_str, content_str in content_dict.items():
                        
                        # Create the new, flattened dictionary
                        flattened_record = {
                            "img_idx": img_idx,
                            "pos_class": int(pos_class_str), # Convert key to integer
                            "content": content_str
                        }
                        
                        # Write the flattened dictionary as a JSON string to the output file
                        json.dump(flattened_record, outfile)
                        outfile.write('\n')
                        records_generated += 1

                lines_processed += 1
        
        print(f"Processing complete.")
        print(f"Total lines processed (excluding header): {lines_processed}")
        print(f"Total flattened records generated: {records_generated}\n")

    except FileNotFoundError:
        print(f"Error: The input file '{input_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def print_file_content(
        filename: str
) -> None:
    """Utility function to print the content of a file."""
    if not os.path.exists(filename):
        print(f"File '{filename}' does not exist.")
        return
        
    print(f"--- Contents of '{filename}' ---")
    with open(filename, 'r') as f:
        for line in f:
            print(line, end='')
    print("--- End of file ---\n")

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

def pretty_metrics(
        metric_collection: tm.MetricCollection
) -> dict:
    return {m: f"{s.item():.4f}" for m, s in metric_collection.items()}

# TODO adapt the registry to take note of the outer class and cast the model to it to have type annotations.
class Registry:
    """
    A class to manage and instantiate registered objects.
    """
    def __init__(self) -> None:
        """Initializes the registry."""
        self._registry = {}

    def register(
            self,
            name: str
    ) -> Callable:
        """
        A decorator to register a new object.

        Args:
            name (str): The name to register the object under.
        """
        def decorator(cls: Self):
            name_lower = name.lower()
            # if name_lower in self._registry:
                # print(f"Warning: Model '{name}' is already registered. Overwriting.")
            self._registry[name_lower] = cls
            return cls
        return decorator

    def get(
            self,
            name: str,
            **kwargs
    ) -> object:
        """
        Gets and instantiates an objects from the registry.

        Args:
            name (str): The name of the object.
            **kwargs: Arbitrary keyword arguments to pass to the object's constructor.

        Returns:
            An instance of the requested object.

        Raises:
            ValueError: If the object name is not found in the registry.
        """
        name_lower = name.lower()
        if name_lower not in self._registry:
            raise ValueError(
                f"Error: object '{name}' not found. "
                f"Available objects: {list(self._registry.keys())}"
            )
        
        # Retrieve the class from the registry
        model_class = self._registry[name_lower]
        
        # Instantiate and return the object
        return model_class(**kwargs)

    def registered_objects(self) -> list:
        """Returns a list of registered object names."""
        return list(self._registry.keys())

def map_tensor(
        input_tensor: torch.Tensor,
        mapping_dict: dict[int, int]
) -> torch.Tensor:
    # Find the maximum key in your dictionary to determine the size of the lookup tensor
    max_key = max(mapping_dict.keys()) if mapping_dict else 0
    max_val = max(mapping_dict.values()) if mapping_dict else 0 # For setting default value dtype

    # Create a lookup tensor
    # Initialize with a default value for unmapped elements (e.g., original value, 0)
    # Make sure the dtype is appropriate for your mapped values.
    lookup_tensor = torch.full((max_key + 1,), 0, dtype=input_tensor.dtype, device=input_tensor.device) # Or original_tensor.dtype

    # Populate the lookup tensor
    for key, value in mapping_dict.items():
        lookup_tensor[key] = value

    # If you have values in your original tensor that are not in the mapping_dict,
    # you might want to handle them. For example, keep their original value.
    # First, create a tensor that would contain the original values as a fallback.
    # Then use torch.where to apply the mapping.

    # Apply the mapping
    mapped_tensor = lookup_tensor[input_tensor.long()].to(input_tensor.dtype)

    # Handle values not present in the mapping_dict (if lookup_tensor was initialized with a distinct default)
    # For example, if -1 indicates an unmapped value and you want to keep the original.
    # mapped_tensor = torch.where(mapped_tensor == -1, original_tensor, mapped_tensor)
    return mapped_tensor

def is_list_of_tensors(item_to_check: Any) -> bool:
    """
    Checks if an item is a list containing only PyTorch tensors.

    This function is optimized for speed by first checking if the item is a list
    and then using a short-circuiting generator expression with `all()`.

    Args:
        item_to_check: The item to be checked.

    Returns:
        True if the item is a list and all its elements are torch.Tensor.
        False otherwise.
    """
    # 1. Quick type check: If it's not a list, it can't be a list of tensors.
    #    This is a fast first-pass filter.
    if not isinstance(item_to_check, list):
        return False

    # 2. Use `all()` with a generator expression.
    #    - `all()` is fast and short-circuits: it stops and returns False
    #      as soon as it finds the first non-tensor element.
    #    - `torch.is_tensor()` is the canonical way to check for a tensor.
    return all(torch.is_tensor(x) for x in item_to_check)

def blend_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Blends two tensors using torch.lerp for a potentially more optimized approach.
    
    torch.lerp(start, end, weight) is equivalent to: start + weight * (end - start)
    which is algebraically the same as: start * (1 - weight) + end * weight
    """
    blended_tensor = torch.lerp(tensor1.float(), tensor2.float(), alpha)
    out_tensor = torch.clamp(blended_tensor, 0, 255).to(torch.uint8)
    return out_tensor

def compile_torch_model(model: torch.nn.Module):
    if get_compute_capability() >= 7.0:
        model = torch.compile(model)
    return model

def create_directory(
        parent_path: Path,
        folder_name: str
) -> Path:
    # Create the directory, including any necessary parents.
    # The 'exist_ok=True' argument prevents an error if the directory already exists.
    (parent_path / folder_name).mkdir(parents=False, exist_ok=True)
    return parent_path / folder_name

def clear_memory(
        ram: bool = True,
        gpu: bool = True,
) -> None:
    gc.collect() if ram else None
    try:
        torch.cuda.empty_cache() if gpu else None
    except MemoryError:
        ...

def get_activation(
        name: str,
        activations: dict[str, torch.Tensor]
) -> Callable:
    """
    This function returns another function (a hook) that will be registered
    to a layer. The hook will save the output of the layer in the
    `activation` dictionary.
    """
    def hook(
            model: nn.Module,
            input: torch.Tensor,
            output: torch.Tensor,
    ) -> None:
        activations[name] = output
    return hook

def subsample_sign_classes(
        data_list: list,
        k: int
) -> list:
    if len(data_list) > k:
        if 0 in data_list and data_list != [0]:
            data_list.remove(0)
        if len(data_list) > k:
            return random.sample(data_list, k)
    return data_list

def nanstd(
        data: torch.Tensor,
        dim: list[int] | int,
        keepdim: bool =False
) -> None:
    result = torch.sqrt(
        torch.nanmean(
            torch.pow(torch.abs(data-torch.nanmean(data, dim=dim).unsqueeze(dim)), 2),
            dim=dim
        )
    )
    if keepdim:
        result = result.unsqueeze(dim)
    return result

def main() -> None:
    ...

if __name__ == "__main__":
    main()
