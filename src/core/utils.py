from core.config import *

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
import json
import math

from core._types import Callable, TypeVar, Any, Any, TypeVar, ABC, deprecated

@deprecated("Should be split into two methods, one flattening and one writing.")
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

def parse_str_to_dict(text: str) -> dict | str:
    """Parses a string containing a dictionary representation and returns the dictionary.

    Args:
        text (str): The string to parse.

    Returns:
        dict | str: The parsed dictionary, or the original string if parsing fails.
    """
    text = extract_json(text)
    try:
        eval_dict = ast.literal_eval(text)
        return eval_dict
    except:
        print("Wrong parsing to dict!")
        return text

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
        
# NOTE It does not worked when applied to Google API Studio, likely for a problem of Error type.
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

class NegativeTextGenerator:
    """
    Generates text negatives by substituting words based on user-defined pools.
    This version uses a simple string replacement method to handle hyphenated words
    and has no external dependencies.
    """
    def __init__(
            self,
            word_pools: dict[str, list[str]]
    ) -> None:
        """
        Initializes the generator and pre-processes the word pools for fast lookups.

        Args:
            word_pools (Dict[str, List[str]]): A dictionary where keys are pool names
                and values are lists of words in that pool.
        """
        self.word_to_pool_map = self._build_lookup(word_pools)
        print(f"NegativeGenerator initialized. Found {len(self.word_to_pool_map)} words across {len(word_pools)} pools.")

    def _build_lookup(
            self,
            word_pools: dict[str, list[str]]
    ) -> dict[str, dict]:
        """Creates a fast reverse-lookup map from a word to its pool and siblings."""
        lookup_map = {}
        for pool_name, words_in_pool in word_pools.items():
            if len(words_in_pool) < 2:
                continue
            for i, word in enumerate(words_in_pool):
                siblings = words_in_pool[:i] + words_in_pool[i+1:]
                lookup_map[word.lower()] = {
                    'pool_name': pool_name,
                    'siblings': siblings
                }
        return lookup_map
    
    def _tokenize(
            self,
            text: str
    ) -> list[str]:
        """
        A simple tokenizer that splits by whitespace and also separates hyphens.
        """
        # Pad hyphens with spaces, so they become separate tokens after splitting
        text = text.replace('-', ' - ')
        text = text.replace(',', ' ,').replace(';', ' ;').replace(':', ' :')
        text = text.replace('.', ' .').replace('!', ' !').replace('?', ' ?')
        return text.split(' ')

    def _reconstruct_text(
            self,
            tokens: list[str]
    ) -> str:
        """
        Reconstructs text from tokens, correctly rejoining hyphenated words.
        """
        text = " ".join(tokens)
        # Reverse the hyphen padding and clean up other common punctuation
        text = text.replace(' - ', '-')
        text = text.replace(' ,', ',').replace(' ;', ';').replace(' :', ':')
        text = text.replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        # text.strip()
        return text

    def _backup_perturbation(
            self,
            tokens: list[str]
    ) -> str:
        """
        Backup method to shuffle tokens if no substitutions are possible.
        """
        if len(tokens) < 2:
            return " ".join(tokens)
        
        shuffled_tokens = list(tokens)
        while shuffled_tokens == tokens:
            random.shuffle(shuffled_tokens)
        
        # Reconstruct the shuffled text to handle any hyphens correctly
        return self._reconstruct_text(shuffled_tokens)

    def generate(
        self,
        positive_text: str,
        num_negatives: int = 1,
        change_probability: float = 0.5
    ) -> list[str]:
        """
        Generates a batch of unique negative texts from a single positive text.
        """
        # --- MODIFICATION: Use our simple custom tokenizer ---
        original_tokens = self._tokenize(positive_text)
        
        # Identify candidate indices based on the lowercase version of tokens
        candidate_indices = [
            i for i, token in enumerate(original_tokens)
            if token.lower() in self.word_to_pool_map
        ]

        # Case 1: No substitutable words found, use backup
        if not candidate_indices:
            generated_negatives = set()
            max_possible_permutations = math.factorial(len(original_tokens))
            num_to_generate = min(num_negatives, max_possible_permutations)
            max_attempts = num_to_generate * 20 + 10
            attempts = 0

            while len(generated_negatives) < num_to_generate and attempts < max_attempts:
                generated_negatives.add(self._backup_perturbation(original_tokens))
                attempts += 1
            
            return list(generated_negatives)

        # Case 2: Substitutable words exist
        generated_negatives = set()
        max_attempts = num_negatives * 20 + 10
        attempts = 0

        while len(generated_negatives) < num_negatives and attempts < max_attempts:
            tokens_to_modify = list(original_tokens)
            
            indices_to_consider = list(candidate_indices)
            
            # 1. Force one change
            forced_choice_idx = random.choice(indices_to_consider)
            word_to_replace = tokens_to_modify[forced_choice_idx].lower()
            siblings = self.word_to_pool_map[word_to_replace]['siblings']
            tokens_to_modify[forced_choice_idx] = random.choice(siblings)
            indices_to_consider.remove(forced_choice_idx)
            
            # 2. Change others based on probability
            for idx in indices_to_consider:
                if random.random() < change_probability:
                    word_to_replace = tokens_to_modify[idx].lower()
                    siblings = self.word_to_pool_map[word_to_replace]['siblings']
                    tokens_to_modify[idx] = random.choice(siblings)
            
            # Reconstruct the text using our custom method
            reconstructed_text = self._reconstruct_text(tokens_to_modify)
            generated_negatives.add(reconstructed_text)
            attempts += 1

        if attempts >= max_attempts and len(generated_negatives) < num_negatives:
            print(f"Warning: Could only generate {len(generated_negatives)} unique negatives out of the requested {num_negatives}.")

        return list(generated_negatives)

diff_text_word_pools = {
    "positional": ["top", "bottom", "left", "right", "middle", "center"],
    "positional_2": ["downwards", "upwards"],
    "colors": ["red", "blue", "green", "yellow"],
    "black_white": ["black", "white"],
    "size": ["big", "large", "small", "tiny", "huge"],
    "signifancy": ["substantial", "significant", "insignificant", "negligible"],
    "size_relative": ["bigger", "larger", "smaller", "tinier"],
    "quality": ["good", "bad", "great", "poor", "excellent", "wrong", "flawed", "flawless"],
    "quality_detailed": ["precise", "imprecise", "regular", "irregular", "complete", "incomplete", "noisy", "chaotic", "rough", "defined", "undefined", "clean", "coarse", "detailed", "harsh", "sharp"],
    "quality_detailed_adverb": ["precisely", "imprecisely", "regularly", "irregularly", "completely", "incompletely", "noisily", "chaotically", "roughly", "definedly", "undefinedly", "cleanly", "coarsly", "Detailedly", "harshly", "sharply"],
    "quality_detailed_adverb_relative": ["rougher", "cleaner", "coarser", "sharper"],
    "quantity_relative": ["more", "less"],
    "quality_adverb": ["well", "badly", "greatly", "poorly", "excellently", "flawlessly"],
    "quality_relative": ["better", "worse"],
    "position_relative": ["above", "below", "beside", "near", "nearby", "around", "along"],
    "area": ["boundary", "border", "interior", "center"],
    "areas": ["boundaries", "edges", "borders", "interiors", "inners"],
    "lower": ["lower", "elevated"],
    "level": ["over", "under"],
    "segmented": ["oversegmented", "undersegmented", "missegmented", "overextended", "underextended", "hallucinated", "extended", "missed", "hallucinated", "overspilled"],
    "segmentation": ["oversegmentation", "undersegmentation", "overspill", "hallucination", "overextension", "underextension"],
    "classes": ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT", "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE", "DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP", "SOFA", "TRAIN", "TVMONITOR", "UNLABELLED"],
}

def try_NegativeTextGenerator() -> None:
    # 1. Define your word pools

    # 2. Create an instance of the generator
    neg_generator = NegativeTextGenerator(word_pools=diff_text_word_pools)
    print("-" * 30)

    answer = "The ground truth PERSON region is almost entirely missed by the prediction. The prediction fails to identify any of the regions which are classified as PERSON in the ground truth mask. The prediction is mainly black, indicating the model is classifying all regions as unlabelled classes."
    print(f"ANSWER: {repr(answer)}")
    neg_answer = neg_generator.generate(answer, num_negatives=32)
    print("\nGenerated 3 negatives (using backup word shuffling):")
    for neg in neg_answer:
        print(f"  - {repr(neg)}")
    print("-" * 30)

if __name__ == "__main__":
    try_NegativeTextGenerator()
