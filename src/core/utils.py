"""
Utility module for the core package.

This module provides various utility functions and classes for data processing,
performance tracking, text manipulation, and negative text generation. It includes:

- JSON data processing and flattening utilities
- Text parsing and extraction functions
- Performance tracking decorators for measuring execution time and memory usage
- Retry decorator for handling transient failures
- Data subsampling utilities
- NegativeTextGenerator class for creating text negatives through word substitution

The module is designed to support machine learning workflows, particularly for
semantic segmentation and vision-language tasks.
"""

from core.config import *

import re
import ast
import time
import functools
import tracemalloc
import asyncio
import numpy as np
import xarray as xr
import pandas as pd
import json

from core._types import Callable, Any, deprecated, Optional

@deprecated("Should be split into two methods, one flattening and one writing.")
def flatten_cs_jsonl(
        input_path: Path,
        output_path: Path
) -> None:
    """
    Reads a JSONL file, flattens its records, and writes to a new JSONL file.

    This function processes a JSONL file where each line (after the first) contains
    an image index and a nested content dictionary. The nested structure is flattened
    such that each key-value pair in the content dictionary becomes a separate record.

    The first line of the input file (typically containing state/metadata) is copied
    directly to the output without modification.

    Args:
        input_path (Path): Path to the input JSONL file to be processed.
        output_path (Path): Path where the flattened JSONL file will be written.

    Input format (lines after the first):
        {"img_idx": int, "content": {str_key: str_val, ...}}

    Output format (lines after the first):
        {"img_idx": int, "pos_class": int, "content": str_val}

    Returns:
        None: Writes the output directly to a file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        Exception: For any other unexpected errors during processing.

    Note:
        This function is deprecated and should be split into separate flattening
        and writing methods for better modularity.
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

def extract_json(text: str) -> str | None:
    """
    Extracts the first JSON object found in a string.

    Searches for the first occurrence of a JSON object (enclosed in curly braces)
    within the given text using regex pattern matching. Handles multiline JSON objects.

    Args:
        text (str): The string to search for a JSON object.

    Returns:
        str | None: The extracted JSON string if found, None otherwise.

    Example:
        >>> text = "Some text before {\"key\": \"value\"} and after"
        >>> extract_json(text)
        '{"key": "value"}'
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None

def extract_uppercase_words(text: str) -> list[str]:
    """
    Extracts all unique uppercase words from the given text.

    Finds all words that consist entirely of uppercase letters (A-Z) and returns
    them as a sorted list of unique values. Word boundaries are respected.

    Args:
        text (str): The text to search for uppercase words.

    Returns:
        list[str]: A sorted list of unique uppercase words found in the text.

    Example:
        >>> extract_uppercase_words("The CAT and DOG are ANIMALS, CAT is small")
        ['ANIMALS', 'CAT', 'DOG']
    """
    return sorted(set(re.findall(r'\b[A-Z]+\b', text)))

def parse_text_to_dict(text: str) -> dict | str:
    """
    Parses a string containing a dictionary representation and returns the dictionary.

    Attempts to extract a JSON object from the text and parse it into a Python dictionary
    using ast.literal_eval for safe evaluation. If parsing fails, returns the original
    text string.

    Args:
        text (str): The string containing a dictionary representation to parse.

    Returns:
        dict | str: The parsed dictionary if successful, otherwise the original string.

    Note:
        Prints "Wrong parsing to dict!" to stdout if parsing fails.

    Example:
        >>> parse_text_to_dict('{"key": "value"}')
        {'key': 'value'}
        >>> parse_text_to_dict('not a dict')
        'not a dict'
    """
    text = extract_json(text)
    try:
        eval_dict = ast.literal_eval(text)
        return eval_dict
    except:
        print("Wrong parsing to dict!")
        return text

def create_empty_dataarray(dims_to_coords: dict[str, list[Any]]) -> xr.DataArray:
    """
    Creates an empty xarray DataArray with the specified dimensions and coordinates.

    Constructs a multi-dimensional array with shape determined by the length of each
    coordinate list. The array is initialized with empty object dtype values.

    Args:
        dims_to_coords (dict[str, list[Any]]): A mapping from dimension names to their
            coordinate lists. The length of each list determines the size of that dimension.

    Returns:
        xr.DataArray: An empty DataArray with the specified dimensions and coordinates,
            using object dtype.

    Example:
        >>> dims = {"trial": [0, 1, 2], "metric": ["time", "memory"]}
        >>> da = create_empty_dataarray(dims)
        >>> da.shape
        (3, 2)
    """
    dims, coords = zip(*dims_to_coords.items())
    shape = [len(c) for c in coords]
    da = xr.DataArray(np.empty(shape, dtype=object), coords=coords, dims=dims)
    return da

def track_performance(n_trials: int = 10) -> Callable:
    """
    A decorator factory that tracks execution time and memory usage over multiple trials.

    Creates a decorator that runs the target function multiple times and collects
    performance metrics (execution time, current memory, peak memory) for each trial.
    Works with both synchronous and asynchronous functions.

    Args:
        n_trials (int, optional): The number of trials to run. Defaults to 10.

    Returns:
        Callable: A decorator that wraps the target function with performance tracking.

    The decorated function returns:
        tuple[Any, pd.DataFrame]: A tuple containing:
            - The result of the function from the last trial
            - A pandas DataFrame with performance metrics for all trials

    Note:
        Uses tracemalloc for memory profiling. Memory values are returned in MB.
        Execution time is measured using time.perf_counter() in seconds.

    Example:
        >>> @track_performance(n_trials=5)
        ... def my_function():
        ...     return sum(range(1000000))
        >>> result, perf_data = my_function()
    """
    def decorator(func: Callable) -> Callable:
        """
        A decorator that tracks the execution time, current memory usage,
        and peak memory usage of the decorated function.
        """
        def shared_prologue() -> float:
            """
            Starts performance tracking by initializing tracemalloc and recording start time.

            Returns:
                float: The start time in seconds (from perf_counter).
            """
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            start_time = time.perf_counter()
            return start_time
        
        def shared_epilogue(t_0: float) -> tuple[float, float, float]:
            """
            Completes performance tracking and computes metrics.

            Args:
                t_0 (float): The start time from shared_prologue.

            Returns:
                tuple[float, float, float]: A tuple containing:
                    - elapsed_time: Execution time in seconds
                    - current_memory: Current memory usage in MB
                    - peak_memory: Peak memory usage in MB
            """
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
            """
            Async wrapper for coroutine functions with performance tracking.

            Args:
                *args: Positional arguments to pass to the wrapped function.
                **kwargs: Keyword arguments to pass to the wrapped function.

            Returns:
                tuple[Any, pd.DataFrame]: The function result and performance metrics DataFrame.
            """
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
            """
            Synchronous wrapper for regular functions with performance tracking.

            Args:
                *args: Positional arguments to pass to the wrapped function.
                **kwargs: Keyword arguments to pass to the wrapped function.

            Returns:
                tuple[Any, pd.DataFrame]: The function result and performance metrics DataFrame.
            """
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
        cooldown_period: float,
        exceptions: list[BaseException]
) -> Callable:
    """
    A decorator factory to retry a function if it raises specified exceptions.

    Creates a decorator that automatically retries a function when it raises one of
    the specified exception types. The decorator is async-aware and works correctly
    with both synchronous and asynchronous functions.

    Args:
        max_retries (int): The maximum number of retry attempts.
        cooldown_period (float): The number of seconds to wait between retry attempts.
        exceptions (list[BaseException]): A list of exception types that should trigger
            a retry. If the function raises an exception not in this list, it will
            propagate immediately without retrying.

    Returns:
        Callable: A decorator that wraps the target function with retry logic.

    Raises:
        The last exception caught if all retry attempts are exhausted.

    Note:
        Progress messages are printed to stdout for each retry attempt.

    Example:
        >>> @retry(max_retries=3, cooldown_period=1.0, exceptions=[ConnectionError])
        ... def fetch_data():
        ...     # May raise ConnectionError
        ...     return download()
    """
    exceptions_to_catch = tuple(exceptions)

    def decorator(func):
        """
        The actual decorator that wraps the function with retry logic.

        Args:
            func (Callable): The function to be decorated.

        Returns:
            Callable: The wrapped function with retry capabilities.
        """
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            """
            Async wrapper for coroutine functions with retry logic.

            Args:
                *args: Positional arguments to pass to the wrapped function.
                **kwargs: Keyword arguments to pass to the wrapped function.

            Returns:
                The return value of the wrapped function.

            Raises:
                The last exception caught after all retries are exhausted.
            """
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions_to_catch as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1}/{max_retries} failed with {type(e).__name__}: {e}.")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {cooldown_period} seconds...")
                        await asyncio.sleep(cooldown_period) # Use asyncio.sleep for async
            
            print(f"All {max_retries} retries failed. Raising the last exception.")
            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            """
            Synchronous wrapper for regular functions with retry logic.

            Args:
                *args: Positional arguments to pass to the wrapped function.
                **kwargs: Keyword arguments to pass to the wrapped function.

            Returns:
                The return value of the wrapped function.

            Raises:
                The last exception caught after all retries are exhausted.
            """
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions_to_catch as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1}/{max_retries} failed with {type(e).__name__}: {e}.")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {cooldown_period} seconds...")
                        time.sleep(cooldown_period) # Use time.sleep for sync
            
            print(f"All {max_retries} retries failed. Raising the last exception.")
            raise last_exception

        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

def subsample_sign_classes(
        data_list: list,
        k: int
) -> list:
    """
    Subsamples a list to at most k elements, with special handling for zero.

    If the list has more than k elements, removes 0 (if present and not the only element)
    and then randomly samples k elements. If the list still exceeds k elements after
    removing 0, performs random sampling.

    Args:
        data_list (list): The list of elements to subsample.
        k (int): The maximum number of elements to keep.

    Returns:
        list: The subsampled list with at most k elements.

    Note:
        The function modifies the input list in place by potentially removing 0.
        Uses random.sample for sampling without replacement.

    Example:
        >>> subsample_sign_classes([0, 1, 2, 3, 4, 5], k=3)
        [1, 3, 5]  # Example output, actual values depend on random sampling
    """
    if len(data_list) > k:
        if 0 in data_list and data_list != [0]:
            data_list.remove(0)
        if len(data_list) > k:
            return random.sample(data_list, k)
    return data_list


class NegativeTextGenerator:
    """
    Generates text negatives by substituting words based on user-defined word pools.

    This class creates negative text examples by replacing words in a positive text
    with alternatives from semantically related word pools. It's useful for data
    augmentation in NLP tasks, particularly for generating contrastive examples.

    The generator uses simple string replacement to handle hyphenated words and
    requires no external dependencies beyond the standard library.

    Attributes:
        word_to_pool_map (dict[str, dict]): A reverse lookup map from each word to
            its pool name and sibling words (alternatives within the same pool).

    Example:
        >>> pools = {
        ...     "colors": ["red", "blue", "green"],
        ...     "sizes": ["big", "small", "tiny"]
        ... }
        >>> gen = NegativeTextGenerator(word_pools=pools)
        >>> negatives = gen.generate("The red big ball", num_negatives=2)
        >>> # You can also override the 'classes' pool for a single run
        >>> voc_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle"]
        >>> negatives_custom_cls = gen.generate(
        ...     "A picture of a BIRD",
        ...     num_negatives=2,
        ...     classes=voc_classes
        ... )
    """
    def __init__(
            self,
            word_pools: dict[str, list[str]]
    ) -> None:
        """
        Initializes the generator and pre-processes the word pools for fast lookups.

        Creates a reverse lookup map that allows efficient retrieval of alternative
        words (siblings) for any given word. Only pools with at least 2 words are
        included in the lookup map.

        Args:
            word_pools (dict[str, list[str]]): A dictionary where keys are pool names
                and values are lists of words in that pool. Words in the same pool
                are considered alternatives to each other.

        Note:
            Prints a summary of initialization to stdout, including the number of
            words and pools processed.
        """
        self.word_to_pool_map = self._build_lookup(word_pools)
        print(f"NegativeGenerator initialized. Found {len(self.word_to_pool_map)} words across {len(word_pools)} pools.")

    def _build_lookup(
            self,
            word_pools: dict[str, list[str]]
    ) -> dict[str, dict]:
        """
        Creates a fast reverse-lookup map from a word to its pool and siblings.

        For each word in each pool, creates a dictionary entry mapping the lowercase
        version of the word to its pool name and all other words (siblings) in that pool.
        Pools with fewer than 2 words are skipped.

        Args:
            word_pools (dict[str, list[str]]): The word pools to process.

        Returns:
            dict[str, dict]: A mapping where each key is a lowercase word and the value
                is a dict containing:
                    - 'pool_name': The name of the pool the word belongs to
                    - 'siblings': A list of alternative words from the same pool
        """
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
        A simple tokenizer that splits by whitespace and separates punctuation.

        Separates hyphens and common punctuation marks (commas, semicolons, colons,
        periods, exclamation marks, question marks) into individual tokens by padding
        them with spaces before splitting on whitespace.

        Args:
            text (str): The text to tokenize.

        Returns:
            list[str]: A list of tokens including words and separated punctuation marks.

        Example:
            >>> self._tokenize("Hello-world, how are you?")
            ['Hello', '-', 'world', ',', 'how', 'are', 'you', '?']
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
        Reconstructs text from tokens, correctly rejoining hyphenated words and punctuation.

        Reverses the tokenization process by joining tokens with spaces and then
        removing spaces around hyphens and punctuation marks (commas, semicolons,
        colons, periods, exclamation marks, question marks).

        Args:
            tokens (list[str]): The list of tokens to reconstruct into text.

        Returns:
            str: The reconstructed text with proper spacing and punctuation.

        Example:
            >>> self._reconstruct_text(['Hello', '-', 'world', ',', 'how', 'are', 'you', '?'])
            'Hello-world, how are you?'
        """
        text = " ".join(tokens)
        # Reverse the hyphen padding and clean up other common punctuation
        text = text.replace(' - ', '-')
        text = text.replace(' ,', ',').replace(' ;', ';').replace(' :', ':')
        text = text.replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        # text.strip()
        return text

    def _backup_perturbation_by_shuffle(
            self,
            tokens: list[str]
    ) -> str:
        """Backup method to shuffle tokens."""
        if len(tokens) < 2:
            return " ".join(tokens)
        
        original_text = self._reconstruct_text(tokens)
        shuffled_tokens = list(tokens)
        
        # Keep shuffling until it's different, for up to 10 tries.
        max_shuffles = 10
        count = 0
        while self._reconstruct_text(shuffled_tokens) == original_text and count < max_shuffles:
            random.shuffle(shuffled_tokens)
            count += 1
        
        return self._reconstruct_text(shuffled_tokens)
    
    def _backup_perturbation_by_removal(
        self,
        tokens: list[str]
    ) -> str:
        """
        Third-stage backup method to remove a random number of tokens.

        This method removes a number of words, `k`, where `k` is randomly chosen
        to be between 1 and half the total number of words in the sentence. The
        words are removed from random, non-adjacent positions.

        Args:
            tokens (list[str]): The list of tokens to modify.

        Returns:
            str: The reconstructed text with tokens removed.
        """
        n = len(tokens)
        if n < 2:
            # Cannot remove 1 to n//2 words if there are fewer than 2.
            return ""

        # Determine how many words to remove: from 1 up to half the total words.
        max_to_remove = n // 2
        k = random.randint(1, max_to_remove)
        
        # Choose k unique indices to remove from the list of tokens.
        indices_to_remove = set(random.sample(range(n), k))
        
        # Build a new list of tokens, keeping only those whose indices were not selected for removal.
        new_tokens = [
            token for i, token in enumerate(tokens) 
            if i not in indices_to_remove
        ]

        return self._reconstruct_text(new_tokens)

    def generate(
        self,
        positive_text: str,
        num_negatives: int = 1,
        change_probability: float = 0.5,
        classes: Optional[list[str]] = None
    ) -> list[str]:
        """
        Generates a batch of unique negative texts from a single positive text.

        This method GUARANTEES to return a list of the requested `num_negatives` length.

        It uses a three-stage fallback process:
        1. Primary: Substitutes words using the provided word pools.
        2. Backup: Shuffles the order of words.
        3. Final Fallback: Removes a random number of words (1 to 50%) from the text.

        If all methods fail to generate enough unique negatives, the returned list
        will be padded with empty strings ('') to meet the requested length.

        Args:
            positive_text (str): The original positive text to generate negatives from.
            num_negatives (int, optional): The number of unique negative examples to
                generate. Defaults to 1.
            change_probability (float, optional): The probability (0.0-1.0) of changing
                each additional substitutable word beyond the first forced change.
                Defaults to 0.5.
            classes (list[str] | None, optional): A list of class names to temporarily
                override the 'classes' word pool for this generation run. If None,
                the default 'classes' pool from initialization is used. Defaults to None.

        Returns:
            list[str]: A list of negative text examples of length `num_negatives`.
        """
        # Start with the instance's default map. It will be replaced if `classes` are overridden.
        current_word_to_pool_map = self.word_to_pool_map

        if classes is not None:
            # Create a temporary copy of the map to modify for this run only.
            temp_map = {k: v.copy() for k, v in self.word_to_pool_map.items()}
            
            # Remove any existing words belonging to the 'classes' pool.
            keys_to_remove = [
                word for word, data in temp_map.items()
                if data.get('pool_name') == 'classes'
            ]
            for key in keys_to_remove:
                del temp_map[key]

            # Build and add the new 'classes' pool from the provided list.
            if len(classes) >= 2:
                for i, word in enumerate(classes):
                    siblings = classes[:i] + classes[i+1:]
                    temp_map[word.lower()] = {
                        'pool_name': 'classes',
                        'siblings': siblings
                    }
            
            current_word_to_pool_map = temp_map

        original_tokens = self._tokenize(positive_text)
        candidate_indices = [
            i for i, token in enumerate(original_tokens)
            if token.lower() in current_word_to_pool_map
        ]
        
        generated_negatives = set()
        
        # --- Stage 1: Primary Generation (Word Substitution) ---
        if candidate_indices:
            max_attempts = num_negatives * 20 + 10
            attempts = 0
            while len(generated_negatives) < num_negatives and attempts < max_attempts:
                tokens_to_modify = list(original_tokens)
                indices_to_consider = list(candidate_indices)
                
                # This check is crucial in case the override removed all candidate words.
                if not indices_to_consider:
                    break

                forced_choice_idx = random.choice(indices_to_consider)
                word_to_replace = tokens_to_modify[forced_choice_idx].lower()
                siblings = current_word_to_pool_map[word_to_replace]['siblings']
                tokens_to_modify[forced_choice_idx] = random.choice(siblings)
                indices_to_consider.remove(forced_choice_idx)
                for idx in indices_to_consider:
                    if random.random() < change_probability:
                        word_to_replace = tokens_to_modify[idx].lower()
                        siblings = current_word_to_pool_map[word_to_replace]['siblings']
                        tokens_to_modify[idx] = random.choice(siblings)
                reconstructed_text = self._reconstruct_text(tokens_to_modify)
                if reconstructed_text != positive_text:
                    generated_negatives.add(reconstructed_text)
                attempts += 1

        # --- Stage 2: Backup Generation (Token Shuffling) ---
        if len(generated_negatives) < num_negatives:
            needed = num_negatives - len(generated_negatives)
            max_backup_attempts = needed * 20 + 10 
            backup_attempts = 0
            while len(generated_negatives) < num_negatives and backup_attempts < max_backup_attempts:
                shuffled = self._backup_perturbation_by_shuffle(original_tokens)
                if shuffled != positive_text:
                    generated_negatives.add(shuffled)
                backup_attempts += 1
        
        # --- Stage 3: Final Fallback (Word Removal) ---
        if len(generated_negatives) < num_negatives:
            needed = num_negatives - len(generated_negatives)
            max_removal_attempts = needed * len(original_tokens) + 10
            removal_attempts = 0
            while len(generated_negatives) < num_negatives and removal_attempts < max_removal_attempts:
                removed = self._backup_perturbation_by_removal(original_tokens)
                if removed != positive_text:
                    generated_negatives.add(removed)
                removal_attempts += 1

        # --- Final Assembly and Padding ---
        final_negatives = list(generated_negatives)

        # If we are still short, pad with empty strings
        if len(final_negatives) < num_negatives:
            slack = num_negatives - len(final_negatives)
            print(f"Warning: Could only generate {len(final_negatives)} unique negatives. Padding with {slack} empty string(s) to meet request.")
            final_negatives.extend([' '] * slack)
        
        # Return a list of exactly the requested size
        return final_negatives[:num_negatives]


diff_text_word_pools = {
    """
    Predefined word pools for generating negative text examples in segmentation tasks.

    This dictionary contains semantically related word groups used by NegativeTextGenerator
    to create contrastive text examples for semantic segmentation evaluation. Each pool
    contains words that are semantically similar and can be substituted for each other
    to create meaningful but incorrect variations.

    The pools cover various aspects of segmentation quality and object properties:
    - Spatial/positional terms (top, bottom, left, right, etc.)
    - Colors and visual properties
    - Size descriptors (absolute and relative)
    - Quality descriptors (precise, rough, clean, etc.)
    - Segmentation-specific terms (oversegmented, undersegmented, etc.)
    - Object class names from PASCAL VOC dataset

    These pools are specifically designed for evaluating vision-language models on
    segmentation tasks where nuanced language understanding is critical.
    """
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
    """
    Demonstrates the usage of NegativeTextGenerator with example text.

    This function serves as a test/demo for the NegativeTextGenerator class,
    showing how to create an instance with predefined word pools and generate
    negative text examples from a sample segmentation evaluation description.

    The example uses diff_text_word_pools and generates 32 unique negative
    variations of a ground truth evaluation statement.

    Returns:
        None: Prints the generated negative examples to stdout.
    """
    # 1. Define your word pools

    # 2. Create an instance of the generator
    neg_generator = NegativeTextGenerator(word_pools=diff_text_word_pools)
    print("-" * 30)

    answer = "The ground truth PERSON region is almost entirely missed by the prediction. The prediction fails to identify any of the regions which are classified as PERSON in the ground truth mask. The prediction is mainly black, indicating the model is classifying all regions as unlabelled classes."
    print(f"ANSWER: {repr(answer)}")
    neg_answer = neg_generator.generate(answer, num_negatives=32, classes=["UNLABELLED", "BACKGROUND", "PERSON", "AAA", "BBB", "CCC", "DDD"])
    print("\nNEGATIVES:")
    for neg in neg_answer:
        print(f"  - {repr(neg)}")
    print("-" * 30)

if __name__ == "__main__":
    try_NegativeTextGenerator()
