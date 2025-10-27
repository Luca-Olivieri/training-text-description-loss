"""Data utility functions for list, dictionary, and batch operations.

This module provides a comprehensive set of utility functions for manipulating
and processing data structures commonly used in machine learning pipelines,
including:

- List operations: flattening, batching, scattering, and unflattening
- Dictionary operations: batching and key counting
- Class-splitted dictionary operations: flattening and unflattening
- Batch processing with custom dispatch functions
- Directory creation utilities

Type Variables:
    T: Generic type variable for items in collections
    R: Generic return type variable
    K: Generic key type for dictionaries
    V: Generic value type for dictionaries
"""

from core.config import *
from pathlib import Path

from core._types import T, R, Any, K, V, ClassSplitted, Callable

def create_directory(
        parent_path: Path,
        folder_name: str
) -> Path:
    """Creates a directory at the specified path.

    Creates the directory and all necessary parent directories. If the directory
    already exists, no error is raised.

    Args:
        parent_path: The parent directory path where the new folder will be created.
        folder_name: The name of the folder to create.

    Returns:
        The full path to the created directory.

    Example:
        >>> from pathlib import Path
        >>> path = create_directory(Path("/home/user"), "my_folder")
        >>> print(path)
        /home/user/my_folder
    """
    # Create the directory, including any necessary parents.
    # The 'exist_ok=True' argument prevents an error if the directory already exists.
    (parent_path / folder_name).mkdir(parents=True, exist_ok=True)
    return parent_path / folder_name

def scatter_list(
        list_to_scatter: list[T],
        indices: list[int]
) -> list[T]:
    """Reorders a list by selecting elements at specified indices.

    Args:
        list_to_scatter: The source list to scatter/reorder.
        indices: A list of indices specifying which elements to select and their order.

    Returns:
        A new list containing elements from `list_to_scatter` at the specified indices.

    Example:
        >>> items = ['a', 'b', 'c', 'd']
        >>> scatter_list(items, [3, 1, 0])
        ['d', 'b', 'a']
    """
    return [list_to_scatter[b_i] for b_i in indices]

def flatten_list(
        nested_list: list[T]
) -> list[T]:
    """Flattens a nested list or tuple into a single list recursively.

    Completely flattens any level of nesting. Lists and tuples are recursively
    flattened, while other types are kept as-is.

    Args:
        nested_list: The nested list or tuple to flatten.

    Returns:
        A single flattened list containing all non-list/tuple elements.

    Example:
        >>> flatten_list([1, [2, 3], [[4], 5]])
        [1, 2, 3, 4, 5]
        >>> flatten_list([1, (2, [3, (4, 5)])])
        [1, 2, 3, 4, 5]
    """
    result = []
    for item in nested_list:
        if isinstance(item, list) or isinstance(item, tuple): result.extend(flatten_list(item))
        else: result.append(item)
    return result

def flatten_list_to_depth(
    nested_list: list[T],
    depth: int | float = float('inf')
) -> list[T]:
    """
    Flattens a nested list to a specified depth.

    By default, it flattens the list completely.

    Args:
        nested_list (list[T]): The nested list to flatten.
        depth (float): The maximum number of levels to flatten.
                       - `float('inf')` (default): Flattens completely.
                       - ....
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
        if isinstance(item, (list)) and depth > 0:
            # Recurse, but with one less level of depth allowed
            result.extend(flatten_list_to_depth(item, depth - 1))
        else:
            # Append the item as is, either because it's not a list/tuple
            # or because we've reached our desired depth
            result.append(item)
    return result

def flatten_cs_dict(
        cs_dict: ClassSplitted[T],
) -> tuple[list[T], list[int]]:
    """Flattens a class-splitted dictionary into separate lists of values and keys.

    Args:
        cs_dict: A dictionary mapping class indices to values.

    Returns:
        A tuple containing:
        - A list of all values from the dictionary.
        - A list of all keys (class indices) from the dictionary.

    Example:
        >>> cs_dict = {0: 'cat', 1: 'dog', 5: 'bird'}
        >>> flatten_cs_dict(cs_dict)
        (['cat', 'dog', 'bird'], [0, 1, 5])
    """
    pos_classes = list(cs_dict.keys())
    values = list(cs_dict.values())
    return values, pos_classes

def flatten_cs_dicts(
        cs_dicts: list[ClassSplitted[T]],
) -> tuple[list[T], list[int], list[int]]:
    """
    Processes a batch of dictionaries into three flat lists:
    1. All values from all dictionaries.
    2. The corresponding classes (keys).
    3. An index mapping each element back to its original dict in the batch.
    
    This format is ideal for vectorized operations (e.g., with NumPy).
    """
    flat_values = []
    flat_pos_classes = []
    batch_indices = []

    for i, cs_dict in enumerate(cs_dicts):
        if not cs_dict:
            continue
        # In Python 3.7+, dicts are ordered, so .keys() and .values()
        # will correspond. It's safe and efficient.
        keys = list(cs_dict.keys())
        values = list(cs_dict.values())
        
        flat_pos_classes.extend(keys)
        flat_values.extend(values)
        # Create the corresponding indices for this dictionary
        batch_indices.extend([i] * len(keys))
        
    return flat_values, flat_pos_classes, batch_indices

def unflatten_cs_dicts(
    flat_values: list[T],
    flat_pos_classes: list[int],
    batch_indices: list[int],
    original_batch_size: int
) -> list[ClassSplitted[T]]:
    """
    Reverts the operation of flatten_cs_dicts.
    
    Reconstructs the original list of dictionaries from the three flat lists.

    Args:
        flat_values: A flat list of all values.
        flat_pos_classes: A flat list of the corresponding keys (classes).
        batch_indices: A flat list mapping each element to its original dict index.
        original_batch_size: The number of dictionaries in the original batch.
                             This is required to reconstruct empty dictionaries.

    Returns:
        The original list of class-splitted dictionaries.
    """
    # 1. Initialize the output list with empty dictionaries.
    #    This correctly handles any dictionaries that were empty in the original list.
    reconstructed_dicts: list[ClassSplitted[T]] = [{} for _ in range(original_batch_size)]

    # 2. Iterate through the flat lists in parallel.
    #    zip is perfect for this, combining corresponding elements.
    for value, key, batch_idx in zip(flat_values, flat_pos_classes, batch_indices):
        # 3. Use the batch_idx to find the correct dictionary in the list
        #    and insert the key-value pair.
        reconstructed_dicts[batch_idx][key] = value
        
    return reconstructed_dicts

def batch_list(
        list_to_batch: list[T],
        batch_size: int
) -> list[list[T]]:
    """Partitions a list into sub-lists of maximum given size.

    Converts the input to a list if it isn't already. The last batch may contain
    fewer elements than `batch_size` if the list length is not evenly divisible.

    Args:
        list_to_batch: The list (or iterable) to partition into batches.
        batch_size: The maximum size of each batch.

    Returns:
        A list of batches, where each batch is a list of at most `batch_size` elements.

    Example:
        >>> batch_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
        >>> batch_list(range(7), 3)
        [[0, 1, 2], [3, 4, 5], [6]]
    """
    if not isinstance(list_to_batch, list):
        list_to_batch = list(list_to_batch)
    return [list_to_batch[i:i + batch_size] for i in range(0, len(list_to_batch), batch_size)]

def batch_dict(
    dict_to_batch: dict[K, V],
    batch_size: int
) -> list[dict[K, V]]:
    """Partitions a dictionary into a list of smaller dictionaries.

    Each dictionary in the output list will have at most `batch_size` items.

    Args:
        dict_to_batch (Dict[K, V]): The dictionary to partition.
        batch_size (int): The maximum number of items in each batch dictionary.

    Returns:
        List[Dict[K, V]]: The list of batched dictionaries.
        
    Raises:
        ValueError: If batch_size is not a positive integer.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    # Convert the dictionary's items to a list to allow for slicing
    items = list(dict_to_batch.items())
    
    # Use a list comprehension similar to the list batching function.
    # For each slice of items, convert it back into a dictionary.
    return [dict(items[i:i + batch_size]) for i in range(0, len(items), batch_size)]

def get_batch_keys_amount(
        list_of_dicts: list[dict]
) -> int:
    """Returns the total number of keys across all dictionaries in a list.

    Sums the number of keys from each dictionary in the provided list.

    Args:
        list_of_dicts: A list of dictionaries.

    Returns:
        The total count of keys across all dictionaries.

    Example:
        >>> dicts = [{0: 'a', 1: 'b'}, {2: 'c'}, {}]
        >>> get_batch_keys_amount(dicts)
        3
    """
    return sum([len(d.keys()) for d in list_of_dicts])

def batch_cs_list(
        list_of_dicts: list[dict],
        max_keys_per_batch: int
) -> list[list[dict]]:
    """Batches a list of dictionaries by limiting the total keys per batch.

    Groups dictionaries into batches such that each batch contains at most
    `max_keys_per_batch` keys in total. Dictionaries are kept intact and not split.
    If adding a dictionary would exceed the limit, a new batch is started.

    Note:
        If a single dictionary contains more keys than `max_keys_per_batch`,
        it will be placed in its own batch, potentially exceeding the limit.

    Args:
        list_of_dicts: The list of dictionaries to batch.
        max_keys_per_batch: The maximum total number of keys allowed per batch.

    Returns:
        A list of batches, where each batch is a list of dictionaries.

    Example:
        >>> dicts = [{'a': 1}, {'b': 2, 'c': 3}, {'d': 4}, {'e': 5, 'f': 6}]
        >>> batch_cs_list(dicts, max_keys_per_batch=3)
        [[{'a': 1}, {'b': 2, 'c': 3}], [{'d': 4}, {'e': 5, 'f': 6}]]
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

def flatten_list_of_lists(
        nested_list: list[list[T]]
) -> tuple[list[T], list[int]]:
    """
    Flattens a list of lists into a single list and a list of lengths.

    The list of lengths is the "metadata" required to unflatten the list
    back to its original structure.

    Args:
        nested_list: A list containing other lists (one level of nesting).
                     Example: [[1, 2], ["a", "b", "c"], [True]]

    Returns:
        A tuple containing:
        - A single list with all the elements from the sub-lists.
          Example: [1, 2, "a", "b", "c", True]
        - A list of integers representing the length of each original sub-list.
          Example: [2, 3, 1]
    """
    # Use a list comprehension to get the length of each sub-list
    lengths = [len(sublist) for sublist in nested_list]
    
    # Use a nested list comprehension to create the single flat list
    flat_list = [item for sublist in nested_list for item in sublist]
    
    return flat_list, lengths

def unflatten_list_of_lists(
        flat_list: list[Any],
        lengths: list[int]
) -> list[list[Any]]:
    """
    Reconstructs a nested list from a flat list and a list of lengths.

    This is the inverse operation of the `flatten` function.

    Args:
        flat_list: A single, flat list of items.
                   Example: [1, 2, "a", "b", "c", True]
        lengths: A list of integers for partitioning the flat_list.
                 Example: [2, 3, 1]

    Returns:
        The reconstructed list of lists.
        Example: [[1, 2], ["a", "b", "c"], [True]]
    """
    if sum(lengths) != len(flat_list):
        raise ValueError("The sum of lengths must be equal to the length of the flat list.")

    nested_list = []
    current_position = 0
    for length in lengths:
        # Slice the flat_list from the current position to the end of the chunk
        chunk = flat_list[current_position : current_position + length]
        nested_list.append(chunk)
        
        # Move the position marker for the next iteration
        current_position += length
        
    return nested_list

def is_state(
            obj: dict
) -> bool:
        """Checks if a dictionary represents a state object.

        A state object is defined as a dictionary containing a "state" key.

        Args:
            obj: The dictionary to check.

        Returns:
            True if the dictionary contains a "state" key, False otherwise.

        Example:
            >>> is_state({'state': 'active', 'id': 1})
            True
            >>> is_state({'id': 1, 'name': 'test'})
            False
        """
        return "state" in obj.keys()

def dispatch_and_process_in_batches(
    items_list: list[T],
    dispatch_indices: list[int],
    functions: list[Callable[[list[T]], list[R]]]
) -> list[R]:
    """
    Splits a list of items based on a dispatch index list, processes them in
    separate batches using corresponding functions, and reconstructs the
    results in the original order.

    This is useful for scenarios where different types of items in a list
    require processing by different models or APIs, and batching is desired
    for efficiency.

    Type Parameters:
        T: The type of items in the input list.
        P: The type of items in the output list.

    Args:
        items_list: The list of items to process.
        dispatch_indices: A list of integers of the same length as `items_list`.
                          Each integer `i` at a given position specifies that
                          the item at that same position in `items_list` should
                          be processed by `functions[i]`.
        functions: A list of functions to apply to the batches. Each function
                   must accept a list of type `T` and return a list of a
                   processed type `P`.

    Returns:
        A list of processed items, in the same order as the input `items_list`.

    Raises:
        ValueError: If the length of `items_list` and `dispatch_indices` do
                    not match, or if an index in `dispatch_indices` is out of
                    bounds for the `functions` list.

    Example:
        >>> # Function to process numbers by doubling them
        >>> def double_numbers(nums: list[int]) -> list[int]:
        ...     print(f"Doubling batch: {nums}")
        ...     return [n * 2 for n in nums]
        ...
        >>> # Function to process numbers by turning them into strings
        >>> def stringify_numbers(nums: list[int]) -> list[str]:
        ...     print(f"Stringifying batch: {nums}")
        ...     return [f"Num_{n}" for n in nums]
        ...
        >>> data = [10, 20, 30, 40, 50]
        >>> # Use function 1 for items at index 0, 3
        >>> # Use function 0 for items at index 1, 2, 4
        >>> indices = [1, 0, 0, 1, 0]
        >>> funcs = [double_numbers, stringify_numbers]
        >>> dispatch_and_process_in_batches(data, indices, funcs)
        Stringifying batch: [10, 40]
        Doubling batch: [20, 30, 50]
        ['Num_10', 40, 60, 'Num_40', 100]
    """
    if not items_list:
        return []

    if len(items_list) != len(dispatch_indices):
        raise ValueError(
            "The `items_list` and `dispatch_indices` must have the same length."
        )

    num_funcs = len(functions)
    
    # 1. SPLIT & INDEX: Create batches for each function.
    # `batches` will store the items, `original_indices` stores their original positions.
    batches = [[] for _ in range(num_funcs)]
    original_indices = [[] for _ in range(num_funcs)]

    for i, item in enumerate(items_list):
        func_index = dispatch_indices[i]
        if not 0 <= func_index < num_funcs:
            raise ValueError(
                f"Dispatch index {func_index} at position {i} is out of bounds "
                f"for the provided list of {num_funcs} functions."
            )
        batches[func_index].append(item)
        original_indices[func_index].append(i)

    # 2. APPLY: Process each non-empty batch with its corresponding function.
    all_results = [
        func(batch) if batch else []
        for func, batch in zip(functions, batches)
    ]

    # 3. COMBINE: Reconstruct the final list in the original order.
    # Create a placeholder list of the correct size.
    final_results: list[R] = [None] * len(items_list)
    print(final_results)

    # Iterate through each function's results and original indices to correctly
    # place the processed items back into the final list.
    for i in range(num_funcs):
        indices_for_batch = original_indices[i]
        results_for_batch = all_results[i]
        print("--------------------------------")
        print(len(results_for_batch))
        print(indices_for_batch)
        print(results_for_batch)
        for original_idx, result in zip(indices_for_batch, results_for_batch):
            final_results[original_idx] = result
            
    return final_results

def main() -> None:
    """Main function for testing and demonstrating utility functions.
    
    Currently tests the flatten and unflatten operations for lists of lists.
    """
    def try_flatten_unflatten_list_of_lists() -> None:
        # 1. Define the original data
        original_data = [
            [1, "apple", 3.14],
            [],
            [True, False],
            ["single_item"],
            [4, 5, 6, 7, 8]
        ]

        print(f"Original Data:\n{original_data}\n")

        # 2. Flatten the data
        flat_data, structure_info = flatten_list_of_lists(original_data)

        print(f"--- Flattening ---")
        print(f"Flattened List: {flat_data}")
        print(f"Structure Info (lengths): {structure_info}\n")

        # 3. Unflatten the data using the saved structure info
        reconstructed_data = unflatten_list_of_lists(flat_data, structure_info)

        print(f"--- Unflattening ---")
        print(f"Reconstructed Data:\n{reconstructed_data}\n")

        # 4. Verify that the reconstructed data is identical to the original
        print("Verification:")
        assert original_data == reconstructed_data
        print("Success! The reconstructed data matches the original data.")

if __name__ == '__main__':
    main()
