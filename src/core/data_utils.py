from core.config import *
from core._types import T, Any
from pathlib import Path

def create_directory(
        parent_path: Path,
        folder_name: str
) -> Path:
    # Create the directory, including any necessary parents.
    # The 'exist_ok=True' argument prevents an error if the directory already exists.
    (parent_path / folder_name).mkdir(parents=True, exist_ok=True)
    return parent_path / folder_name

def flatten_list(
        nested_list: list[T]
) -> list[list[T]]:
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

def batch_list(
        list_to_batch: list[T],
        batch_size: int
) -> list[list[T]]:
    """Partitions a list into sub-lists of maximum given size.

    Args:
        list_ (list[Any]): The list to partition.
        batch_size (int): The maximum size of each batch.

    Returns:
        list[list[Any]]: The list of batches.
    """
    if not isinstance(list_to_batch, list):
        list_to_batch = list(list_to_batch)
    return [list_to_batch[i:i + batch_size] for i in range(0, len(list_to_batch), batch_size)]

def get_batch_keys_amount(
        list_of_dicts: list[dict]
) -> int:
    """Returns the total number of keys in a list of dictionaries.

    Args:
        list_of_dicts (list[dict]): The list of dictionaries.

    Returns:
        int: The total number of keys.
    """
    return sum([len(d.keys()) for d in list_of_dicts])

def batch_cs_list(
        list_of_dicts: list[dict],
        max_keys_per_batch: int
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

def flatten_list_of_lists(
        nested_list: list[list[Any]]
) -> tuple[list[Any], list[int]]:
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
        """
        Checks if a dictionary represents a state object.

        Args:
            obj: Dictionary to check.

        Returns:
            True if the object is a state, False otherwise.
        """
        return "state" in obj.keys()

def main() -> None:
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
