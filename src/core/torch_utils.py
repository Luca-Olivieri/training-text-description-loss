"""
PyTorch utility functions for tensor operations, memory management, and model optimization.

This module provides a collection of utility functions for working with PyTorch tensors,
including tensor mapping and blending, memory management, activation hooks, statistical
operations, tensor list flattening/unflattening, and model compilation utilities.
"""

from core.config import *

import gc
from collections import OrderedDict

import torch
from torch import nn

from core._types import Callable, TensorStructureInfo, ListStructureInfo

def get_compute_capability() -> float:
    """
    Get the CUDA compute capability of the current GPU device.
    
    Returns:
        float: The compute capability as a decimal number where the major version
               is the integer part and the minor version is the decimal part 
               (e.g., 7.5 for compute capability 7.5).
    """
    compute_capability = torch.cuda.get_device_capability()
    compute_capability = compute_capability[0] + 0.1*compute_capability[1]
    return compute_capability

def compile_torch_model(model: torch.nn.Module):
    """
    Compile a PyTorch model using torch.compile if the GPU compute capability is sufficient.
    
    This function only compiles the model if the CUDA compute capability is 7.0 or higher,
    as torch.compile requires newer GPU architectures for optimal performance.
    
    Args:
        model (torch.nn.Module): The PyTorch model to compile.
    
    Returns:
        torch.nn.Module: The compiled model if compute capability >= 7.0, otherwise 
                        the original model unchanged.
    """
    if get_compute_capability() >= 7.0:
        model = torch.compile(model)
    return model

def map_tensors(
        input_tensor: torch.Tensor,
        mapping_dict: dict[int, int]
) -> torch.Tensor:
    """
    Map tensor values according to a dictionary using a lookup tensor for efficient remapping.
    
    This function creates a lookup tensor from the mapping dictionary and uses it to remap
    all values in the input tensor in a vectorized manner. Values not present in the mapping
    dictionary are mapped to 0 by default.
    
    Args:
        input_tensor (torch.Tensor): The input tensor whose values need to be remapped.
        mapping_dict (dict[int, int]): A dictionary mapping original values to new values.
    
    Returns:
        torch.Tensor: A tensor with the same shape as input_tensor but with values 
                     remapped according to mapping_dict. Unmapped values are set to 0.
    
    Example:
        >>> tensor = torch.tensor([1, 2, 3, 1, 2])
        >>> mapping = {1: 10, 2: 20, 3: 30}
        >>> result = map_tensors(tensor, mapping)
        >>> # result: tensor([10, 20, 30, 10, 20])
    """
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

def is_list_of_tensors(
        item_to_check: list
) -> bool:
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

def blend_tensors(
        tensor1: torch.Tensor, 
        tensor2: torch.Tensor,
        alpha: float
) -> torch.Tensor:
    """
    Blend two tensors using linear interpolation (lerp) and clamp the result to uint8 range.
    
    This function performs linear interpolation between two tensors using torch.lerp,
    which computes: tensor1 + alpha * (tensor2 - tensor1), equivalent to 
    tensor1 * (1 - alpha) + tensor2 * alpha. The result is clamped to [0, 255] 
    and converted to uint8, making it suitable for image blending operations.
    
    Args:
        tensor1 (torch.Tensor): The first tensor (start point of interpolation).
        tensor2 (torch.Tensor): The second tensor (end point of interpolation).
        alpha (float): The interpolation weight. When alpha=0, returns tensor1;
                      when alpha=1, returns tensor2; values between 0 and 1
                      produce a blend.
    
    Returns:
        torch.Tensor: The blended tensor as uint8, clamped to [0, 255].
    
    Note:
        torch.lerp(start, end, weight) computes: start + weight * (end - start),
        which is algebraically equivalent to: start * (1 - weight) + end * weight
    """
    blended_tensor = torch.lerp(tensor1.float(), tensor2.float(), alpha)
    out_tensor = torch.clamp(blended_tensor, 0, 255).to(torch.uint8)
    return out_tensor

def clear_memory(
        ram: bool = True,
        gpu: bool = True,
) -> None:
    """
    Clear memory by running garbage collection and/or emptying the CUDA cache.
    
    This function helps free up memory by triggering Python's garbage collector
    and clearing PyTorch's CUDA memory cache. It can be useful when dealing with
    memory-intensive operations or when switching between different models.
    
    Args:
        ram (bool, optional): If True, run Python's garbage collector to free 
                             CPU memory. Defaults to True.
        gpu (bool, optional): If True, empty PyTorch's CUDA cache to free 
                             GPU memory. Defaults to True.
    
    Returns:
        None
    
    Note:
        MemoryError exceptions during cache clearing are silently ignored.
    """
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
    Create a forward hook function to capture layer activations during forward pass.
    
    This function returns a hook that can be registered to a PyTorch module using
    `module.register_forward_hook()`. When the module's forward pass is executed,
    the hook captures the output activation and stores it in the provided dictionary
    with the specified name as the key.
    
    Args:
        name (str): The identifier/key under which to store the activation in the
                   activations dictionary.
        activations (dict[str, torch.Tensor]): A dictionary to store the captured
                                               activations. This dict is modified in-place.
    
    Returns:
        Callable: A hook function that can be registered to a PyTorch module to
                 capture its output activations.
    
    Example:
        >>> activations = {}
        >>> hook = get_activation('layer1', activations)
        >>> model.layer1.register_forward_hook(hook)
        >>> output = model(input)
        >>> layer1_activation = activations['layer1']
    """
    def hook(
            model: nn.Module,
            input: torch.Tensor,
            output: torch.Tensor,
    ) -> None:
        activations[name] = output
    return hook

def nanstd(
        data: torch.Tensor,
        dim: list[int] | int,
        keepdim: bool = False
) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor along specified dimensions, ignoring NaN values.
    
    This function calculates the standard deviation while treating NaN values as if they
    don't exist in the computation. It uses the formula: sqrt(mean((x - mean(x))^2)),
    where mean operations ignore NaN values.
    
    Args:
        data (torch.Tensor): The input tensor containing the data.
        dim (list[int] | int): The dimension or dimensions along which to compute
                              the standard deviation.
        keepdim (bool, optional): If True, the output tensor retains the reduced
                                 dimension(s) with size 1. Defaults to False.
    
    Returns:
        torch.Tensor: The standard deviation computed along the specified dimension(s),
                     with NaN values ignored in the calculation.
    
    Note:
        This function uses torch.nanmean for all mean calculations, ensuring NaN
        values are properly excluded from both the mean and variance computations.
    """
    result = torch.sqrt(
        torch.nanmean(
            torch.pow(torch.abs(data-torch.nanmean(data, dim=dim).unsqueeze(dim)), 2),
            dim=dim
        )
    )
    if keepdim:
        result = result.unsqueeze(dim)
    return result

def flatten_tensor_list(
        tensor_list: TensorStructureInfo,
) -> tuple[torch.Tensor, ListStructureInfo]:
    """
    Flattens a list (potentially nested) of tensors into a single tensor
    and returns the information needed to unflatten it.

    Args:
        tensor_list: A list, potentially nested, containing PyTorch tensors.
                     All tensors must have the same number of dimensions and
                     the same shape except for the first dimension.

    Returns:
        A tuple containing:
        - flat_tensor (torch.Tensor): A single tensor containing all the data from
          the input tensors, concatenated along dimension 0.
        - structure_info (list): A nested list that mirrors the original structure,
          but with tensors replaced by the size of their first dimension.
    """
    if not isinstance(tensor_list, list):
        raise TypeError("Input must be a list of tensors.")

    flat_tensors = []
    structure_info = []

    # Helper function to recursively traverse the list
    def _traverse(
            sub_list: TensorStructureInfo,
            current_structure: ListStructureInfo
    ) -> None:
        for item in sub_list:
            if isinstance(item, torch.Tensor):
                # Store the tensor for concatenation
                flat_tensors.append(item)
                # Store the size of its first dimension in the structure info
                current_structure.append(item.shape[0])
            elif isinstance(item, list):
                # Recurse into the nested list
                new_sub_structure = []
                current_structure.append(new_sub_structure)
                _traverse(item, new_sub_structure)
            else:
                raise TypeError(f"Unsupported type in list: {type(item)}")

    _traverse(tensor_list, structure_info)
    
    if not flat_tensors:
        # Handle empty list case
        return torch.tensor([]), []

    # Concatenate all found tensors along the first dimension
    flat_tensor = torch.cat(flat_tensors, dim=0)

    return flat_tensor, structure_info

def unflatten_tensor_list(
        flat_tensor: torch.Tensor,
        structure_info: ListStructureInfo
) -> TensorStructureInfo:
    """
    Reconstruct a nested list of tensors from a flattened tensor and structure information.
    
    This function reverses the operation performed by flatten_tensor_list, reconstructing
    the original nested list structure from a concatenated tensor and its metadata.
    The structure_info parameter contains the sizes and nesting information needed to
    split and reshape the flat tensor back into its original form.

    Args:
        flat_tensor (torch.Tensor): The flattened tensor containing all concatenated data.
        structure_info (ListStructureInfo): Metadata describing the original nested
                                           structure and tensor sizes, as returned by
                                           flatten_tensor_list.

    Returns:
        TensorStructureInfo: A nested list of tensors reconstructed in the same format
                            as the original input to flatten_tensor_list.
    
    Example:
        >>> flat, info = flatten_tensor_list([[tensor1, tensor2], [tensor3]])
        >>> reconstructed = unflatten_tensor_list(flat, info)
        >>> # reconstructed == [[tensor1, tensor2], [tensor3]]
    """
    # First, get a flat list of all tensor sizes from the structure info
    sizes = []
    def _get_sizes(
            sub_structure: TensorStructureInfo
    ) -> list:
        for item in sub_structure:
            if isinstance(item, int):
                sizes.append(item)
            elif isinstance(item, list):
                _get_sizes(item)
    
    _get_sizes(structure_info)

    if not sizes:
        return []

    # Split the flat tensor back into a list of original tensors
    # torch.split returns a tuple, so we make it an iterator
    split_tensors = iter(torch.split(flat_tensor, sizes, dim=0))

    # Helper function to recursively rebuild the nested list structure
    def _rebuild(
            sub_structure: TensorStructureInfo
    ) -> list:
        rebuilt_list = []
        for item in sub_structure:
            if isinstance(item, int):
                # This was a tensor's location, so pull the next one from the iterator
                rebuilt_list.append(next(split_tensors))
            elif isinstance(item, list):
                # This was a nested list, so recurse
                rebuilt_list.append(_rebuild(item))
        return rebuilt_list

    return _rebuild(structure_info)

def unprefix_state_dict(
        prefixed_state_dict: OrderedDict,
        prefix: str
) -> OrderedDict:
    """
    Remove a prefix from all keys in a PyTorch model state dictionary.
    
    This function is particularly useful when working with compiled models or models
    wrapped in certain PyTorch containers that add prefixes to parameter names.
    E.g., the function can remove the '_orig_mod.' prefix from all keys in the state dict,
    which is commonly added by torch.compile().
    
    Args:
        prefixed_state_dict (OrderedDict): The state dictionary with prefixed keys.
        prefix (str): The prefix to remove from keys.
    
    Returns:
        OrderedDict: A new state dictionary with the prefix removed from all keys.
    
    Example:
        >>> state_dict = {'_orig_mod.layer1.weight': tensor1, '_orig_mod.layer1.bias': tensor2}
        >>> clean_dict = unprefix_state_dict(state_dict, '_orig_mod.')
        >>> # clean_dict: {'layer1.weight': tensor1, 'layer1.bias': tensor2}
    """
    unprefixed_state_dict = {key.replace(prefix, ''): value for key, value in prefixed_state_dict.items()}
    return unprefixed_state_dict

def group_by_tensor(
        keys: torch.Tensor,
        values: torch.Tensor
) -> dict[int, list[int]]:
    """
    Groups elements of the 'values' tensor based on the corresponding elements 
    in the 'keys' tensor.
    
    Args:
        keys: Tensor containing the grouping keys.
        values: Tensor containing the values to be grouped.
        
    Returns:
        A dictionary where keys come from 'keys' and values are lists of 
        corresponding elements from 'values'.
    """
    # 1. Flatten inputs to ensure we are working with 1D arrays
    flat_keys = keys.flatten()
    flat_values = values.flatten()

    if flat_keys.numel() != flat_values.numel():
        raise ValueError(f"Tensors must have the same number of elements. "
                         f"Got keys={flat_keys.numel()}, values={flat_values.numel()}")

    # 2. Sort keys to ensure identical keys are contiguous
    # We use argsort to apply the exact same permutation to the values
    sort_indices = torch.argsort(flat_keys)
    sorted_keys = flat_keys[sort_indices]
    sorted_values = flat_values[sort_indices]

    # 3. Find unique keys and the count of each key (requires sorted input)
    unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)

    # 4. Split the sorted values based on the counts of the keys
    # We convert counts to a list because torch.split expects a list of section sizes
    grouped_values = torch.split(sorted_values, counts.tolist())

    # 5. Construct the final dictionary
    # .item() converts 0-d tensor key to int
    # .tolist() converts the value tensor chunk to a list of ints
    return {k.item(): v.tolist() for k, v in zip(unique_keys, grouped_values)}

def main() -> None:
    """
    Main function for testing and demonstrating module functionality.
    
    This function serves as an entry point for running tests and examples
    of the utility functions defined in this module.
    """
    
    def try_flatten_unflatten_tensor_list() -> None:
        C, H = 3, 4
        t1 = torch.randn(2, C, H)  # N=2
        t2 = torch.randn(5, C, H)  # N=5
        t3 = torch.randn(1, C, H)  # N=1
        t4 = torch.randn(3, C, H)  # N=3
        l = [t1, t2, t3, t4]
        print(l)
        flat_l, struct = flatten_tensor_list(l)
        print(flat_l.shape)
        new_l = unflatten_tensor_list(flat_l, struct)
        print(new_l)
        print(all([(new_t == t).all() for new_t, t in zip(new_l, l)]))

    def try_group_by_tensor() -> None:
        A = torch.tensor([1, 2, 1, 3, 2, 1])
        B = torch.tensor([10, 20, 30, 40, 50, 60])

        result = group_by_tensor(A, B)
        
        print(result)
        # Output: {1: [10, 30, 60], 2: [20, 50], 3: [40]}

    try_group_by_tensor()

if __name__ == '__main__':
    main()

