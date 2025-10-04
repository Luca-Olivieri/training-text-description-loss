from core.config import *

import gc

import torch
from torch import nn

from core._types import Callable, TensorStructure, StructureInfo

def get_compute_capability() -> float:
    compute_capability = torch.cuda.get_device_capability()
    compute_capability = compute_capability[0] + 0.1*compute_capability[1]
    return compute_capability

def compile_torch_model(model: torch.nn.Module):
    if get_compute_capability() >= 7.0:
        model = torch.compile(model)
    return model

def map_tensors(
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
    Blends two tensors using torch.lerp for a potentially more optimized approach.
    
    torch.lerp(start, end, weight) is equivalent to: start + weight * (end - start)
    which is algebraically the same as: start * (1 - weight) + end * weight
    """
    blended_tensor = torch.lerp(tensor1.float(), tensor2.float(), alpha)
    out_tensor = torch.clamp(blended_tensor, 0, 255).to(torch.uint8)
    return out_tensor

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

def flatten_tensor_list(
        tensor_list: TensorStructure,
) -> tuple[torch.Tensor, StructureInfo]:
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
            sub_list: TensorStructure,
            current_structure: StructureInfo
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
        structure_info: StructureInfo
) -> TensorStructure:
    """
    Reconstructs an original nested list of tensors from a flattened tensor
    and its corresponding structure information.

    Args:
        flat_tensor (torch.Tensor): The flattened tensor.
        structure_info (list): The metadata describing the original nested
                               structure and tensor sizes.

    Returns:
        A nested list of tensors in the same format as the original.
    """
    # First, get a flat list of all tensor sizes from the structure info
    sizes = []
    def _get_sizes(
            sub_structure: TensorStructure
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
            sub_structure: TensorStructure
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

def main() -> None:
    
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

if __name__ == '__main__':
    main()

