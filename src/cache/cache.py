from __future__ import annotations

from core.config import *

import os
import shutil
import json
from pathlib import Path
import sys
from collections import defaultdict

import torch
import torchmetrics as tm

from core._types import Optional, CacheItem, CacheKey, Any, Iterator, Union, ABC, abstractmethod, override

class Cache:
    """
    A dictionary-like cache for storing and retrieving image tensors and associated text.

    The cache supports different storage backends and memory loading targets,
    making it flexible for various hardware configurations and workflows.

    It indexes items by a string ID: ID -> (image_tensor, text_string).

    Args:
        storage_device (str): Where to primarily store the data.
            - "cpu": Store all data in system RAM. Fastest for small datasets.
            - "cuda": Store all tensors directly on the GPU. Requires GPU memory.
            - "disk": Store all data as individual files on disk. Best for large
                      datasets that don't fit in RAM/VRAM.
        memory_device (str): The device where tensors should be loaded upon retrieval.
            - "cpu": Tensors are returned on the CPU.
            - "cuda": Tensors are returned on the GPU.
        cache_dir (str, optional): The directory to use when storage_device is "disk".
            Required if storage_device is "disk".
    """

    def __init__(
            self,
            storage_device: str = "cpu",
            memory_device: str = "cpu",
            cache_dir: Optional[str | Path ] = None
    ) -> None:

        # Validate storage_device
        self.storage_device = storage_device.lower()
        if self.storage_device not in ["cpu", "cuda", "disk"]:
            raise ValueError("storage_device must be one of 'cpu', 'cuda', or 'disk'.")

        # Validate memory_device
        self.memory_device = memory_device.lower()
        if self.memory_device not in ["cpu", "cuda"]:
            raise ValueError("memory_device must be one of 'cpu' or 'cuda'.")

        # Handle CUDA availability checks
        if (self.storage_device == "cuda" or self.memory_device == "cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but 'cuda' was specified as a device.")

        # Handle disk storage setup
        self.cache_dir = None
        if self.storage_device == "disk":
            if cache_dir is None:
                raise ValueError("cache_dir must be provided when storage_device is 'disk'.")
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # For disk storage, we use an in-memory set to track keys for speed.
            self._index = {p.stem for p in self.cache_dir.glob("*.pt")}
        else:
            # For cpu and cuda, the data is stored in a dictionary.
            self._data: dict[CacheKey, CacheItem] = {}

    def __setitem__(
            self,
            key: CacheKey,
            value: CacheItem
    ) -> None:
        """Adds or updates an item in the cache."""
        if not isinstance(key, str):
            raise TypeError("Cache key (ID) must be a string.")
        if not (isinstance(value, tuple) and len(value) == 2 and
                isinstance(value[0], torch.Tensor) and isinstance(value[1], str)):
            raise TypeError("Value must be a tuple of (torch.Tensor, str).")

        image, text = value

        if self.storage_device == "cpu":
            # Store on CPU in RAM, will be moved to memory_device on retrieval
            self._data[key] = (image.cpu(), text)
        elif self.storage_device == "cuda":
            # Move tensor to GPU for storage
            self._data[key] = (image.to("cuda"), text)
        elif self.storage_device == "disk":
            # Save the item to a file on disk
            filepath = self.cache_dir / f"{key}.pt"
            # Always save tensors from CPU to avoid GPU-specific metadata issues
            torch.save({'image': image.cpu(), 'text': text}, filepath)
            self._index.add(key)

    def update(
            self,
            items: dict[CacheKey, CacheItem]
    ) -> None:
        """
        Adds or updates multiple items in the cache in a batch.

        Args:
            items (Dict[str, Tuple[torch.Tensor, str]]): A dictionary of
                {id: (image_tensor, text)} to add to the cache.
        """
        if self.storage_device == "cpu":
            processed_items = {k: (v[0].cpu(), v[1]) for k, v in items.items()}
            self._data.update(processed_items)
        elif self.storage_device == "cuda":
            processed_items = {k: (v[0].to("cuda"), v[1]) for k, v in items.items()}
            self._data.update(processed_items)
        elif self.storage_device == "disk":
            for key, (image, text) in items.items():
                filepath = self.cache_dir / f"{key}.pt"
                # Saving in a loop is necessary for file-based storage,
                # but we can update the index in one go.
                torch.save({'image': image.cpu(), 'text': text}, filepath)
            self._index.update(items.keys())

    def __getitem__(
            self,
            key: CacheKey
    ) -> CacheItem:
        """Retrieves an item from the cache and places the tensor on memory_device."""
        if key not in self:
            raise KeyError(f"Key '{key}' not found in the cache.")

        if self.storage_device in ["cpu", "cuda"]:
            image, text = self._data[key]
        elif self.storage_device == "disk":
            filepath = self.cache_dir / f"{key}.pt"
            data = torch.load(filepath, map_location='cpu') # Load to CPU first
            image, text = data['image'], data['text']
        else:
            # This should be unreachable due to __init__ checks
            raise RuntimeError("Invalid storage device.")

        # Move tensor to the requested memory device before returning
        return image.to(self.memory_device), text

    def get_many(
            self,
            keys: list[CacheKey]
    ) -> dict[CacheKey, CacheItem]:
        """
        Retrieves multiple items from the cache in a batch.

        This can be more efficient than repeated single gets, especially when
        data needs to be moved to a GPU, as it batches the device transfer.

        Args:
            keys (List[str]): A list of IDs to retrieve.

        Returns:
            Dict[str, Tuple[torch.Tensor, str]]: A dictionary mapping the
                found keys to their (image, text) values.

        Raises:
            KeyError: If any of the requested keys are not found in the cache.
        """
        # First, check for missing keys to ensure atomic failure
        # missing_keys = set(keys) - self.keys()
        # if missing_keys:
        #    raise KeyError(f"Keys not found in cache: {', '.join(missing_keys)}")

        results = {}
        if self.storage_device in ["cpu", "cuda"]:
            # Retrieve all data first
            raw_items = {key: self._data.get(key, (torch.tensor(torch.nan), None)) for key in keys}
        elif self.storage_device == "disk":
            raw_items = {}
            for key in keys:
                filepath = self.cache_dir / f"{key}.pt"
                data = torch.load(filepath, map_location='cpu')
                raw_items[key] = (data['image'], data['text'])

        # Now, batch the device transfer
        for key, (image, text) in raw_items.items():
            results[key] = (image.to(self.memory_device), text)
        
        return results

    def __delitem__(
            self,
            key: CacheKey
    ) -> None:
        """Deletes an item from the cache."""
        if key not in self:
            raise KeyError(f"Key '{key}' not found in the cache.")

        if self.storage_device in ["cpu", "cuda"]:
            del self._data[key]
        elif self.storage_device == "disk":
            filepath = self.cache_dir / f"{key}.pt"
            if filepath.exists():
                os.remove(filepath)
            self._index.remove(key)

    def delete_many(
            self,
            keys: list[CacheKey]
    ) -> None:
        """
        Deletes multiple items from the cache in a batch.

        Args:
            keys (List[str]): A list of IDs to delete.

        Raises:
            KeyError: If any of the keys to be deleted do not exist.
        """
        # Check for missing keys first for atomic failure
        missing_keys = set(keys) - self.keys()
        if missing_keys:
            raise KeyError(f"Cannot delete. Keys not found: {', '.join(missing_keys)}")

        if self.storage_device in ["cpu", "cuda"]:
            for key in keys:
                del self._data[key]
        elif self.storage_device == "disk":
            for key in keys:
                filepath = self.cache_dir / f"{key}.pt"
                # os.remove can fail, but we've already checked existence.
                # A try/except could make this more robust to race conditions.
                if filepath.exists():
                    os.remove(filepath)
            # Efficiently remove multiple items from the set index
            self._index.difference_update(set(keys))

    def __contains__(
            self,
            key: Any
    ) -> bool:
        """Checks if a key exists in the cache."""
        if self.storage_device == "disk":
            return key in self._index
        else:
            return key in self._data

    def contains_many(
            self,
            keys: list[CacheKey]
    ) -> dict[CacheKey, bool]:
        """
        Checks for the existence of multiple keys in a batch.

        Args:
            keys (List[str]): A list of IDs to check.

        Returns:
            Dict[str, bool]: A dictionary mapping each key to a boolean
                             indicating its presence in the cache.
        """
        if self.storage_device == "disk":
            existing_keys = self._index
        else:
            existing_keys = self._data.keys()

        # This is more efficient than a Python loop for large lists of keys
        return {key: key in existing_keys for key in keys}

    def __len__(self) -> int:
        """Returns the number of items in the cache."""
        if self.storage_device == "disk":
            return len(self._index)
        else:
            return len(self._data)

    def __iter__(self) -> Iterator[CacheKey]:
        """Returns an iterator over the keys in the cache."""
        if self.storage_device == "disk":
            return iter(self._index)
        else:
            return iter(self._data)

    def keys(self) -> Union[list[CacheKey], Iterator[CacheKey]]:
        """Returns a view of the cache's keys."""
        if self.storage_device == "disk":
            return list(self._index)
        else:
            return self._data.keys()

    def values(self) -> Iterator[CacheItem]:
        """Returns an iterator over the cache's values."""
        for key in self:
            yield self[key]

    def items(self) -> Iterator[tuple[CacheKey, CacheItem]]:
        """Returns an iterator over the cache's (key, value) pairs."""
        for key in self:
            yield key, self[key]

    def clear(self) -> None:
        """Removes all items from the cache."""
        if self.storage_device in ["cpu", "cuda"]:
            self._data.clear()
        elif self.storage_device == "disk":
            for key in list(self._index):
                self.__delitem__(key)
            self._index.clear()


    def save(
            self,
            directory_path: Union[str, Path]
    ) -> None:
        """
        Saves the entire cache to a specified directory.

        The saved format is independent of the current storage_device, ensuring
        it can be loaded into any cache configuration.

        Args:
            directory_path (str or Path): The directory where the cache will be saved. It will be created if it doesn't exist.
        """
        save_path = Path(directory_path)
        data_dir = save_path / "data"

        # Clean up old save if it exists
        if save_path.exists():
            shutil.rmtree(save_path)

        data_dir.mkdir(parents=True)

        # Save metadata
        metadata = {
            "original_storage_device": self.storage_device,
            "original_memory_device": self.memory_device,
            "item_count": len(self)
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Save each item to a file, ensuring consistency
        for key, (image, text) in self.items():
            # Note: self.items() already handles retrieving from any storage backend
            filepath = data_dir / f"{key}.pt"
            torch.save({'image': image.cpu(), 'text': text}, filepath)

        print(f"Cache saved successfully to {save_path}")

    @classmethod
    def load(
            cls,
            directory_path: Union[str, Path],
            storage_device: str = "cpu",
            memory_device: str = "cpu",
            cache_dir: Union[str, Path, None] = None
    ) -> "Cache":
        """
        Loads a cache from a directory into a new ImageTextCache instance.

        Args:
            directory_path (str or Path): The directory from which to load the cache.
            storage_device (str): The storage device for the *new* cache instance.
            memory_device (str): The memory device for the *new* cache instance.
            cache_dir (str, Path, optional): Directory for the new cache if its
                                             storage_device is 'disk'.

        Returns:
            ImageTextCache: A new, populated instance of the ImageTextCache.
        """
        load_path = Path(directory_path)
        if not load_path.exists() or not (load_path / "metadata.json").exists():
            raise FileNotFoundError(f"Cache directory not found or is invalid: {load_path}")

        data_dir = load_path / "data"

        # Create the new cache instance with the desired configuration
        new_cache = cls(storage_device, memory_device, cache_dir)

        # Load all items from the saved files into the new cache
        pt_files = list(data_dir.glob("*.pt"))
        print(f"Loading {len(pt_files)} items into a new cache with storage='{storage_device}'...")
        for filepath in pt_files:
            key = filepath.stem
            data = torch.load(filepath, map_location='cpu')
            new_cache[key] = (data['image'], data['text'])

        print("Cache loaded successfully.")
        return new_cache

    def get_usage(
            self,
            unit: str = "MB"
    ) -> dict[str, float]:
        """
        Calculates the approximate resource usage of the cache.

        This method provides an estimate of the space consumed by the cache's
        data on its primary storage device.

        Note: RAM/VRAM calculations are for the stored data and do not include
        all Python object overhead, which can be significant.

        Args:
            unit (str): The unit for the returned usage ('B', 'KB', 'MB', 'GB').
                        Defaults to 'MB'.

        Returns:
            dict[str, float]: A dictionary containing the usage for 'cpu', 'vram',
                              and 'disk' in the specified unit.
        """
        divisors = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
        if unit.upper() not in divisors:
            raise ValueError(f"Invalid unit '{unit}'. Choose from 'B', 'KB', 'MB', 'GB'.")
        divisor = divisors[unit.upper()]

        ram_bytes, vram_bytes, disk_bytes = 0, 0, 0

        if self.storage_device == "cpu":
            for image, text in self._data.values():
                # Tensors are stored on CPU in 'cpu' mode
                ram_bytes += image.numel() * image.element_size()
                ram_bytes += sys.getsizeof(text)

        elif self.storage_device == "cuda":
            for image, text in self._data.values():
                # Tensors are on GPU, text and other objects are in RAM
                vram_bytes += image.numel() * image.element_size()
                ram_bytes += sys.getsizeof(text)

        elif self.storage_device == "disk":
            if self._index: # Avoid unnecessary iteration if empty
                for key in self._index:
                    filepath = self.cache_dir / f"{key}.pt"
                    if filepath.exists():
                        disk_bytes += filepath.stat().st_size

        return {
            "cpu": ram_bytes / divisor,
            "vram": vram_bytes / divisor,
            "disk": disk_bytes / divisor
        }

    def __repr__(self) -> str:
        return f"ImageTextCache(storage='{self.storage_device}', memory='{self.memory_device}', "f"size={len(self)}), usage={self.get_usage()}"
    
class MaskTextCache:
    def __init__(
            self,
            cache: Cache,
            update_policy: CacheUpdatePolicy
    ) -> None:
        self.cache = cache
        self.update_policy = update_policy

    def get_many(
            self,
            img_uids: list[str]
    ) -> dict[str, tuple[torch.Tensor, str]]:
        return self.cache.get_many(keys=img_uids)
    
    def get_keys_to_update(
            self,
            keys: list[str],
            new_images: torch.Tensor,
    ) -> list[str]:
        images = [img for img, txt in self.cache.get_many(keys).values()]
        filtered_keys = self.update_policy.filter_keys(keys, images, new_images)
        return filtered_keys
    
    def get_cs_keys_to_update(
            self,
            new_cs_images: dict[str, dict[int, torch.Tensor]],
    ) -> dict[str, list[int]]:
        uids = list(new_cs_images.keys())
        flat_keys = [f'{uid}-{pos_c}' for uid in uids for pos_c in new_cs_images[uid].keys()]
        flat_new_images = torch.cat([torch.stack(list(new_imgs.values()), dim=0) for new_imgs in new_cs_images.values()])
        keys_to_update = self.get_keys_to_update(flat_keys, flat_new_images)
        cs_keys_to_update = group_uids(keys_to_update)
        return cs_keys_to_update
        
    def update(
            self,
            batch: dict[str, tuple[torch.Tensor, str]]
    ) -> None:
        self.cache.update(batch)

    def get_metric_diffs(
            self,
            batch: dict[str, tuple[torch.Tensor, str]]
    ) -> dict[str, float]:
        keys = list(batch.keys())
        cached_masks = [img for img, txt in self.cache.get_many(keys).values()]
        new_masks = [img for img, txt in batch.values()]
        return self.update_policy.get_metric_diffs(keys, cached_masks, new_masks)

    def get_cs_metric_diffs(
            self,
            batch: dict[str, dict[int, tuple[torch.Tensor, str]]]
    ) -> list[float]:
        uids = list(batch.keys())
        flat_keys = [f'{uid}-{pos_c}' for uid in uids for pos_c in batch[uid].keys()]
        cached_masks = self.get_many(img_uids=flat_keys)
        # new_masks = [img for img, txt in batch.values()]
        new_masks = [batch[uid][pos_c][0] for uid in uids for pos_c in batch[uid].keys()]
        return self.update_policy.get_metric_diffs(flat_keys, cached_masks, new_masks)
    
    def __repr__(self) -> str:
        return self.cache.__repr__()



class CacheUpdatePolicy(ABC):
    @abstractmethod
    def __init__(
            self
    ) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def filter_keys(
            self,
            keys: str,
            images_1: list[torch.Tensor],
            images_2: list[torch.Tensor],
    ) -> list[str]:
        raise NotImplementedError
    
    def get_metric_diffs(
            self,
            keys: str,
            images_1: list[torch.Tensor],
            images_2: list[torch.Tensor],
    ) -> dict[str, float]:
        nan_mask: list[bool] = [torch.any(torch.isnan(t)).item() for t in images_1]
        metrics = {key: self.metric(img_1, img_2) if not nan_mask[i] else torch.tensor(0.).to(img_1.device) for i, (key, img_1, img_2) in enumerate(zip(keys, images_1, images_2))}
        return metrics


class PercentilePolicy(CacheUpdatePolicy):
    def __init__(
            self,
            metric: tm.Metric,
            percentile: float,
            trend: Trend
    ) -> None:
        self.metric = metric
        self.percentile = percentile
        self.trend = trend
    
    def get_and_update_state(
            self,
            batch_quantile: float
    ) -> float:
        return self.trend.update_and_get_value(new_value=batch_quantile)

    @override
    def filter_keys(
            self,
            keys: str,
            images_1: list[torch.Tensor],
            images_2: list[torch.Tensor],
    ) -> list[str]:
        metrics_dict = self.get_metric_diffs(images_1=images_1, images_2=images_2, keys=[str(n) for n in range(len(images_1))])
        metrics = torch.stack(list(metrics_dict.values()))
        batch_quantile = torch.quantile(metrics, q=self.percentile)
        quantile = self.get_and_update_state(batch_quantile)
        keys_to_keep_mask: list[bool] = [bool(m <= quantile) for m in metrics]
        filtered_keys = [k for i, k in enumerate(keys) if keys_to_keep_mask[i] is True]
        return filtered_keys


class Trend(ABC):
    def __init__(self) -> None:
        raise NotImplementedError

    def get_current_value(self) -> float:
        if self.value is None:
            raise ValueError("The value has not been initialised.")
        return self.value
    
    def update_value(
            self,
            new_value: float
    ) -> None:
        raise NotImplementedError
    
    def update_and_get_value(
            self,
            new_value: float
    ) -> float:
        self.update_value(new_value)
        return self.get_current_value()


class Identity(Trend):

    @override
    def __init__(self) -> None:
        self.value: Optional[float] = None

    @override
    def update_value(
            self,
            new_value: float
    ) -> None:
        self.value = new_value


class SimpleExpSmoothing(Trend):

    @override
    def __init__(
            self,
            alpha: float,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = alpha
        self.value: Optional[float] = None

    @override
    def update_value(
            self,
            new_value: float
    ) -> None:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha*self.value + (1. - self.alpha)*new_value


def group_uids(
        data: list[str]
) -> dict[str, list[int]]:
    """
    Arranges a list of strings 'uid-pos_c' into a dictionary grouping pos_c by uid.

    Args:
        data: A list of strings, where each string is formatted as "uid-pos_c".
              Example: ["user1-10", "user2-5", "user1-15"]

    Returns:
        A dictionary where keys are the uids (str) and values are lists of
        the associated positions (int).
        Example: {"user1": [10, 15], "user2": [5]}
    """
    # Initialize a defaultdict with a list factory.
    # When a new key is accessed, it will automatically be created with an empty list.
    grouped_data = defaultdict(list)

    for item in data:
        # Split the string into two parts at the first hyphen.
        # Using maxsplit=1 is a good practice in case the uid itself contains a hyphen.
        # We use rsplit to be even safer, splitting from the right.
        try:
            uid, pos_c_str = item.rsplit('-', 1)
            pos_c = int(pos_c_str)
            grouped_data[uid].append(pos_c)
        except ValueError:
            # Handle cases where the string might not be in the correct format
            print(f"Skipping malformed item: {item}")
            
    return dict(grouped_data) # Convert back to a regular dict for the final output

def main() -> None:
    # Create dummy data
    B = 1500*10
    IMG_H, IMG_W = 224, 224
    image = torch.randn(IMG_H, IMG_W) > 0.5
    text = "Example text blah, blah, blah, blah, blah, blah, blah, blah, blah, blah."

    # 1. Using 'cpu' storage, loading to 'cpu'
    print("--- 1. CPU Storage / CPU Memory Demo ---")
    cpu_cache = Cache(storage_device="cpu", memory_device="cpu")
    for b in range(B):
        cpu_cache[f'{b}'] = (image.clone(), text[::-1][::-1])

    print(f"Cache state: {cpu_cache}")
    print(f"Number of items: {len(cpu_cache)}")
    print(f"'0' in cache: {'0' in cpu_cache}")

    # Retrieve an item
    retrieved_img, retrieved_text = cpu_cache["0"]
    print(f"Retrieved image for '0'. Shape: {retrieved_img.shape}, Device: {retrieved_img.device}")
    print("-" * 20)

    batch_data = {f'{b}': (image.clone(), text[::-1][::-1]) for b in range(B)}

    # 1. Using 'cpu' storage, loading to 'cpu'
    print("--- 1. CPU Storage / CPU Memory Demo ---")
    cpu_cache = Cache(storage_device="cpu", memory_device="cpu")

    cpu_cache.update(batch_data)

    print(f"Cache state: {cpu_cache}")
    print(f"Number of items: {len(cpu_cache)}")
    print(f"'0' in cache: {'0' in cpu_cache}")

    # Retrieve an item
    retrieved_img, retrieved_text = cpu_cache["0"]
    print(f"Retrieved image for '0'. Shape: {retrieved_img.shape}, Device: {retrieved_img.device}")
    print("-" * 20)

    # 2. Using 'disk' storage
    print("\n--- 2. Disk Storage Demo ---")
    DISK_CACHE_DIR = "./my_disk_cache"
    disk_cache = Cache(storage_device="disk", memory_device="cpu", cache_dir=DISK_CACHE_DIR)
    for b in range(B):
        disk_cache[f'{b}'] = (image, text)

    print(f"Cache state: {disk_cache}")
    retrieved_img_disk, _ = disk_cache["0"]
    print(f"Retrieved image for '0'. Device: {retrieved_img_disk.device}")

    # Check that files were created
    files_in_dir = os.listdir(DISK_CACHE_DIR)
    print(f"Files in {DISK_CACHE_DIR}: {files_in_dir}")

    # Delete an item
    del disk_cache["0"]
    print(f"After deletion: '0' in cache: {'0' in disk_cache}")
    files_in_dir_after_del = os.listdir(DISK_CACHE_DIR)
    print(f"Files in {DISK_CACHE_DIR} after deletion: {files_in_dir_after_del}")

    # Clean up disk cache directory for this demo
    shutil.rmtree(DISK_CACHE_DIR)
    print("-" * 20)

    # 3. Save and Load functionality
    print("\n--- 3. Save and Load Demo ---")
    SAVE_DIR = "./saved_cache_instance"

    # Use the cpu_cache from step 1
    print(f"Saving cache: {cpu_cache}")
    cpu_cache.save(SAVE_DIR)

    # Now, load it into a new cache with a different configuration
    # For example, load it into a 'disk' cache that returns tensors on CUDA (if available)
    load_memory_device = "cuda" if torch.cuda.is_available() else "cpu"
    LOADED_DISK_CACHE_DIR = "./loaded_disk_cache"

    loaded_cache = Cache.load(
        SAVE_DIR,
        storage_device="disk",
        memory_device=load_memory_device,
        cache_dir=LOADED_DISK_CACHE_DIR
    )

    print(f"Loaded cache state: {loaded_cache}")
    retrieved_img_loaded, _ = loaded_cache["0"]
    print(f"Retrieved '0' from loaded cache. Shape: {retrieved_img_loaded.shape}, Device: {retrieved_img_loaded.device}")

    # Verify iteration works
    print("Iterating through loaded cache items:")
    for key, (img, txt) in loaded_cache.items():
        print(f"  - Key: {key}, Text: '{txt}', Image Device: {img.device}")
        break

    # Clean up save/load directories
    shutil.rmtree(SAVE_DIR)
    shutil.rmtree(LOADED_DISK_CACHE_DIR)

    # 4. CUDA Storage (if available)
    if torch.cuda.is_available():
        print("\n--- 4. CUDA Storage Demo ---")
        cuda_cache = Cache(storage_device="cpu", memory_device="cuda")

        for b in range(B):
            cuda_cache[f'{b}'] = (image, text)


        print(cuda_cache)
        # The tensor is stored on the GPU
        # We can verify this non-public member for demonstration
        stored_tensor_device = cuda_cache._data["0"][0].device
        print(f"Internally, the tensor for '0' is on device: {stored_tensor_device}")

        # But when we retrieve it, it respects memory_device='cpu'
        retrieved_img_from_gpu, _ = cuda_cache["0"]
        print(f"When retrieved, the tensor is on device: {retrieved_img_from_gpu.device}")
    else:
        print("Test skipped because no CUDA available")

if __name__ == '__main__':
    main()
