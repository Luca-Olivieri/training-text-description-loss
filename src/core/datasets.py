"""
PyTorch Dataset implementations for semantic segmentation tasks.

This module provides dataset classes for working with various segmentation benchmarks
and custom data formats. It includes support for:

- **PASCAL VOC 2012**: Standard segmentation benchmark with 21 semantic classes
- **COCO 2017**: Large-scale dataset with 80+ object categories
- **JSONL datasets**: Efficient lazy-loading for large JSON Lines files
- **Image-caption pairs**: Combined datasets for vision-language tasks

Key Features:
    - Unified interface through abstract SegDataset base class
    - Flexible preprocessing with resizing, center cropping, and device placement
    - Class remapping support for label alignment across datasets
    - Memory-efficient lazy loading for large JSONL files
    - Integration with PyTorch DataLoader for batched training

Classes:
    SegDataset: Abstract base class for segmentation datasets
    VOC2012SegDataset: PASCAL VOC 2012 segmentation dataset
    COCO2017SegDataset: MS COCO 2017 segmentation dataset
    JSONLDataset: Efficient lazy-loading dataset for JSONL files
    ImageDataset: Simple dataset for loading image collections
    ImageCaptionDataset: Combined dataset for image-caption pairs

Example:
    >>> from pathlib import Path
    >>> import torch
    >>> 
    >>> # Load VOC 2012 dataset
    >>> dataset = VOC2012SegDataset(
    ...     root_path=Path("/data/VOCdevkit"),
    ...     split="train",
    ...     device=torch.device("cuda"),
    ...     resize_size=512
    ... )
    >>> image, mask = dataset[0]
"""

from core.config import *
from core.data import get_image, get_mask
from core.color_map import full_color_map
from core.torch_utils import map_tensors
from core.data import is_state, JsonlIO

import numpy as np
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from torchvision.datasets import VOCSegmentation

from pycocotools.coco import COCO

from core._types import ABC, Optional, Literal, RGB_tuple, Callable, Any, deprecated
import json
from glob import glob

class SegDataset(Dataset, ABC):
    """
    Abstract base class for segmentation datasets.
    
    This class provides common interface and utility methods for loading images,
    ground truth masks, and prediction masks for semantic segmentation tasks.
    All concrete segmentation dataset implementations should inherit from this class.
    
    Attributes:
        image_UIDs: Array of unique identifiers for images in the dataset.
    """
    def __init__(self) -> None:
        raise NotImplementedError
    
    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.image_UIDs)
    
    def get_sc(
        self,
        path: Path,
        device: torch.device,
        resize_size: None | int | tuple[int, int] = None,
        resize_mode: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
        center_crop: bool = True
    ) -> torch.Tensor:
        """
        Load and preprocess a scene image.
        
        Args:
            path: Path to the image file.
            device: Torch device to load the image tensor to.
            resize_size: Target size for resizing. Can be int or (height, width) tuple.
            resize_mode: Interpolation mode for resizing.
            center_crop: Whether to apply center cropping to make the image square.
            
        Returns:
            Preprocessed image tensor.
        """
        return get_image(path, device, resize_size, resize_mode, center_crop)
    
    def get_gt(
        self,
        path: str,
        device: torch.device,
        class_map: Optional[dict] = None,
        resize_size: Optional[int | tuple[int, int]] = None,
        resize_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
        center_crop: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Load and preprocess a ground truth segmentation mask.
        
        Args:
            path: Path to the mask file.
            device: Torch device to load the mask tensor to.
            class_map: Optional mapping from original class indices to new indices.
            resize_size: Target size for resizing. Can be int or (height, width) tuple.
            resize_mode: Interpolation mode for resizing (typically NEAREST for masks).
            center_crop: Whether to apply center cropping to make the mask square.
            
        Returns:
            Preprocessed ground truth mask tensor.
        """
        return get_mask(path, device, class_map, resize_size, resize_mode, center_crop)

    def get_pr(
        self,
        path: str,
        device: torch.device,
        class_map: Optional[dict] = None,
        resize_size: Optional[int | tuple[int, int]] = None,
        resize_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
        center_crop: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Load and preprocess a prediction segmentation mask.
        
        Args:
            path: Path to the prediction mask file.
            device: Torch device to load the mask tensor to.
            class_map: Optional mapping from original class indices to new indices.
            resize_size: Target size for resizing. Can be int or (height, width) tuple.
            resize_mode: Interpolation mode for resizing (typically NEAREST for masks).
            center_crop: Whether to apply center cropping to make the mask square.
            
        Returns:
            Preprocessed prediction mask tensor.
        """
        return get_mask(path, device, class_map, resize_size, resize_mode, center_crop)

def download_VOC2012(
        root_path: Path
) -> None:
    """
    Download the PASCAL VOC 2012 dataset.
    
    Uses torchvision's VOCSegmentation utility to download the dataset to the specified
    root directory. Downloads the 'trainval' split which includes both training and
    validation sets.
    
    Args:
        root_path: Root directory where the dataset will be downloaded and extracted.
    """
    VOCSegmentation(root=root_path, image_set='trainval', download=True)

class VOC2012SegDataset(SegDataset):
    """
    Dataset class for PASCAL VOC 2012 semantic segmentation.
    
    This dataset provides access to the PASCAL VOC 2012 segmentation benchmark with
    support for different splits, image preprocessing, and optional prediction masks.
    
    The dataset contains 21 semantic classes including background, with an optional
    22nd class for unlabeled pixels.
    
    Attributes:
        root_path: Root directory containing the VOC2012 dataset.
        split: Dataset split ('train', 'val', 'trainval', or 'prompts_split').
        image_UIDs: Array of image unique identifiers.
        sc_root_path: Path to scene images directory.
        gt_root_path: Path to ground truth masks directory.
        scs_paths: Array of paths to scene images.
        gts_paths: Array of paths to ground truth masks.
        prs_paths: Array of paths to prediction masks (if provided).
        class_map: Mapping from original to target class indices.
        resize_size: Target size for image/mask resizing.
        device: Torch device for loading tensors.
    """
    def get_image_UIDs(
            self,
            seed: int,
            split: Literal['trainval',
                           'train'
                           'val',
                           'prompts_split'] = "trainval",
            uids_to_exclude: list[str] = [],
    ) -> np.ndarray[str]:
        """
        Retrieve image UIDs for the specified dataset split.
        
        Reads image identifiers from the VOC2012 split files. For 'prompts_split',
        shuffles all but the first 23 images and returns the first 80.
        
        Args:
            seed: Random seed for shuffling in 'prompts_split' mode.
            split: Dataset split to load. Options are 'train', 'val', 'trainval',
                or 'prompts_split' (special split for prompt generation).
            uids_to_exclude: List of image UIDs to exclude from the dataset.
        
        Returns:
            Array of image UIDs for the specified split.
        """
        image_UIDs = []
        prompt_split = False
        if split == 'prompts_split':
            prompt_split = True
            split = 'trainval'
        with open(self.root_path / 'VOC2012' / 'ImageSets' / 'Segmentation' / f"{split}.txt", "r") as f:
            i = 0
            for line in f:
                i += 1
                image_id = line.strip()  # Remove any leading/trailing whitespace
                if image_id not in uids_to_exclude:
                    image_UIDs.append(image_id)
            image_UIDs = sorted(image_UIDs)
        if prompt_split:
            to_shuffle = image_UIDs[23:]
            rng = random.Random(seed)
            rng.shuffle(to_shuffle)
            image_UIDs[23:] = to_shuffle
            image_UIDs = image_UIDs[:80] # NOTE hard-coded
        return np.array(image_UIDs)
    
    def __init__(
            self,
            root_path: Path,
            split: Literal['train',
                           'val',
                           'trainval',
                           'prompts_split'],
            device: torch.device,
            resize_size: Optional[int | list[int, int]] = None,
            sc_resize_mode: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
            mask_resize_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
            img_idxs: Optional[list[int] | slice] = None, # if None, all samples are considered
            uids_to_exclude: list[str] = [],
            center_crop: bool = False,
            with_unlabelled: bool = True, # only effective for the masks class mapping
            mask_prs_path: Path = None,
            output_uids: bool = False
    ) -> None:
        """
        Initialize the VOC2012 segmentation dataset.
        
        Args:
            root_path: Root directory containing the VOCdevkit/VOC2012 dataset.
            split: Dataset split ('train', 'val', 'trainval', or 'prompts_split').
            device: Torch device to load tensors to.
            resize_size: Target size for resizing images/masks. If None, original size is kept.
            sc_resize_mode: Interpolation mode for resizing scene images.
            mask_resize_mode: Interpolation mode for resizing masks.
            img_idxs: Optional indices or slice to select a subset of images.
            uids_to_exclude: List of image UIDs to exclude from the dataset.
            center_crop: Whether to apply center cropping to make images/masks square.
            with_unlabelled: Whether to include unlabeled pixels as a separate class (22nd class).
            mask_prs_path: Optional path to prediction masks directory.
            output_uids: Whether to include image UIDs in the output.
        """
        self.root_path = root_path
        self.mask_prs_path = mask_prs_path
        self.split = split
        self.img_idxs = img_idxs
        self.device = device

        self.image_UIDs = self.get_image_UIDs(42, self.split, uids_to_exclude=uids_to_exclude)

        if self.img_idxs:
            self.image_UIDs = self.image_UIDs[self.img_idxs]

        self.sc_root_path = root_path / 'VOC2012' / 'JPEGImages'
        self.gt_root_path = root_path / 'VOC2012' / 'SegmentationClass'

        self.scs_paths = np.array([root_path / 'VOC2012' / 'JPEGImages' / f"{uid}.jpg" for uid in self.image_UIDs])
        self.gts_paths = np.array([root_path / 'VOC2012' / 'SegmentationClass' / f"{uid}.png" for uid in self.image_UIDs])

        self.mask_prs_path = mask_prs_path
        if mask_prs_path:
            if self.img_idxs:
                self.prs_paths = np.array([mask_prs_path / f'mask_pr_{img_i}.png' for img_i in img_idxs])
            else:
                self.prs_paths = np.array([mask_prs_path / f'mask_pr_{i}.png' for i, uid in enumerate(self.image_UIDs)])
        
        if len(self.scs_paths) != len(self.gts_paths):
            raise AttributeError(f"There is a different number of samples of scenes ({len(self.scs_paths)}) and ground truths ({len(self.gts_paths)}).")
        
        self.with_unlabelled = with_unlabelled
        self.class_map = self.get_class_map(with_unlabelled)
        
        self.resize_size = resize_size
        self.sc_resize_mode = sc_resize_mode
        self.mask_resize_mode = mask_resize_mode
        self.center_crop = center_crop
        self.output_uids = output_uids

    @classmethod
    def get_classes(
            self,
            with_unlabelled: bool = False
    ) -> list[str]:
        # 21 classes
        classes = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT", "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE", "DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP", "SOFA", "TRAIN", "TVMONITOR"] # 21 classes

        if with_unlabelled:
            classes += ["UNLABELLED"] # 22 classes

        return classes
    
    def get_class_map(
            self,
            with_unlabelled: bool = False
    ) -> dict[int, int]:
        class_map = {i: i for i in range(len(self.get_classes(with_unlabelled=False)))} | {255: 0} # default mapping

        if with_unlabelled:
            class_map = class_map | {255: 21} # 'UNLABELLED' class (idx. 255) is mapped to idx. 21 for continuity.

        return class_map

    def set_class_map(
            self,
            with_unlabelled: bool
    ) -> dict:
        self.class_map = self.get_class_map(with_unlabelled)

    def get_num_classes(
            self,
            with_unlabelled: bool = False
    ) -> int:
        num_classes = len(set(self.get_class_map(with_unlabelled=with_unlabelled).values())) # actual number of classes
        return num_classes

    def __getitem__(
            self,
            idx: int | list[int] | slice
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(idx, int):
            sc = self.get_sc(path=self.scs_paths[idx], device=self.device, resize_size=self.resize_size, center_crop=self.center_crop)
            gt = self.get_gt(path=self.gts_paths[idx], device=self.device, class_map=self.class_map, resize_size=self.resize_size, center_crop=self.center_crop)
            if self.mask_prs_path:
                pr = self.get_pr(path=self.prs_paths[idx], device=self.device, class_map=self.class_map, resize_size=self.resize_size, center_crop=self.center_crop)
                if self.output_uids:
                    return self.image_UIDs[idx], sc, gt, pr
                else:
                    return sc, gt, pr
            else:
                if self.output_uids:
                    return self.image_UIDs[idx], sc, gt
                else:
                    return sc, gt
        elif isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
        elif isinstance(idx, list):
            indices = idx
        scs = [self.get_sc(path=self.scs_paths[i], device=self.device, resize_size=self.resize_size, center_crop=self.center_crop) for i in indices]
        gts = [self.get_gt(path=self.gts_paths[i], device=self.device, class_map=self.class_map, resize_size=self.resize_size, center_crop=self.center_crop) for i in indices]
        if self.mask_prs_path:
            prs = [self.get_pr(path=self.prs_paths[i], device=self.device, class_map=self.class_map, resize_size=self.resize_size, center_crop=self.center_crop) for i in indices]
            if self.output_uids:
                return self.image_UIDs[indices], scs, gts, prs
            else:
                return scs, gts, prs
        else:
            if self.output_uids:
                return self.image_UIDs[indices], scs, gts
            else:
                return scs, gts
    
    def get_imgs_by_uid(
            self,
            uids: str | list[str] | np.ndarray[str]
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(uids, str):
            uid = uids
            sc = self.get_sc(path=self.sc_root_path / f'{uid}.jpg', device=self.device, resize_size=self.resize_size, center_crop=self.center_crop)
            gt = self.get_gt(path=self.gt_root_path / f'{uid}.png', device=self.device, class_map=self.class_map, resize_size=self.resize_size, center_crop=self.center_crop)
            if self.output_uids:
                return uids, sc, gt
            else:
                return sc, gt
        elif isinstance(uids, list):
            uids = np.array(uids)
        if isinstance(uids, np.ndarray):
            scs = [self.get_sc(path=self.sc_root_path / f'{uid}.jpg', device=self.device, resize_size=self.resize_size, center_crop=self.center_crop) for uid in uids]
            gts = [self.get_gt(path=self.gt_root_path / f'{uid}.png', device=self.device, class_map=self.class_map, resize_size=self.resize_size, center_crop=self.center_crop) for uid in uids]
            if self.output_uids:
                return uids, scs, gts
            else:
                return scs, gts
            
    @deprecated('Use the one above')
    def get_img_by_uid(
            self,
            uid: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sc = self.get_sc(path=self.sc_root_path / f"{uid}.jpg", device=self.device, resize_size=self.resize_size, center_crop=self.center_crop)
        gt = self.get_gt(path=self.gt_root_path / f"{uid}.png", device=self.device, class_map=self.class_map, resize_size=self.resize_size, center_crop=self.center_crop)
        return sc, gt
            
    def get_color_map_dict(
            self,
            with_unlabelled: bool = False
    ) -> dict[int, RGB_tuple]:
        """Gets the color map as dictionary {cls_idx: (r, g, b)} for the 21 VOC classes.

        Returns:
            Dictionary mapping class index to RGB tuple.
        """
        color_map_list = full_color_map()[:21].tolist()
        if with_unlabelled:
            color_map_list += [full_color_map()[255].tolist()]
        return {i: tuple(rgb) for i, rgb in enumerate(color_map_list)}


class COCO2017SegDataset(SegDataset):

    voc_idx_to_coco_idx = {
        0: 0, # background -> unlabeled
        1: 5, # aeroplane -> airplane
        2: 2, # bicycle -> bicycle
        3: 16, # bird -> bird
        4: 9, # boat -> boat
        5: 44, # bottle -> bottle
        6: 6, # bus -> bus
        7: 3, # car -> car
        8: 17, # cat -> cat
        9: 62, # chair -> chair
        10: 21, # cow -> cow
        11: 67, # diningtable -> dining table
        12: 18, # dog -> dog
        13: 19, # horse -> horse
        14: 4, # motorbike -> motorcycle
        15: 1, # person -> person
        16: 64, # pottedplant -> potted plant
        17: 20, # sheep -> sheep
        18: 63, # sofa -> couch
        19: 7, # train -> train
        20: 72 # tvmonitor -> tv
    }
    coco_idx_to_voc_idx = {c: v for v, c in voc_idx_to_coco_idx.items()}

    def __init__(
        self,
        root_path: Path,
        split: Literal['train',
                       'val'],
        device: torch.device,
        resize_size: Optional[int | list[int, int]] = None,
        sc_resize_mode: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
        mask_resize_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
        img_idxs: Optional[list[int] | slice] = None, # if None, all samples are considered
        uids_to_exclude: list[int] = None,
        center_crop: bool = False,
        with_unlabelled: bool = True, # as far, ineffective
        mask_prs_path: Path = None, # as far, ineffective
        only_VOC_labels: bool = False
    ) -> None:
                
        self.root_path = root_path
        self.split = split
        self.device = device

        self.coco = COCO(self.root_path / 'annotations' / f'instances_{self.split}2017.json')
        
        self.only_VOC_labels = only_VOC_labels
        if self.only_VOC_labels: # consider only labels present in VOC and images showing them. No other label is displayed at all.
            self.cat_ids = list(set(COCO2017SegDataset.voc_idx_to_coco_idx.values())) # only keeps cat_ids also present in VOC2012
            self.cat_ids.remove(0)
            self.image_UIDs = self.get_image_UIDs(self.cat_ids) # only retrieves images with labels mapped in VOC2012
        else:
            self.cat_ids = self.coco.getCatIds()
            self.image_UIDs = self.get_image_UIDs()
        
        self.cats = self.coco.loadCats(self.cat_ids)
        
        self.img_idxs = img_idxs
        if img_idxs:
            self.image_UIDs = self.image_UIDs[self.img_idxs]

        self.uids_to_exclude = uids_to_exclude
        if self.uids_to_exclude is not None:
            mask_exclude = np.isin(self.image_UIDs, uids_to_exclude)
            self.image_UIDs = self.image_UIDs[~mask_exclude]
    
        self.scs_root_path = self.root_path / f"{self.split}2017"

        self.resize_size = resize_size
        self.sc_resize_mode = sc_resize_mode
        self.mask_resize_mode = mask_resize_mode
        self.center_crop = center_crop

        self.class_map = self.get_class_map(with_unlabelled)

    def get_image_UIDs(
            self,
            only_cat_ids_to_keep: list[int] = None
    ) -> np.ndarray[int]:
        img_ids = np.array(self.coco.getImgIds())

        # NOTE: even considering all classes, the images effectively retrieved are less than the total number in all splits
        # because some images only contain the BACKGROUND (0) class. These images are not retrieve by 'getImgIds'.
        # This should not be a problem for the image-text generation pipeline because only-background images are discarded anyway.
        if only_cat_ids_to_keep:
            img_ids_keep = np.array(list(set().union(*[self.coco.getImgIds(catIds=cat_ids) for cat_ids in only_cat_ids_to_keep])))
            mask_keep = np.isin(img_ids, img_ids_keep)
            img_ids = img_ids[mask_keep]
        
        return np.array(img_ids)

    def get_classes(
            self,
            with_unlabelled: bool = False
    ) -> list[str]:
        # 81 classes
        classes = ['BACKGROUND'] + [d['name'].upper() for d in self.cats]

        return classes
    
    def get_class_map(
            self,
            with_unlabelled: bool = False
    ) -> dict[int, int]:
        class_map = {0: 0} | {c_d['id']: i+1 for i, c_d in enumerate(self.cats)}

        return class_map
    
    def get_num_classes(
            self,
            with_unlabelled: bool = False
    ) -> int:
        num_classes = len(set(self.get_class_map(with_unlabelled=with_unlabelled).values())) # actual number of classes
        return num_classes
    
    def get_color_map_dict(
            self,
            with_unlabelled: bool = False
    ) -> dict[int, RGB_tuple]:
        """Gets the color map as dictionary {cls_idx: (r, g, b)} for the 21 VOC classes.

        Returns:
            Dictionary mapping class index to RGB tuple.
        """
        color_map_list = full_color_map()[:81].tolist()
        if with_unlabelled:
            color_map_list += [full_color_map()[255].tolist()]
        return {i: tuple(rgb) for i, rgb in enumerate(color_map_list)}
    
    def _create_COCO_masks(
            self,
            sc_dict: dict,
    ) -> torch.Tensor:
        """
        Generates a semantic segmentation mask from COCO annotations.

        This method is optimized by:
        1. Performing the iterative mask creation on the CPU using NumPy, which is
        very fast for these operations.
        2. Transferring the final, complete mask to the GPU in a single operation.
        3. Using a non-blocking transfer to allow the CPU to continue working while
        the data is being copied by the GPU's DMA engine.
        
        This approach is generally faster than creating the mask directly on the GPU
        due to the high overhead of many small CPU->GPU transfers.
        """
        # device = torch.device()
        h, w = sc_dict['height'], sc_dict['width']

        # 1. Get all annotations for the given image and categories
        ann_ids = self.coco.getAnnIds(imgIds=sc_dict['id'], catIds=self.cat_ids, iscrowd=None)
        if not ann_ids:
            # Return an empty mask if there are no annotations
            return torch.zeros((1, h, w), dtype=torch.uint8, device=self.device)
            
        anns = self.coco.loadAnns(ann_ids)

        # 2. Efficiently create the mask on the CPU using NumPy.
        # The for-loop and boolean indexing in NumPy are highly optimized C operations.
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            # coco.annToMask() is a CPU operation that returns a NumPy array
            instance_mask = self.coco.annToMask(ann)
            
            # "Paint" the category ID onto the mask.
            # This overwrites previous annotations in case of overlap, matching
            # the original logic where the last annotation in the list wins.
            mask[instance_mask == 1] = ann['category_id']

        # 3. Transfer the completed mask to the GPU in one go.
        # torch.from_numpy is slightly more efficient than torch.tensor for this.
        # .to(..., non_blocking=True) is key for performance in data loading pipelines.
        gt = torch.from_numpy(mask).to(self.device, non_blocking=True).unsqueeze(0)
        
        # The dtype will be torch.uint8, which is what we want.
        return gt
    
    def get_gt(
        self,
        sc_dict: dict,
        class_map: Optional[dict] = None,
        resize_size: Optional[int | tuple[int, int]] = None,
        resize_mode: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
        center_crop: Optional[bool] = None
    ) -> torch.Tensor:
        gt: torch.Tensor = self._create_COCO_masks(sc_dict)
        gt = gt.to(self.device)
        if class_map:
           gt = map_tensors(gt, class_map)
        if resize_size:
            gt = TF.resize(gt, resize_size, resize_mode)
        if center_crop:
            gt = TF.center_crop(gt, output_size=min(gt.shape[-2:]))
        return gt
    
    def __getitem__(
            self,
            idx: int | list[int] | slice
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(idx, int):
            sc_dict = self.coco.loadImgs([int(self.image_UIDs[idx])])[0] # the output is a singleton list
            sc = self.get_sc(self.scs_root_path / sc_dict["file_name"], self.device, resize_size=self.resize_size, resize_mode=self.sc_resize_mode, center_crop=self.center_crop)
            gt = self.get_gt(sc_dict, self.class_map, resize_size=self.resize_size, resize_mode=self.mask_resize_mode, center_crop=self.center_crop)
            return sc, gt
        elif isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
        elif isinstance(idx, list):
            indices = idx
        
        scs_dict = self.coco.loadImgs([int(uid) for uid in self.image_UIDs[indices]])
        scs = [self.get_sc(self.scs_root_path / sc_d["file_name"], self.device, resize_size=self.resize_size, resize_mode=self.mask_resize_mode, center_crop=self.center_crop) for sc_d in scs_dict]            
        gts = [self.get_gt(sc_d, self.class_map, resize_size=self.resize_size, resize_mode=self.mask_resize_mode, center_crop=self.center_crop) for sc_d in scs_dict]

        return scs, gts

    def get_imgs_by_uid(
            self,
            uids: str | list[str] | np.ndarray[str]
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(uids, str):
            uid = uids
            sc_dict = self.coco.loadImgs([int(uid)])[0] # the output is a singleton list
            sc = self.get_sc(self.scs_root_path / sc_dict["file_name"], self.device, resize_size=self.resize_size, resize_mode=self.sc_resize_mode, center_crop=self.center_crop)
            gt = self.get_gt(sc_dict, self.class_map, resize_size=self.resize_size, resize_mode=self.mask_resize_mode, center_crop=self.center_crop)
            return sc, gt
        elif isinstance(uids, list):
            uids = np.array(uids)
        if isinstance(uids, np.ndarray):
            scs_dict = self.coco.loadImgs([int(uid) for uid in uids])
            scs = [self.get_sc(self.scs_root_path / sc_d["file_name"], self.device, resize_size=self.resize_size, resize_mode=self.mask_resize_mode, center_crop=self.center_crop) for sc_d in scs_dict]            
            gts = [self.get_gt(sc_d, self.class_map, resize_size=self.resize_size, resize_mode=self.mask_resize_mode, center_crop=self.center_crop) for sc_d in scs_dict]
            return scs, gts
    
    def get_img_by_uid(
            self,
            uid: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sc_dict = self.coco.loadImgs([int(uid)])[0] # the output is a singleton list
        sc = self.get_sc(self.scs_root_path / sc_dict["file_name"], self.device, resize_size=self.resize_size, resize_mode=self.sc_resize_mode, center_crop=self.center_crop)
        gt = self.get_gt(sc_dict, self.class_map, resize_size=self.resize_size, resize_mode=self.mask_resize_mode, center_crop=self.center_crop)
        return sc, gt
    

def get_answer_objects(
        path: Path,
        idxs: Optional[int | list[int]],
        jsonlio: JsonlIO,
        return_state: bool = False,
        format_to_dict: bool = False
) -> dict | list[dict]:
    if idxs is None:
        answer_gts = jsonlio.get_many_item(path, return_state, format_to_dict)
        return answer_gts
    if isinstance(idxs, int):
        answer_gt = jsonlio.get_one_item(path, idxs, return_state, format_to_dict)
        return answer_gt
    answer_gts = jsonlio.get_many_item(path, return_state, format_to_dict)
    # if isinstance(idxs, slice):
        # indices = range(*idxs.indices(len(self)))
    if isinstance(idxs, list):
        indices = idxs
    answer_gts = [gt for i, gt in enumerate(answer_gts) if i in indices]
    return answer_gts


class JSONLDataset(Dataset):
    """
    An efficient PyTorch Dataset for lazy-loading and parsing large JSONL (JSON Lines) files.

    This class avoids loading the entire file into memory. Instead, it creates an
    index of the byte offsets for each line during initialization. When an item
    is requested, it seeks directly to the corresponding line's position in the
    file and reads only that line.

    This makes it highly memory-efficient and fast for random access, which is
    essential for use with PyTorch's DataLoader, especially with `shuffle=True`.

    It also supports slicing (e.g., `dataset[10:20]`).

    Args:
        file_path (Path): The path to the JSONL file.
        transform (Optional[Callable]): An optional function to be applied to the
            data object after it is loaded and parsed.
    """
    def __init__(
            self,
            file_path: Path,
            transform: Optional[Callable] = None,
            line_idxs: Optional[list[int] | slice] = None
    ):
        super().__init__()
        self.file_path = file_path
        self.transform = transform
        self._file = None  # File handle will be opened lazily by each worker
        self.state_line = None


        # Create an index of byte offsets for each line
        self.line_offsets = self._build_index()

        self.line_idxs = line_idxs
        if line_idxs:
            self.line_offsets = self.line_offsets[self.line_idxs]

    def _build_index(self) -> list[int]:
        """
        Scans the file once to create an index of the starting byte offset
        for each line. If the first line is a state object, it's stored
        and not included in the data indices.
        """
        line_offsets = []
        # Open in binary mode to correctly handle byte offsets
        with open(self.file_path, 'rb') as f:
            # Check first line for state
            first_line_offset = f.tell()
            first_line = f.readline()

            if not first_line:
                return []

            try:
                # Need to decode for json.loads
                state_candidate = json.loads(first_line.decode('utf-8'))
                if is_state(state_candidate):
                    self.state_line = state_candidate
                    # State found, start indexing from the next line
                    offset = f.tell()
                else:
                    # No state, first line is data
                    line_offsets.append(first_line_offset)
                    offset = f.tell()
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If it's not valid JSON or can't be decoded, treat as data
                line_offsets.append(first_line_offset)
                offset = f.tell()

            while True:
                line = f.readline()
                if not line:
                    break
                line_offsets.append(offset)
                offset = f.tell()
        return line_offsets

    def __len__(self) -> int:
        """Returns the total number of lines in the file."""
        return len(self.line_offsets)

    def _get_single_item(
            self,
            idx: int
    ) -> dict[str, Any]:
        """
        Retrieves, parses, and optionally transforms a single item by its index.
        This is the core logic for data retrieval.
        """
        # Handle negative indices
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} is out of range for a dataset of size {len(self)}.")

        # Lazily open the file handle in each worker process
        if self._file is None:
            # We open in text mode here for json.loads, which expects strings.
            # The offsets were calculated in binary mode for accuracy.
            self._file = open(self.file_path, 'r', encoding='utf-8')

        # Seek to the pre-calculated byte offset
        offset = self.line_offsets[idx]
        self._file.seek(offset)

        # Read the line and parse the JSON
        line = self._file.readline()
        data = json.loads(line)

        # Apply transformation if it exists
        if self.transform:
            data = self.transform(data)

        return data

    def __getitem__(
            self,
            idx: int | slice
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Supports both integer indexing and slicing.
        - If idx is an integer, returns a single data item.
        - If idx is a slice, returns a list of data items.
        """
        if isinstance(idx, int):
            return self._get_single_item(idx)
        elif isinstance(idx, slice):
            # Resolve slice indices
            start, stop, step = idx.indices(len(self))
            return [self._get_single_item(i) for i in range(start, stop, step)]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}. Must be int or slice.")

    def __del__(self) -> None:
        """Ensures the file handle is closed when the dataset object is garbage collected."""
        if self._file:
            self.close()

    def close(self) -> None:
        """Closes the file handle."""
        if self._file:
            self._file.close()
            self._file = None

class ImageDataset(Dataset):
    def __init__(
            self,
            path: Path,
            device: torch.device,
            resize_size: Optional[int | list[int, int]] = None,
            resize_mode: Optional[str] = None,
            idxs: Optional[list[int] | slice] = None,
            center_crop: bool = False,
    ) -> None:
        self.image_paths = np.array(sorted(glob(str(path / "*.png"))))
        self.device = device
        if idxs:
            self.image_paths = self.image_paths[idxs]
        self.resize_size = resize_size
        self.resize_mode = resize_mode
        self.center_crop = center_crop

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
            self,
            idx: int
    ) -> list[torch.Tensor] | torch.Tensor:
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            imgs = [get_image(path=self.image_paths[i], device=self.device, resize_size=self.resize_size, center_crop=self.center_crop) for i in indices]
            return imgs
        else:
            img = get_image(path=self.image_paths[idx], device=self.device, resize_size=self.resize_size, center_crop=self.center_crop)
            return img


class ImageCaptionDataset(Dataset):
    """
    A dataset that merges a segmentation dataset and a JSONL dataset.
    It retrieves items from both datasets for a given index, supporting both
    integer indexing and slicing.
    """
    def __init__(
            self,
            img_dataset: ImageDataset | SegDataset,
            jsonl_dataset: JSONLDataset
    ) -> None:
        self.jsonl_dataset = jsonl_dataset
        
        if isinstance(img_dataset, ImageDataset):
            self.img_dataset = img_dataset
            
            # sort in the same order found in the jsonl dataset
            uidposc_2_i = {f'{d["img_uid"]}-{d["pos_class"]}': i for i, d in enumerate(self.jsonl_dataset)}
            self.img_dataset.image_paths = sorted(self.img_dataset.image_paths, key=lambda x: self.sort_fn(x, uidposc_2_i))

            if len(self.img_dataset) != len(self.jsonl_dataset):
                raise AttributeError(f"The segmentation and JSONL datasets have different lengths.\nImageDataset: {len(self.img_dataset)}, JSONLDataset: {len(self.jsonl_dataset)}.")
            
            if any(str(Path(img).stem) != f'{jsl["img_uid"]}-{jsl["pos_class"]}' for img, jsl in zip(self.img_dataset.image_paths, self.jsonl_dataset)):
                raise AttributeError("The images and JSONL samples are in a different order.")
        elif isinstance(img_dataset, SegDataset):
            img_dataset.image_UIDs = [d["img_uid"] for d in self.jsonl_dataset]
            self.img_dataset = img_dataset
    
    def sort_fn(
            self,
            p: str,
            uidposc_2_i: dict
    ):
        p = Path(p).stem
        return uidposc_2_i[p]

    def __len__(self) -> int:
        return min(len(self.img_dataset), len(self.jsonl_dataset))

    def __getitem__(self, idx: int | slice) -> tuple | list[tuple]:
        """
        Retrieves items from both datasets.
        - If idx is an integer, returns a single tuple: (image, jsonl_data).
        - If idx is a slice, returns a list of such tuples.
        """
        img_data = self.img_dataset[idx]
        jsonl_data = self.jsonl_dataset[idx]

        if isinstance(idx, slice):
            if isinstance(self.img_dataset, SegDataset):
                img_data, _ = zip(*img_data)
            # jsonl_data is a list of dicts
            return [(id, jd) for id, jd in zip(img_data, jsonl_data)]
        else:
            if isinstance(self.img_dataset, SegDataset):
                img_data, _ = img_data
            # jsonl_data is a dict
            return (img_data, jsonl_data)
