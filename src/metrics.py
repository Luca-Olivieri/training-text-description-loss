from config import *

from typing import Optional

import torch
import torchmetrics as tm

class MyMeanIoU(tm.Metric):

    def __init__(
            self,
            num_classes: int,
            per_class: bool = False,
            ignore_index: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.per_class = per_class
        self.ignore_index = ignore_index

        self.intersection_counts_per_class = torch.zeros(self.num_classes, device=CONFIG['device'])
        self.union_counts_per_class = torch.zeros(self.num_classes, device=CONFIG['device'])

    def update(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor
    ) -> None:
        preds_flat = preds.view(-1) # flatten the preds
        targets_flat = targets.view(-1) # flatten the targets

        if self.ignore_index:
            preds_flat[targets_flat == self.ignore_index] = self.ignore_index # make preds and targets agree on the ignored cells

        intersection = preds_flat[preds_flat == targets_flat].float() # flat accuracy mask

        intersection_counts_per_class_ = torch.histc(intersection, bins=self.num_classes, min=0, max=self.num_classes-1)
        preds_counts_per_class = torch.histc(preds_flat.float(), bins=self.num_classes, min=0, max=self.num_classes-1)
        targets_counts_per_class = torch.histc(targets_flat.float(), bins=self.num_classes, min=0, max=self.num_classes-1)

        self.intersection_counts_per_class += intersection_counts_per_class_
        self.union_counts_per_class += (preds_counts_per_class + targets_counts_per_class - intersection_counts_per_class_)

    def compute(
            self
    ) -> torch.Tensor:
        
        if self.ignore_index and 0 <= self.ignore_index < self.num_classes:
            mask_keep = torch.ones(self.num_classes, dtype=torch.bool)
            mask_keep[self.ignore_index] = False

            valid_intersection_counts_per_class = self.intersection_counts_per_class[mask_keep]
            valid_union_counts_per_class = self.union_counts_per_class[mask_keep]

            iou_per_class = valid_intersection_counts_per_class / valid_union_counts_per_class
        else:
            iou_per_class = self.intersection_counts_per_class / self.union_counts_per_class

        if self.per_class:
            return iou_per_class
        else:
            return iou_per_class.nanmean()
        
    def reset(self) -> None:
        self.intersection_counts_per_class = torch.zeros(self.num_classes, device=CONFIG['device'])
        self.union_counts_per_class = torch.zeros(self.num_classes, device=CONFIG['device'])
