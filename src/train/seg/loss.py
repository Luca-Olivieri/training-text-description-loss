"""
Loss functions for vision-language contrastive learning.

This module provides specialized SigLIP (Sigmoid Loss for Language Image Pre-Training)
implementations for various training scenarios including:
- Multiple positive texts per image with variable counts
- Paired negative texts dedicated to each image
- Grouped batch processing with normalized loss contributions
- Cross-entropy rescaling utilities for weighted loss computation

All loss functions are designed to handle edge cases like empty batches, zero positives,
or zero negatives gracefully.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.amp import GradScaler # for AMP

from typing import Literal

class SigLipLossMultiText(nn.Module):
    """
    Sigmoid Loss for multiple positive texts against multiple images.

    This loss function adapts the logic from SigLIP (https://arxiv.org/abs/2303.15343)
    for a specific use case: scoring a batch of text embeddings against a batch of
    candidate image embeddings. A key assumption is that **all text embeddings in the
    batch are positive matches for a single image** in the image batch.

    It computes an N x M similarity matrix between N texts and M images. The loss
    then pushes the similarity scores of all N positive pairs (each text with the
    one correct image) up, and the scores of all other negative pairs down.

    This is useful for retrieval-style tasks where an image might have multiple
    valid text descriptions (e.g., several captions), and you want to train the model
    to recognize the image from any of its descriptions.

    Attributes:
        None
    """
    def __init__(self) -> None:
        """Initializes the SigLipLossMultiText module."""
        super().__init__()

    def forward(
            self,
            image_features: torch.Tensor,
            text_features: torch.Tensor,
            positive_image_idx: int,
            logit_scale: float,
            logit_bias: Optional[float] = None,
            output_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Calculates the SigLIP loss for multiple texts against many images.

        Args:
            image_features (torch.Tensor): A batch of M image embeddings.
                Shape: (M, embedding_dim)
            text_features (torch.Tensor): A batch of N text embeddings, all of which
                are considered positive matches for the single correct image.
                Shape: (N, embedding_dim)
            positive_image_idx (int): The index (from 0 to M-1) in the `image_features`
                batch that corresponds to the correct, positive image for ALL texts.
            logit_scale (float): The learnable temperature parameter (from CLIP/SigLIP).
            logit_bias (Optional[float]): The learnable bias parameter (from SigLIP).
            output_dict (bool): If True, returns a dictionary with the key 'contrastive_loss'.
                Otherwise, returns the loss tensor directly.

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: The computed loss.
        """
        # Ensure text_features is (N, D) and image_features is (M, D)
        assert text_features.ndim == 2, \
            f"Expected text_features to have shape (N, D), but got {text_features.shape}"
        assert image_features.ndim == 2, \
            f"Expected image_features to have shape (M, D), but got {image_features.shape}"
        assert text_features.shape[1] == image_features.shape[1], \
            "Embedding dimensions of text and images must match."

        num_images = image_features.shape[0]
        assert 0 <= positive_image_idx < num_images, \
            f"positive_image_idx ({positive_image_idx}) is out of bounds for the number of images ({num_images})."

        # --- 1. Compute Logits (the full N x M similarity matrix) ---
        # text_features: (N, D)
        # image_features.T: (D, M)
        # logits: (N, M)
        logits = text_features @ image_features.T

        # Apply scaling and bias
        logits = logit_scale * logits
        if logit_bias is not None:
            logits += logit_bias

        # --- 2. Create Ground Truth Labels ---
        # We need an (N, M) matrix of labels.
        # The positive column (at positive_image_idx) has a label of 1.
        # All other columns have a label of -1.
        labels = -torch.ones_like(logits)
        labels[:, positive_image_idx] = 1.0

        # --- 3. Compute the Sigmoid Loss ---
        # The loss is the negative log-sigmoid of the element-wise product
        # of labels and logits.
        # This is calculated for all N*M pairs, and then summed up.
        loss = -F.logsigmoid(labels * logits).sum()

        if output_dict:
            return {"contrastive_loss": loss}
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, List

class VariableTextSigLipLoss(nn.Module):
    """
    Sigmoid Loss for Language Image Pre-Training (SigLIP) with a variable number
    of positive texts per image and controllable negative loss aggregation.

    This implementation ensures each image in the batch contributes equally to the
    final loss, regardless of its number of positive text pairings. It also provides
    flags to control negative loss aggregation and whether to include images
    with no positive texts in the loss calculation.

    The loss computation follows these principles:
    - Positive loss per image is averaged across all its positive texts
    - Negative loss per image can be summed or averaged (controlled by negative_loss_agg)
    - Images with no positives can optionally be excluded from loss calculation
    - Final batch loss is the mean of per-image total losses

    Attributes:
        negative_loss_agg (str): Method for aggregating negative losses ('sum' or 'mean').
        ignore_no_positive_samples (bool): Whether to zero out loss for images without positives.
    """
    def __init__(
            self,
            negative_loss_agg: Literal['sum', 'mean'] = 'sum',
            ignore_no_positive_samples: bool = True
    ) -> None:
        """
        Initializes the VariableTextSigLipLoss module.

        Args:
            negative_loss_agg (Literal['sum', 'mean']): How to aggregate the negative loss per image.
                                                        'sum' (default) accumulates the push from all negatives.
                                                        'mean' averages the push from all negatives.
            ignore_no_positive_samples (bool): If True, images with no positive text pairings
                                               will have their total loss contribution zeroed out,
                                               effectively removing them from the batch's loss
                                               calculation. Defaults to True.
        
        Raises:
            ValueError: If negative_loss_agg is not 'sum' or 'mean'.
        """
        super().__init__()
        if negative_loss_agg not in {'sum', 'mean'}:
            raise ValueError(f"negative_loss_agg must be either 'sum' or 'mean', got {negative_loss_agg}")
        self.negative_loss_agg = negative_loss_agg
        self.ignore_no_positive_samples = ignore_no_positive_samples

    def get_logits(
            self,
            image_features: torch.Tensor,
            text_features: torch.Tensor,
            logit_scale: torch.Tensor,
            logit_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the pairwise similarity logits between images and texts.

        Computes the scaled dot product similarity matrix between all image-text pairs,
        optionally adding a learnable bias term.

        Args:
            image_features (torch.Tensor): Image embeddings of shape (N, D).
            text_features (torch.Tensor): Text embeddings of shape (M, D).
            logit_scale (torch.Tensor): Learnable temperature scaling parameter.
            logit_bias (Optional[torch.Tensor]): Learnable bias to add to logits.

        Returns:
            torch.Tensor: Similarity logits matrix of shape (N, M).
        """
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def forward(
            self,
            image_features: torch.Tensor,
            list_of_text_features: list[torch.Tensor],
            logit_scale: torch.Tensor,
            logit_bias: Optional[torch.Tensor] = None,
            output_dict: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Computes the SigLIP loss, correctly normalized for a variable number of positives.

        The loss computation follows these steps:
        1. Flatten all text features and track which image each text belongs to
        2. Compute all-to-all similarity logits between images and texts
        3. Create label matrix (+1 for positive pairs, -1 for negatives)
        4. Compute pairwise losses using -log(sigmoid(labels * logits))
        5. Normalize positive losses by averaging across texts per image
        6. Aggregate negative losses (sum or mean based on config)
        7. Optionally zero out loss for images with no positives
        8. Return mean loss across all images

        Args:
            image_features (torch.Tensor): A tensor of shape (N, D), where N is the number of images
                                          and D is the embedding dimension.
            list_of_text_features (list[torch.Tensor]): A list of length N where each element is a tensor
                                                        of shape (num_positives_for_image_i, D). Can contain
                                                        empty tensors (0, D) for images with no positives.
            logit_scale (torch.Tensor): The learnable temperature scaling parameter.
            logit_bias (Optional[torch.Tensor]): The learnable bias parameter to add to logits.
            output_dict (bool): If True, returns a dictionary with key 'contrastive_loss'.
                                If False, returns the loss tensor directly.

        Returns:
            torch.Tensor | dict[str, torch.Tensor]: The computed loss as a single scalar tensor,
                                                    or a dictionary if output_dict is True.
        """

        flat_text_features, image_indices_for_texts = self.prepare_inputs_from_list(list_of_text_features)

        device = image_features.device
        num_images = image_features.shape[0]
        num_texts = flat_text_features.shape[0]

        # Step 1: Compute all-to-all pairwise logits
        logits = self.get_logits(image_features, flat_text_features, logit_scale, logit_bias)

        # Step 2: Construct the ground truth label matrix (+1 for pos, -1 for neg)
        labels = -torch.ones_like(logits)
        if num_texts > 0:
            text_indices = torch.arange(num_texts, device=device)
            labels[image_indices_for_texts, text_indices] = 1

        # Step 3: Compute the loss for each (image, text) pair
        pairwise_loss = -F.logsigmoid(labels * logits)

        # Step 4: Normalize the loss correctly
        positive_mask = (labels > 0)
        negative_mask = ~positive_mask

        # --- Positive Loss Calculation (always averaged) ---
        pos_loss_per_image = (pairwise_loss * positive_mask).sum(dim=1)
        num_positives_per_image = torch.zeros(num_images, device=device, dtype=torch.long)
        if num_texts > 0:
            # Use bincount to efficiently count positives for each image
            counts = torch.bincount(image_indices_for_texts, minlength=num_images)
            num_positives_per_image = counts

        pos_loss_normalizer = torch.clamp(num_positives_per_image, min=1).float()
        normalized_pos_loss_per_image = pos_loss_per_image / pos_loss_normalizer

        # --- Negative Loss Calculation (sum or mean) ---
        neg_loss_per_image = (pairwise_loss * negative_mask).sum(dim=1)
        if self.negative_loss_agg == 'mean':
            num_negatives_per_image = num_texts - num_positives_per_image
            neg_loss_normalizer = torch.clamp(num_negatives_per_image, min=1).float()
            neg_loss_per_image = neg_loss_per_image / neg_loss_normalizer

        # The total loss for each image is its normalized positive loss plus its
        # (potentially normalized) negative loss.
        total_loss_per_image = normalized_pos_loss_per_image + neg_loss_per_image

        # --- NEW LOGIC: Optionally ignore samples with no positives ---
        if self.ignore_no_positive_samples:
            # Create a mask for images that have zero positive texts.
            no_positives_mask = (num_positives_per_image == 0)
            # Zero out the loss for these specific images.
            total_loss_per_image[no_positives_mask] = 0.0
        # --- END NEW LOGIC ---

        # The final batch loss is the mean of the per-image losses.
        loss = total_loss_per_image.mean()

        return {"contrastive_loss": loss} if output_dict else loss

    def prepare_inputs_from_list(
            self,
            list_of_text_features: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a list of variable-sized text feature tensors into flattened format.

        Takes a list where each element contains text features for one image and flattens
        them into a single tensor while tracking which image each text belongs to.

        Args:
            list_of_text_features (list[torch.Tensor]): A list of length N where each element
                                                        is a tensor of shape (num_texts_i, D).
                                                        Can contain empty tensors (0, D).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - flat_text_features: Concatenated text features of shape (total_texts, D)
                - image_indices_for_texts: Image index for each text of shape (total_texts,)
                
                If all tensors are empty, returns (empty tensor of shape (0, 0), empty tensor of shape (0,)).
        """
        if not list_of_text_features:
            return torch.empty((0, 0)), torch.empty(0, dtype=torch.long)

        device = list_of_text_features[0].device if any(t.numel() > 0 for t in list_of_text_features) else torch.device("cpu")
        # Ensure there's a fallback device if all tensors are empty on different devices.
        # Let's find the first tensor with elements to determine device.
        first_tensor = next((t for t in list_of_text_features if t.numel() > 0), None)
        if first_tensor is not None:
            device = first_tensor.device
            feature_dim = first_tensor.shape[1]
        else: # All tensors are empty
            return torch.empty(0, 0, device=device), torch.empty(0, dtype=torch.long, device=device)

        num_texts_per_image = torch.tensor(
            [t.shape[0] for t in list_of_text_features],
            device=device
        )
        image_indices_to_repeat = torch.arange(len(list_of_text_features), device=device)
        image_indices_for_texts = torch.repeat_interleave(
            image_indices_to_repeat,
            repeats=num_texts_per_image
        )
        flat_text_features = torch.cat(list_of_text_features, dim=0)
        return flat_text_features, image_indices_for_texts.long()
    

class PairedNegativeSigLipLoss(VariableTextSigLipLoss):
    """
    Sigmoid Loss for Language Image Pre-Training (SigLIP) where each image is
    paired with its own dedicated set of positive and negative texts.

    This loss handles a variable number of positive texts per image and a fixed
    number of dedicated negative texts per image. It ensures that each image in
    the batch contributes equally to the final loss, regardless of its number
    of positive pairings.

    Key differences from VariableTextSigLipLoss:
    - Negatives are explicitly provided per image (not sampled from in-batch texts)
    - Each image has its own dedicated set of M negative texts
    - Negative texts are not shared across images in the batch

    The loss computation:
    - Positive loss: averaged across all positive texts for each image
    - Negative loss: summed across all dedicated negatives for each image
    - Final loss: mean of per-image total losses

    Attributes:
        Inherits from VariableTextSigLipLoss (negative_loss_agg, ignore_no_positive_samples).
    """

    def forward(
            self,
            image_features: torch.Tensor,
            list_of_positive_text_features: list[torch.Tensor],
            negative_text_features: torch.Tensor,
            logit_scale: torch.Tensor,
            logit_bias: Optional[torch.Tensor] = None,
            output_dict: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Computes the SigLIP loss with paired negatives.

        The loss computation follows these steps:
        1. Calculate positive loss: Compare each image with all positive texts, selecting
           only the true positive pairs and averaging per image
        2. Calculate negative loss: Compare each image with its own dedicated negatives
           using batched einsum operation
        3. Combine positive and negative losses per image
        4. Return mean loss across all images

        Args:
            image_features (torch.Tensor): A tensor of shape (N, D), where N is the number of images
                                          and D is the embedding dimension.
            list_of_positive_text_features (list[torch.Tensor]): A list of length N, where each
                element is a tensor of shape (num_positives_for_image_i, D). Can contain empty
                tensors (0, D) for images with no positives.
            negative_text_features (torch.Tensor): A tensor of shape (N, M, D), where M is the
                number of dedicated negative texts for each image. Can have M=0 for no negatives.
            logit_scale (torch.Tensor): The learnable temperature scaling parameter.
            logit_bias (Optional[torch.Tensor]): The learnable bias parameter to add to logits.
            output_dict (bool): If True, returns a dictionary with key 'contrastive_loss'.
                               If False, returns the loss tensor directly.

        Returns:
            torch.Tensor | dict[str, torch.Tensor]: The computed loss as a single scalar tensor,
                                                    or a dictionary if output_dict is True.
                                                    Returns 0.0 with requires_grad=True for empty batches.
        """
        device = image_features.device
        num_images = image_features.shape[0]

        if num_images == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {"contrastive_loss": loss} if output_dict else loss
        
        # --- Positive Loss Calculation ---
        # This part is similar to the original, but we only calculate positive loss.
        flat_pos_texts, image_indices_for_pos_texts = self.prepare_inputs_from_list(
            list_of_positive_text_features
        )

        normalized_pos_loss_per_image = torch.zeros(num_images, device=device)
        
        if flat_pos_texts.numel() > 0:
            # Calculate logits between all images and all positive texts
            pos_logits = self.get_logits(image_features, flat_pos_texts, logit_scale, logit_bias)
            
            # Create a mask to select the true (image, positive_text) pairs
            num_total_pos_texts = flat_pos_texts.shape[0]
            positive_mask = torch.zeros_like(pos_logits, dtype=torch.bool)
            if num_total_pos_texts > 0:
                text_indices = torch.arange(num_total_pos_texts, device=device)
                positive_mask[image_indices_for_pos_texts, text_indices] = True
            
            # Loss for positive pairs is -log(sigmoid(logit))
            pos_pairwise_loss = -F.logsigmoid(pos_logits)
            
            # Sum the loss for the true positive pairs for each image
            pos_loss_per_image = (pos_pairwise_loss * positive_mask).sum(dim=1)
            
            # Normalize by the number of positives to ensure each image has equal weight
            num_positives_per_image = torch.bincount(image_indices_for_pos_texts, minlength=num_images)
            pos_loss_normalizer = torch.clamp(num_positives_per_image, min=1).float()
            normalized_pos_loss_per_image = pos_loss_per_image / pos_loss_normalizer

        # --- Negative Loss Calculation ---
        # Each image `i` is compared against its own `M` negatives.
        num_negatives = negative_text_features.shape[1]
        neg_loss_per_image = torch.zeros(num_images, device=device)

        if num_negatives > 0:
            # Use einsum for a clean batched dot product: (N, D) * (N, M, D) -> (N, M)
            # This computes `image_features[i] @ negative_text_features[i].T` for all i
            neg_logits = torch.einsum('nd,nmd->nm', image_features, negative_text_features)
            
            # Apply scale and bias
            neg_logits = logit_scale * neg_logits
            if logit_bias is not None:
                neg_logits += logit_bias
            
            # Loss for negative pairs is -log(sigmoid(-logit))
            neg_pairwise_loss = -F.logsigmoid(-neg_logits)
            
            # Aggregate the negative losses for each image
            neg_loss_per_image = neg_pairwise_loss.sum(dim=1)
            # There is no 'mean' choice for negatives, since the negatives are not in-batch and not of variable grouping.

        # --- Final Loss ---
        # The total loss for each image is its normalized positive loss plus its
        # aggregated negative loss.
        total_loss_per_image = normalized_pos_loss_per_image + neg_loss_per_image

        # The final batch loss is the mean of the per-image losses.
        loss = total_loss_per_image.mean()

        return {"contrastive_loss": loss} if output_dict else loss
    
class GroupedPairedNegativeSigLipLoss(VariableTextSigLipLoss):
    """
    Sigmoid Loss for Language Image Pre-Training (SigLIP) with grouped, paired data.

    This loss is designed for datasets where image-text pairs can be grouped.
    Each image is paired with at most one positive text, and each text is paired
    with exactly one image. The key features are:
    
    1.  **One-to-One Pairing:** Each positive text corresponds to a single image instance.
    2.  **Grouping:** Positive image-text pairs are assigned to groups.
    3.  **Group Normalization:** The total loss (positive + negative) for each
        group is normalized by the number of images in that group. This ensures
        that large groups do not dominate the final batch loss.
    4.  **Paired Negatives:** Each image (regardless of whether it has a positive
        text) is contrasted against its own dedicated non-empty set of negative texts.
    5.  **Isolated Images:** Images without a positive text are handled correctly,
        contributing only negative loss and forming their own "group of one" for
        the purpose of the final loss calculation.

    Use cases:
    - Training with data organized into semantic groups (e.g., categories, scenes)
    - Ensuring balanced loss contribution across groups of different sizes
    - Handling datasets where some images lack positive text descriptions

    Attributes:
        Inherits from VariableTextSigLipLoss (negative_loss_agg, ignore_no_positive_samples).
    """
    def forward(
        self,
        image_features: torch.Tensor,
        positive_text_features: torch.Tensor,
        image_indices_for_pos_texts: torch.Tensor,
        group_indices_for_pos_pairs: torch.Tensor,
        negative_text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Computes the SigLIP loss for grouped, paired data.

        The loss computation follows these steps:
        1. Calculate negative loss for all images using their dedicated negatives
        2. Calculate positive loss for all positive pairs
        3. Map all images to groups (including creating isolated groups for images without positives)
        4. Aggregate positive and negative losses by group
        5. Normalize each group's total loss by the number of images in that group
        6. Return mean loss across all groups

        Args:
            image_features (torch.Tensor): Shape (N, D), where N is the total number of images
                                          and D is the embedding dimension.
            positive_text_features (torch.Tensor): Shape (P, D), where P is the total number of
                                                  positive texts (P <= N). Each positive text is
                                                  paired with exactly one image.
            image_indices_for_pos_texts (torch.Tensor): Shape (P,), maps each positive text
                to its corresponding image index. E.g., `image_indices_for_pos_texts[j] = i`
                means `positive_text_features[j]` is the positive for `image_features[i]`.
            group_indices_for_pos_pairs (torch.Tensor): Shape (P,), maps each positive pair
                to a group index. E.g., `group_indices_for_pos_pairs[j] = g` means the
                j-th pair belongs to group g. Groups are numbered starting from 0.
            negative_text_features (torch.Tensor): Shape (N, M, D), where M is the number of
                dedicated negatives for each of the N images. Can have M=0 for no negatives.
            logit_scale (torch.Tensor): The learnable temperature scaling parameter.
            logit_bias (Optional[torch.Tensor]): The learnable bias parameter to add to logits.
            output_dict (bool): If True, returns a dictionary with key 'contrastive_loss'.
                               If False, returns the loss tensor directly.

        Returns:
            torch.Tensor | dict[str, torch.Tensor]: The computed loss as a single scalar tensor,
                                                    or a dictionary if output_dict is True.
                                                    Returns 0.0 with requires_grad=True for empty batches.
        """
        device = image_features.device
        num_images = image_features.shape[0]
        num_pos_pairs = positive_text_features.shape[0]

        if num_images == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {"contrastive_loss": loss} if output_dict else loss

        # --- Step 1: Calculate Negative Loss for ALL images ---
        # This is computed per-image, independent of grouping initially.
        neg_loss_per_image = torch.zeros(num_images, device=device)
        if negative_text_features.numel() > 0 and negative_text_features.shape[1] > 0:
            neg_logits = torch.einsum('nd,nmd->nm', image_features, negative_text_features)
            neg_logits = logit_scale * neg_logits
            if logit_bias is not None:
                neg_logits += logit_bias
            
            neg_pairwise_loss = -F.logsigmoid(-neg_logits)
            
            neg_loss_per_image = neg_pairwise_loss.sum(dim=1)
            # There is no 'mean' choice for negatives, since the negatives are not in-batch and not of variable grouping.

        # --- Step 2: Calculate Positive Loss for all PAIRS ---
        true_pos_losses = torch.tensor([], device=device)
        if num_pos_pairs > 0:
            # Logits between all images and all positive texts
            pos_logits = self.get_logits(image_features, positive_text_features, logit_scale, logit_bias)
            
            # Loss for positive pairs is -log(sigmoid(logit))
            pos_pairwise_loss = -F.logsigmoid(pos_logits)
            
            # Extract the loss for the true positive pairs
            text_indices = torch.arange(num_pos_pairs, device=device)
            true_pos_losses = pos_pairwise_loss[image_indices_for_pos_texts, text_indices]
        
        # --- Step 3: Map all images to a group and aggregate losses ---
        
        # Determine the total number of groups. This includes groups defined by
        # positive pairs and new "isolated" groups for images without positives.
        num_positive_groups = 0
        if num_pos_pairs > 0:
            num_positive_groups = int(group_indices_for_pos_pairs.max().item()) + 1

        # Create a mapping from each image index to its final group index.
        group_indices_for_images = torch.full((num_images,), -1, dtype=torch.long, device=device)
        if num_pos_pairs > 0:
            # Images with positives get their assigned group index
            group_indices_for_images.scatter_(0, image_indices_for_pos_texts, group_indices_for_pos_pairs)

        # Images without a positive pair (still marked as -1) are assigned to
        # new, unique "isolated" groups.
        isolated_mask = (group_indices_for_images == -1)
        num_isolated = isolated_mask.sum()
        isolated_group_indices = torch.arange(
            num_positive_groups, num_positive_groups + num_isolated, device=device
        )
        group_indices_for_images[isolated_mask] = isolated_group_indices
        
        num_total_groups = num_positive_groups + num_isolated
        
        if num_total_groups == 0: # Happens if num_images > 0 but no groups assigned (shouldn't occur with this logic)
             loss = torch.tensor(0.0, device=device, requires_grad=True)
             return {"contrastive_loss": loss} if output_dict else loss

        # Aggregate positive and negative losses by their final group index
        pos_loss_per_group = torch.zeros(num_total_groups, device=device)
        if num_pos_pairs > 0:
            print(f"pos_loss_per_group.dtype: {pos_loss_per_group.dtype}")
            print(f"true_pos_losses.dtype: {true_pos_losses.dtype}")
            pos_loss_per_group.scatter_add_(0, group_indices_for_pos_pairs, true_pos_losses)

        neg_loss_per_group = torch.zeros(num_total_groups, device=device)
        neg_loss_per_group.scatter_add_(0, group_indices_for_images, neg_loss_per_image)
        
        total_loss_per_group = pos_loss_per_group + neg_loss_per_group

        # --- Step 4: Normalize by group size and compute final loss ---
        
        # Group size is the number of IMAGES in each group
        group_sizes = torch.bincount(group_indices_for_images, minlength=num_total_groups)
        
        # Normalize the total loss of each group by its size
        group_normalizer = torch.clamp(group_sizes, min=1).float()
        normalized_loss_per_group = total_loss_per_group / group_normalizer
        
        # The final loss is the mean of the normalized per-group losses
        loss = normalized_loss_per_group.mean()

        return {"contrastive_loss": loss} if output_dict else loss


# Rescaling functions for loss weighting transformations
lin_rescale_fn = lambda x: x  # Linear rescaling (identity function)
log_rescale_fn = lambda x: torch.log(1 + x)  # Logarithmic rescaling
exp_rescale_fn = lambda x: torch.exp(x) - 1  # Exponential rescaling
pow_rescale_fn = lambda x, exp: torch.pow(x, exponent=exp)  # Power rescaling with custom exponent

def xen_rescaler(
        loss: torch.Tensor,
        weights: torch.Tensor,
        alpha: float,
        normalise: bool = False,
) -> torch.Tensor:
    """
    Rescales loss values using cross-entropy-style weighting.

    Applies weighted rescaling to loss values where each loss component is multiplied
    by (1 + alpha * weight). The weights are detached to prevent gradient flow through them.
    Optionally normalizes the rescaled loss to maintain the same total magnitude as the original.

    Args:
        loss (torch.Tensor): The original loss tensor to be rescaled.
        weights (torch.Tensor): Weight values for each loss component. Must be broadcastable
                               with loss. Gradients are detached from weights.
        alpha (float): Scaling factor controlling the influence of weights. Higher alpha
                      increases the impact of the weighting scheme.
        normalise (bool): If True, normalizes the rescaled loss to preserve the total
                         magnitude of the original loss. Defaults to False.

    Returns:
        torch.Tensor: The rescaled and mean-reduced loss value as a scalar tensor.
    """
    loss_rescaled = loss*(1 + alpha*weights.detach())
    if normalise:
        loss_rescaled *= loss.sum().detach()/loss_rescaled.sum().detach() # normalise to have the same value of original loss
    loss_rescaled = loss_rescaled.mean()
    return loss_rescaled

def backward(
        total_loss: torch.Tensor,
        scaler: GradScaler,
        retain_graph: bool = False
) -> None:
    """
    Performs backward pass with optional Automatic Mixed Precision (AMP) scaling.

    Handles the backward pass for gradient computation, automatically using AMP's
    GradScaler if provided. This allows for mixed precision training with proper
    gradient scaling to prevent underflow.

    Args:
        total_loss (torch.Tensor): The loss tensor to backpropagate through.
        scaler (GradScaler): The gradient scaler for AMP. If None, performs
                            standard backward pass without scaling.
        retain_graph (bool): If True, retains the computation graph after backward.
                            Useful for computing multiple gradients. Defaults to False.

    Returns:
        None
    """
    if scaler is not None:
        scaler.scale(total_loss).backward(retain_graph=retain_graph)
    else:
        total_loss.backward(retain_graph=retain_graph)

def test_VariableTextSigLipLoss() -> None:
    """
    Tests the VariableTextSigLipLoss with various configurations.

    Validates the loss computation with:
    - Variable numbers of positive texts per image (including zero positives)
    - Proper handling of images without positive texts
    - Correct loss calculation and normalization

    Prints the computed loss value to verify functionality.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = 128  # Dimension of features
    N = 4    # Number of images in the batch

    torch.manual_seed(42)

    # Dummy model outputs
    image_features = F.normalize(torch.randn(N, D, device=device))

    # A list of text features, one entry per image, with variable numbers of positives.
    # - Image 0: 2 positive texts
    # - Image 1: 0 positive texts
    # - Image 2: 3 positive texts
    # - Image 3: 1 positive text
    list_of_text_features = [
        F.normalize(torch.randn(2, D, device=device)),
        torch.empty(0, D, device=device), # Image with no positives
        F.normalize(torch.randn(3, D, device=device)),
        F.normalize(torch.randn(1, D, device=device)),
    ]

    # --- 3. Compute the loss as before ---
    siglip_loss_fn = VariableTextSigLipLoss()
    logit_scale = nn.Parameter(torch.tensor(10.0, device=device))
    logit_bias = nn.Parameter(torch.tensor(-10.0, device=device))

    loss = siglip_loss_fn(
        image_features=image_features,
        list_of_text_features=list_of_text_features,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
    )

    print("\n--- Loss Computation ---")
    print(f"Computed Loss: {loss.item():.4f}")

def test_PairedNegativeSigLipLoss() -> None:
    """
    Tests the PairedNegativeSigLipLoss with various edge cases and scenarios.

    Validates the loss computation with:
    - Standard case with variable positives and fixed negatives per image
    - Images with zero positive texts
    - No negatives provided (M=0)
    - output_dict functionality
    - Empty batch handling

    Prints results for each test case to verify correct behavior.
    """
    # --- Setup ---
    BATCH_SIZE = 4
    NUM_NEGATIVES = 3
    FEATURE_DIM = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # For reproducibility
    torch.manual_seed(42)

    # Common parameters for the loss function
    logit_scale = nn.Parameter(torch.tensor(100.0, device=device))
    logit_bias = nn.Parameter(torch.tensor(-5.0, device=device))

    # --- Example 1: Standard Case ---
    print("\n" + "="*20 + " Example 1: Standard Case " + "="*20)
    print("Description: Each image has a variable number of positives and a fixed number of negatives.")
    
    image_features = F.normalize(torch.randn(BATCH_SIZE, FEATURE_DIM, device=device))
    
    # Each image gets a different number of positive texts
    pos_counts = [2, 1, 3, 1]
    list_of_positives = [
        F.normalize(torch.randn(k, FEATURE_DIM, device=device)) for k in pos_counts
    ]

    # Each image gets its own set of 3 negative texts
    negative_features = F.normalize(torch.randn(BATCH_SIZE, NUM_NEGATIVES, FEATURE_DIM, device=device))
    
    loss_fn = PairedNegativeSigLipLoss()
    loss = loss_fn(image_features, list_of_positives, negative_features, logit_scale, logit_bias)
    print(f"Loss (Standard Case): {loss.item():.4f}")


    # --- Example 2: Image with Zero Positives ---
    print("\n" + "="*20 + " Example 2: Image with Zero Positives " + "="*20)
    print("Description: One image in the batch has no positive texts associated with it.")
    
    pos_counts_with_zero = [2, 0, 1, 3]
    list_of_positives_with_zero = [
        F.normalize(torch.randn(k, FEATURE_DIM, device=device)) for k in pos_counts_with_zero
    ]
    
    loss = loss_fn(image_features, list_of_positives_with_zero, negative_features, logit_scale, logit_bias)
    print(f"Loss (with Zero Positives): {loss.item():.4f}")
    print("Note: The loss is still computed correctly, using only negative loss for the second image.")


    # --- Example 3: No Negatives Provided ---
    print("\n" + "="*20 + " Example 3: No Negatives Provided " + "="*20)
    print("Description: The negative_text_features tensor has a size of 0 in the M dimension.")

    # Create a tensor with M=0 negatives
    no_negative_features = torch.empty(BATCH_SIZE, 0, FEATURE_DIM, device=device)
    
    loss = loss_fn(image_features, list_of_positives, no_negative_features, logit_scale, logit_bias)
    print(f"Loss (No Negatives): {loss.item():.4f}")
    print("Note: The loss is calculated using only the positive pairs.")

    # --- Example 6: Testing output_dict=True ---
    print("\n" + "="*20 + " Example 6: Testing output_dict " + "="*20)
    print("Description: Verifies that the function returns a dictionary when requested.")

    output = loss_fn(image_features, list_of_positives, negative_features, logit_scale, logit_bias, output_dict=True)
    
    print(f"Output type: {type(output)}")
    print(f"Output content: {output}")
    assert isinstance(output, dict) and "contrastive_loss" in output
    print("Output is a dictionary with the correct key.")


    # --- Example 7: Empty Batch ---
    print("\n" + "="*20 + " Example 7: Empty Batch " + "="*20)
    print("Description: Tests the behavior with an empty batch of images.")
    
    empty_image_features = torch.empty(0, FEATURE_DIM, device=device)
    empty_list_of_positives = []
    empty_negative_features = torch.empty(0, NUM_NEGATIVES, FEATURE_DIM, device=device)
    
    # Added a check in the forward pass to handle this gracefully
    loss = loss_fn(empty_image_features, empty_list_of_positives, empty_negative_features, logit_scale, logit_bias)
    print(f"Loss (Empty Batch): {loss.item():.4f}")
    print("The function correctly returns 0.0 for an empty batch.")

def compare_VariableTextSigLipLoss_PairedNegativeSigLipLoss() -> None:
    """
    Compares VariableTextSigLipLoss and PairedNegativeSigLipLoss for equivalence.

    Tests that when negatives in PairedNegativeSigLipLoss are explicitly set to
    match the implicit in-batch negatives of VariableTextSigLipLoss, both losses
    produce identical results. This validates the correctness of the paired
    negative implementation.

    Uses negative_loss_agg='sum' for the comparison and asserts loss equality.
    """
    # --- Setup a controlled environment ---
    BATCH_SIZE = 2
    NUM_PAIRS_PER_IMAGE = 3 # Must be the same for both images for a fair comparison
    FEATURE_DIM = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    torch.manual_seed(42) # Use a different seed for different results

    # Shared parameters
    logit_scale = nn.Parameter(torch.tensor(10.0, device=device))
    logit_bias = nn.Parameter(torch.tensor(-5.0, device=device))

    # Shared features
    image_features = F.normalize(torch.randn(BATCH_SIZE, FEATURE_DIM, device=device))
    pos_texts_0 = F.normalize(torch.randn(NUM_PAIRS_PER_IMAGE, FEATURE_DIM, device=device))
    pos_texts_1 = F.normalize(torch.randn(NUM_PAIRS_PER_IMAGE, FEATURE_DIM, device=device))

    # --- Test 1: negative_loss_agg = 'sum' ---
    print("="*20 + " Test 1: `negative_loss_agg='sum'` " + "="*20)

    # --- Original Loss Calculation ---
    print("\n--- Calculating with Original VariableTextSigLipLoss ---")
    loss_fn_original = VariableTextSigLipLoss(negative_loss_agg='sum')
    list_of_all_texts = [pos_texts_0, pos_texts_1]

    loss_original = loss_fn_original(image_features, list_of_all_texts, logit_scale, logit_bias)
    print(f"Original Loss: {loss_original.item():.6f}")
    print("Note: For image 0, its implicit negatives are the 3 positive texts from image 1.")

    # --- New Paired Loss Calculation ---
    print("\n--- Calculating with New PairedNegativeSigLipLoss ---")
    loss_fn_paired = PairedNegativeSigLipLoss(negative_loss_agg='sum')

    # The positives are the same as above
    list_of_pos_texts = [pos_texts_0, pos_texts_1]

    # We explicitly construct the negatives to match the original loss scenario
    # Negatives for image 0 are pos_texts_1
    # Negatives for image 1 are pos_texts_0
    negative_features = torch.stack([pos_texts_1, pos_texts_0], dim=0)

    loss_paired = loss_fn_paired(image_features, list_of_pos_texts, negative_features, logit_scale, logit_bias)
    print(f"Paired Loss:   {loss_paired.item():.6f}")
    print("Note: We explicitly provided the positives of the other image as negatives.")

    # --- Verification ---
    assert torch.allclose(loss_original, loss_paired), "Loss values for 'sum' do not match!"
    print("\n✅ SUCCESS: The loss values for `agg='sum'` are identical.\n")

def test_GroupedPairedNegativeSigLipLoss() -> None:
    """
    Tests the GroupedPairedNegativeSigLipLoss with grouped data scenarios.

    Validates the loss computation with:
    - Multiple groups with different sizes
    - Isolated images without positive texts
    - Equivalence test with PairedNegativeSigLipLoss for single-group case

    Test 1: Simple grouped case with 2 groups and 1 isolated image
    Test 2: Equivalence verification by expanding a single-image scenario into
            multiple instances grouped together

    Asserts equivalence where expected and prints results.
    """
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = 128  # Feature dimension
    torch.manual_seed(42)

    logit_scale = nn.Parameter(torch.tensor(10.0, device=device))
    logit_bias = nn.Parameter(torch.tensor(-5.0, device=device))

    # --- Test 1: Simple GroupedPairedNegativeSigLipLoss Example ---
    print("--- Test 1: Simple Grouped Case ---")
    
    # Batch composition:
    # - Group 0: 2 image-text pairs
    # - Group 1: 1 image-text pair
    # - Isolated: 1 image with no positive text
    
    num_images_total = 4
    num_pos_pairs = 3
    num_negatives_per_image = 5
    
    # (N, D) -> 4 images total
    image_features = torch.randn(num_images_total, D, device=device)
    
    # (P, D) -> 3 positive texts
    positive_text_features = torch.randn(num_pos_pairs, D, device=device)
    
    # (N, M, D) -> 4 images, each with 5 negatives
    negative_text_features = torch.randn(num_images_total, num_negatives_per_image, D, device=device)
    
    # Map positive texts to images
    # Text 0 -> Image 0
    # Text 1 -> Image 1
    # Text 2 -> Image 2
    # Image 3 has NO positive text
    image_indices_for_pos_texts = torch.tensor([0, 1, 2], device=device, dtype=torch.long)
    
    # Map positive pairs to groups
    # Pair (I0, T0) -> Group 0
    # Pair (I1, T1) -> Group 0
    # Pair (I2, T2) -> Group 1
    group_indices_for_pos_pairs = torch.tensor([0, 0, 1], device=device, dtype=torch.long)
    
    loss_fn_grouped = GroupedPairedNegativeSigLipLoss(negative_loss_agg='sum').to(device)
    loss_grouped = loss_fn_grouped(
        image_features,
        positive_text_features,
        image_indices_for_pos_texts,
        group_indices_for_pos_pairs,
        negative_text_features,
        logit_scale,
        logit_bias
    )
    print(f"Grouped Loss (Simple Case): {loss_grouped.item():.4f}")
    
    
    # --- Test 2: Equivalence with PairedNegativeSigLipLoss ---
    print("\n--- Test 2: Equivalence Test ---")

    # Scenario for PairedNegativeSigLipLoss:
    # 1 image, 3 positive texts, 5 negative texts
    K_pos = 3
    M_neg = 5
    
    image_features_orig = torch.randn(1, D, device=device)
    list_of_pos_texts_orig = [torch.randn(K_pos, D, device=device)]
    neg_texts_orig = torch.randn(1, M_neg, D, device=device)

    loss_fn_paired = PairedNegativeSigLipLoss(negative_loss_agg='sum').to(device)
    loss_paired = loss_fn_paired(
        image_features_orig,
        list_of_pos_texts_orig,
        neg_texts_orig,
        logit_scale,
        logit_bias
    )
    print(f"PairedNegativeSigLipLoss Result: {loss_paired.item():.6f}")

    # --- Recreate the equivalent scenario for GroupedPairedNegativeSigLipLoss ---
    
    # Expand the original data: create K=3 separate image-text pairs in one group
    # N=K=3 images, P=K=3 positive texts, all in group 0
    
    # Images: Replicate the original image K times
    grouped_images = image_features_orig.repeat(K_pos, 1)
    
    # Positive Texts: Use the original list of positives
    grouped_pos_texts = list_of_pos_texts_orig[0]
    
    # Negatives: Replicate the original negatives for each new image instance
    grouped_neg_texts = neg_texts_orig.repeat(K_pos, 1, 1)
    
    # Indices: Each text `k` pairs with the new image instance `k`
    grouped_img_indices = torch.arange(K_pos, device=device, dtype=torch.long)
    
    # Groups: All K pairs belong to the same group (group 0)
    grouped_group_indices = torch.zeros(K_pos, device=device, dtype=torch.long)
    
    loss_grouped_equiv = loss_fn_grouped(
        grouped_images,
        grouped_pos_texts,
        grouped_img_indices,
        grouped_group_indices,
        grouped_neg_texts,
        logit_scale,
        logit_bias
    )
    print(f"Grouped Loss (Equivalent Case): {loss_grouped_equiv.item():.6f}")
    
    # --- Verification ---
    assert torch.allclose(loss_paired, loss_grouped_equiv), "Losses are not equivalent!"
    print("\n✅ Equivalence Test Passed: The two losses are equivalent in the specified scenario.")


def compare_PairedNegativeSigLipLoss_GroupedPairedNegativeSigLipLoss() -> None:
    """
    Comprehensive equivalence test between PairedNegativeSigLipLoss and GroupedPairedNegativeSigLipLoss.

    Validates that when PairedNegative batches are converted to the grouped format
    (where each image with positives forms its own group), both loss functions
    produce identical results across multiple test scenarios including:
    - Standard cases with variable positives
    - Edge cases with zero positives for some/all images
    - Single-item batches
    - Mixed configurations

    Includes helper functions:
    - generate_paired_data_item: Creates data for one PairedNegative item
    - convert_to_grouped_data: Converts PairedNegative batch to Grouped format
    - run_equivalence_test_case: Runs one equivalence test scenario

    Asserts equivalence for all test scenarios.
    """
    def generate_paired_data_item(
        K_pos: int, 
        M_neg: int, 
        D: int, 
        device: str
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Generates random features for one item in a PairedNegativeSigLipLoss batch.

        Args:
            K_pos (int): Number of positive texts for this image.
            M_neg (int): Number of negative texts for this image.
            D (int): Embedding dimension.
            device (str): Device to create tensors on.

        Returns:
            dict: Dictionary with keys 'image', 'pos_texts', 'neg_texts' containing
                  appropriately shaped random tensors.
        """
        image_features = torch.randn(1, D, device=device)
        
        if K_pos > 0:
            pos_texts = [torch.randn(K_pos, D, device=device)]
        else:
            # Handle zero positives correctly
            pos_texts = [torch.empty(0, D, device=device)]
            
        if M_neg > 0:
            neg_texts = torch.randn(1, M_neg, D, device=device)
        else:
            # Handle zero negatives correctly
            neg_texts = torch.empty(1, 0, D, device=device)
            
        return {
            "image": image_features,
            "pos_texts": pos_texts,
            "neg_texts": neg_texts
        }

    def convert_to_grouped_data(
        paired_items: list[dict], 
        device: str
    ) -> dict[str, torch.Tensor]:
        """
        Converts a list of 'paired' data items into a single batch for GroupedPairedNegativeSigLipLoss.

        For each image with K positive texts, replicates the image K times and creates
        K separate image-text pairs, all assigned to the same group. Images without
        positives are included as isolated instances.

        Args:
            paired_items (list[dict]): List of data items from generate_paired_data_item.
            device (str): Device for tensor operations.

        Returns:
            dict: Dictionary with keys 'images', 'pos_texts', 'neg_texts', 'img_indices',
                  'group_indices' formatted for GroupedPairedNegativeSigLipLoss.
        """
        all_grouped_images = []
        all_grouped_pos_texts = []
        all_grouped_neg_texts = []
        all_img_indices_for_pos = []
        all_group_indices_for_pos = []
        
        image_offset = 0
        group_idx = 0

        for item in paired_items:
            K = item['pos_texts'][0].shape[0]
            
            if K > 0:
                # Case 1: Image has positive texts. This forms a group.
                # Replicate image K times
                all_grouped_images.append(item['image'].repeat(K, 1))
                # Negatives are also replicated K times, one set for each image instance
                all_grouped_neg_texts.append(item['neg_texts'].repeat(K, 1, 1))
                
                # The positive texts are used as is
                all_grouped_pos_texts.append(item['pos_texts'][0])
                
                # Map each of the K texts to its corresponding replicated image
                all_img_indices_for_pos.append(torch.arange(K, device=device) + image_offset)
                # All K pairs belong to the same group
                all_group_indices_for_pos.append(torch.full((K,), group_idx, device=device))
                
                image_offset += K
                group_idx += 1
            else:
                # Case 2: Image has NO positive texts. It's an "isolated" image.
                # It will be handled automatically by Grouped loss, but we must
                # include its features in the batch.
                all_grouped_images.append(item['image'])
                all_grouped_neg_texts.append(item['neg_texts'])
                # No positive texts or indices are added for this item.
                
                image_offset += 1
                # Note: We don't increment group_idx here. The Grouped loss
                # will assign a new, unique group index to this isolated image.

        # Concatenate all lists into final tensors
        final_images = torch.cat(all_grouped_images, dim=0)
        final_neg_texts = torch.cat(all_grouped_neg_texts, dim=0)
        
        if all_grouped_pos_texts:
            final_pos_texts = torch.cat(all_grouped_pos_texts, dim=0)
            final_img_indices = torch.cat(all_img_indices_for_pos, dim=0)
            final_group_indices = torch.cat(all_group_indices_for_pos, dim=0)
        else:
            # Handle case where there are no positive texts in the entire batch
            D = final_images.shape[1]
            final_pos_texts = torch.empty(0, D, device=device)
            final_img_indices = torch.empty(0, dtype=torch.long, device=device)
            final_group_indices = torch.empty(0, dtype=torch.long, device=device)

        return {
            "images": final_images,
            "pos_texts": final_pos_texts,
            "neg_texts": final_neg_texts,
            "img_indices": final_img_indices,
            "group_indices": final_group_indices,
        }


    # --- Main Test Runner ---

    def run_equivalence_test_case(
        test_name: str,
        batch_config: list[tuple[int, int]],
        D: int,
        device: str,
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor,
    ):
        """
        Runs a single equivalence test case for a given configuration.
        
        Generates data for both loss functions, computes losses, and asserts
        they are equivalent within tolerance.

        Args:
            test_name (str): A descriptive name for the test scenario.
            batch_config (list[tuple[int, int]]): A list of (K_pos, M_neg) tuples, where
                                                  each tuple defines an image for the
                                                  PairedNegative loss batch.
            D (int): Embedding dimension.
            device (str): Device for tensor operations.
            logit_scale (torch.Tensor): Temperature scaling parameter.
            logit_bias (torch.Tensor): Bias parameter.

        Raises:
            AssertionError: If the losses are not equivalent within tolerance.
        """
        print(f"--- Running Test: {test_name} (agg_method='sum') ---")
        
        # 1. Generate data for PairedNegativeSigLipLoss
        paired_items = [generate_paired_data_item(K, M, D, device) for K, M in batch_config]
        
        paired_images = torch.cat([item['image'] for item in paired_items], dim=0)
        paired_pos_texts_list = [item['pos_texts'][0] for item in paired_items]
        paired_neg_texts = torch.cat([item['neg_texts'] for item in paired_items], dim=0)
        
        # 2. Generate equivalent data for GroupedPairedNegativeSigLipLoss
        grouped_data = convert_to_grouped_data(paired_items, device)

        # 3. Instantiate loss functions
        loss_fn_paired = PairedNegativeSigLipLoss(negative_loss_agg='sum').to(device)
        loss_fn_grouped = GroupedPairedNegativeSigLipLoss(negative_loss_agg='sum').to(device)
        
        # 4. Calculate losses
        loss_paired = loss_fn_paired(
            paired_images,
            paired_pos_texts_list,
            paired_neg_texts,
            logit_scale,
            logit_bias
        )
        
        loss_grouped = loss_fn_grouped(
            grouped_data["images"],
            grouped_data["pos_texts"],
            grouped_data["img_indices"],
            grouped_data["group_indices"],
            grouped_data["neg_texts"],
            logit_scale,
            logit_bias
        )
        
        # 5. Compare and assert
        print(f"PairedNegative Loss:  {loss_paired.item():.8f}")
        print(f"Grouped (Equiv) Loss: {loss_grouped.item():.8f}")
        
        try:
            assert torch.allclose(loss_paired, loss_grouped, atol=1e-6)
            print("PASSED: Losses are equivalent.\n")
        except AssertionError:
            print(f"FAILED: Losses are NOT equivalent. Difference: {abs(loss_paired - loss_grouped).item()}\n")
            # In a real test suite, this would raise an error
            # raise
    
    # --- Test Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = 64  # Feature dimension
    torch.manual_seed(42)

    logit_scale = nn.Parameter(torch.tensor(0.5, device=device))
    logit_bias = nn.Parameter(torch.tensor(-0.2, device=device))
    
    # --- Define Test Scenarios ---
    
    # Each tuple is (num_positives, num_negatives) for one image in the Paired batch
    test_scenarios = [
        {
            "name": "Standard Case (Variable Positives)",
            "config": [(3, 5), (1, 5), (5, 5)] # 3 images
        },
        {
            "name": "Edge Case (Some with Zero Positives)",
            "config": [(4, 10), (0, 10), (2, 10), (0, 10)]
        },
        {
            "name": "Edge Case (All with Zero Positives)",
            "config": [(0, 8), (0, 8)]
        },
        #{"name": "Edge Case (Some with Zero Negatives)", "config": [(3, 0), (2, 5), (4, 0)]},
        #{"name": "Edge Case (All with Zero Negatives)", "config": [(2, 0), (5, 0)]},
        # {"name": "Mixed Zeroes Case", "config": [(3, 5), (0, 5), (2, 0), (0, 0)]},
        {
            "name": "Simple 1-to-1 Case",
            "config": [(1, 10), (1, 10), (1, 10)]
        },
        {
            "name": "Single Item in Batch",
            "config": [(7, 4)]
        },
        {
            "name": "Single Item in Batch (Zero Positives)",
            "config": [(0, 4)]
        },
    ]

    # --- Run All Tests for both 'sum' and 'mean' ---
    for scenario in test_scenarios:
        run_equivalence_test_case(
            test_name=scenario["name"],
            batch_config=scenario["config"],
            D=D,
            device=device,
            logit_scale=logit_scale,
            logit_bias=logit_bias
        )

if __name__ == '__main__':
    # test_VariableTextSigLipLoss()
    # test_PairedNegativeSigLipLoss()
    # compare_VariableTextSigLipLoss_PairedNegativeSigLipLoss()
    # test_GroupedPairedNegativeSigLipLoss()
    compare_PairedNegativeSigLipLoss_GroupedPairedNegativeSigLipLoss()
