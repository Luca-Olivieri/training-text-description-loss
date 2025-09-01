import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

class SigLipLossSingleText(nn.Module):
    """
    Simplified Sigmoid Loss for a single text against multiple images.

    This loss function adapts the logic from SigLIP (https://arxiv.org/abs/2303.15343)
    for a specific use case: scoring a single text embedding against a batch of candidate
    image embeddings, where one image is the correct "positive" match.

    Instead of building a full NxN similarity matrix between N texts and N images,
    this function computes a single 1xM similarity vector (a "row" of the matrix).
    The loss then pushes the similarity score of the positive pair up and the scores
    of all negative pairs down.

    This is useful for retrieval-style tasks or evaluation where you want to rank
    a set of images for a given text prompt.
    """
    def __init__(self) -> None:
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
        Calculates the SigLIP loss for one text against many images.

        Args:
            text_features (torch.Tensor): A single text embedding.
                Shape: (1, embedding_dim)
            image_features (torch.Tensor): A batch of M image embeddings.
                Shape: (M, embedding_dim)
            positive_image_idx (int): The index (from 0 to M-1) in the `image_features`
                batch that corresponds to the correct, positive image for the given text.
            logit_scale (float): The learnable temperature parameter (from CLIP/SigLIP).
            logit_bias (Optional[float]): The learnable bias parameter (from SigLIP).
            output_dict (bool): If True, returns a dictionary with the key 'contrastive_loss'.
                Otherwise, returns the loss tensor directly.

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: The computed loss.
        """
        # Ensure text_features is (1, D) and image_features is (M, D)
        assert text_features.ndim == 2 and text_features.shape[0] == 1, \
            f"Expected text_features to have shape (1, D), but got {text_features.shape}"
        assert image_features.ndim == 2, \
            f"Expected image_features to have shape (M, D), but got {image_features.shape}"
        assert text_features.shape[1] == image_features.shape[1], \
            "Embedding dimensions of text and images must match."
        
        num_images = image_features.shape[0]
        assert 0 <= positive_image_idx < num_images, \
            f"positive_image_idx ({positive_image_idx}) is out of bounds for the number of images ({num_images})."


        # --- 1. Compute Logits (a single row of the similarity matrix) ---
        # text_features: (1, D)
        # image_features.T: (D, M)
        # logits: (1, M)
        logits = text_features @ image_features.T
        
        # Apply scaling and bias
        logits = logit_scale * logits
        if logit_bias is not None:
            logits += logit_bias
        
        # Squeeze to make it a 1D vector of M similarity scores
        logits = logits.squeeze()  # Shape: (M,)

        # --- 2. Create Ground Truth Labels for this row ---
        # The positive pair has a label of 1, all negative pairs have a label of -1.
        labels = -torch.ones(num_images, device=logits.device, dtype=logits.dtype)
        labels[positive_image_idx] = 1.0
        
        # --- 3. Compute the Sigmoid Loss ---
        # The loss is the negative log-sigmoid of the element-wise product of labels and logits.
        # -log(sigmoid(1 * S_positive)) - log(sigmoid(-1 * S_negative_1)) - ...
        # Summing over all M pairs gives the total loss for this single text prompt.
        loss = -F.logsigmoid(labels * logits).sum()

        if output_dict:
            return {"contrastive_loss": loss}
        return loss


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
    """
    def __init__(self) -> None:
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

lin_rescale_fn = lambda x: x
log_rescale_fn = lambda x: torch.log(1 + x)
exp_rescale_fn = lambda x: torch.exp(x) - 1
pow_rescale_fn = lambda x, exp: torch.pow(x, exponent=exp)

def xen_rescaler(
        loss: torch.Tensor,
        weights: torch.Tensor,
        alpha: float,
        normalise: bool = False,
) -> torch.Tensor:
    loss_rescaled = loss*(1 + alpha*weights)
    if normalise:
        loss_rescaled *= loss.sum().detach()/loss_rescaled.sum().detach() # normalise to have the same value of original loss
    loss_rescaled = loss_rescaled.mean()
    return loss_rescaled
