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

class VariableTextSigLipLoss(nn.Module):
    """
    Sigmoid Loss for Language Image Pre-Training (SigLIP) with a variable number
    of positive texts per image and controllable negative loss aggregation.

    This implementation ensures each image in the batch contributes equally to the
    final loss, regardless of its number of positive text pairings. It also provides
    a flag to control whether the negative loss is summed or averaged.
    """
    def __init__(
            self,
            negative_loss_agg: Literal['sum', 'mean'] = 'sum'
    ) -> None:
        super().__init__()
        assert negative_loss_agg in ('sum', 'mean'), \
            "negative_loss_agg must be either 'sum' or 'mean'"
        self.negative_loss_agg = negative_loss_agg

    def get_logits(
            self, 
            image_features: torch.Tensor,
            text_features: torch.Tensor,
            logit_scale: torch.Tensor,
            logit_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the pairwise similarity logits.
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

        Args:
            image_features (torch.Tensor): A tensor of shape (N, D), where N is the number of images.
            flat_text_features (torch.Tensor): A tensor of shape (T, D), where T is the total number of
                                          texts in the batch (flattened).
            image_indices_for_texts (torch.Tensor): A 1D tensor of shape (T,) containing the index
                                                    of the image corresponding to each text feature.
            logit_scale (torch.Tensor): The learnable temperature parameter.
            logit_bias (Optional[torch.Tensor]): The learnable bias parameter.
            negative_loss_agg (Literal['sum', 'mean']): How to aggregate the negative loss per image.
                                                        'sum' (default) accumulates the push from all negatives.
                                                        'mean' averages the push from all negatives.
            output_dict (bool): If True, returns a dictionary with the loss key.

        Returns:
            The computed loss as a single scalar tensor, or a dictionary if output_dict is True.
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
            counts = torch.bincount(image_indices_for_texts, minlength=num_images)
            num_positives_per_image[:len(counts)] = counts # [:len(counts)] is not needed if the above line has minlength=num_images
            
        pos_loss_normalizer = torch.clamp(num_positives_per_image, min=1).float()
        normalized_pos_loss_per_image = pos_loss_per_image / pos_loss_normalizer

        # --- Negative Loss Calculation (sum or mean) ---
        neg_loss_per_image = (pairwise_loss * negative_mask).sum(dim=1)
        if self.negative_loss_agg == 'mean':
            # Number of negatives for each image = total texts - number of positives
            num_negatives_per_image = num_texts - num_positives_per_image
            # Clamp to avoid division by zero if an image somehow has all texts as positives
            neg_loss_normalizer = torch.clamp(num_negatives_per_image, min=1).float()
            neg_loss_per_image = neg_loss_per_image / neg_loss_normalizer
            
        # The total loss for each image is its normalized positive loss plus its
        # (potentially normalized) negative loss.
        total_loss_per_image = normalized_pos_loss_per_image + neg_loss_per_image

        # The final batch loss is the mean of the per-image losses.
        loss = total_loss_per_image.mean()

        return {"contrastive_loss": loss} if output_dict else loss

    def prepare_inputs_from_list(
            self,
            list_of_text_features: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a list of variable-sized text feature tensors into the flattened
        format required by VariableSigLipLoss.

        Args:
            list_of_text_features (List[torch.Tensor]): A list where each element is a tensor
                of shape (num_positives_for_image_i, feature_dim). The length of the list
                is the number of images in the batch. Tensors can have a size of 0 in the
                first dimension if an image has no positive texts.

        Returns:
            A tuple containing:
            - flat_text_features (torch.Tensor): A single tensor of shape (total_texts, feature_dim)
            containing all text features concatenated together.
            - image_indices_for_texts (torch.Tensor): A 1D tensor of shape (total_texts,) mapping
            each text feature in the flattened tensor back to its original image index.
        """
        # Handle the edge case of an empty list of features (e.g., an empty batch).
        if not list_of_text_features:
            # We need to return empty tensors with the correct device and dtype if possible.
            # This is a reasonable default. The feature dimension doesn't matter if there are 0 texts.
            return torch.empty((0, 0)), torch.empty(0, dtype=torch.long)

        # All tensors in the list should be on the same device.
        device = list_of_text_features[0].device

        # 1. Get the number of positive texts for each image.
        # This is a fast list comprehension.
        num_texts_per_image = torch.tensor(
            [t.shape[0] for t in list_of_text_features], 
            device=device
        )

        # 2. Create the image_indices_for_texts tensor.
        # torch.repeat_interleave is highly optimized for this exact task.
        # It will correctly handle cases where the number of repeats is 0.
        image_indices_to_repeat = torch.arange(len(list_of_text_features), device=device)
        image_indices_for_texts = torch.repeat_interleave(
            image_indices_to_repeat, 
            repeats=num_texts_per_image
        )

        # 3. Concatenate all text feature tensors into a single "flat" tensor.
        # torch.cat is the most efficient way to do this. It correctly handles
        # empty tensors within the list.
        flat_text_features = torch.cat(list_of_text_features, dim=0)

        return flat_text_features, image_indices_for_texts.long()


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
    if scaler is not None:
        scaler.scale(total_loss).backward(retain_graph=retain_graph)
    else:
        total_loss.backward(retain_graph=retain_graph)

def main() -> None:
    print("start")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = 128  # Dimension of features
    N = 4    # Number of images in the batch

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

if __name__ == '__main__':
    main()
