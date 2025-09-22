from typing import List, Union

import torch
from torch.nn.functional import cross_entropy, mse_loss

from llava.constants import IGNORE_INDEX

__all__ = ["soft_cross_entropy"]


def soft_cross_entropy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    soft_tokens: Union[torch.Tensor, List[int]],
    std: float = 1,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    # Remove last token from outputs and first token from targets
    outputs = outputs[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()

    # Flatten outputs and targets
    targets = targets.view(-1)
    outputs = outputs.view(targets.size(0), -1)

    # Remove outputs and targets with ignore_index
    indices = targets != ignore_index
    outputs = outputs[indices]
    targets = targets[indices]

    # Convert soft token IDs to tensor
    if isinstance(soft_tokens, list):
        soft_tokens = torch.tensor(soft_tokens).to(targets)

    # Calculate loss for non-soft tokens
    indices = torch.isin(targets, soft_tokens, invert=True)
    loss = cross_entropy(outputs[indices], targets[indices], reduction="sum")

    # Calculate loss for soft tokens
    indices = torch.isin(targets, soft_tokens)
    targets_indices = torch.zeros_like(outputs[indices])
    for k, target in enumerate(targets[indices]):
        dist = torch.exp(-((target - soft_tokens) ** 2) / (2 * std**2))
        targets_indices[k][soft_tokens] = dist / dist.sum()
    loss += cross_entropy(outputs[indices], targets_indices, reduction="sum")

    # Return average loss
    return loss / targets.size(0)


def metric_scale_factor_loss_function(
    outputs: torch.Tensor,  # shape: (num_geo, out_dim)
    targets: torch.Tensor,  # shape: (num_geo, out_dim)
) -> torch.Tensor:
    """
        Calculate the loss for the metric scale factor.
    """
    log_hat = torch.log(outputs)
    log_star = torch.log(targets.detach() + 1e-8)
    return mse_loss(log_hat, log_star, reduction="mean") / outputs.size(0)
