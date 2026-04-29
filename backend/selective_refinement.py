from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RefinementConfig:
    """Lightweight placeholder for thesis-time selective refinement control."""

    outer_weight: float = 0.25
    inner_weight: float = 1.0
    feather_radius: int = 3


def latent_edit_region(edit_mask: torch.Tensor) -> torch.Tensor:
    """Normalize edit masks to a boolean latent-space region."""

    if edit_mask.ndim == 4:
        return edit_mask.bool()
    if edit_mask.ndim == 3:
        return edit_mask[:, None].bool()
    if edit_mask.ndim == 2:
        return edit_mask[None, None].bool()
    raise ValueError(f"Unexpected edit mask shape: {tuple(edit_mask.shape)}")


def refinement_weight_map(edit_mask: torch.Tensor, config: RefinementConfig) -> torch.Tensor:
    """
    Return a simple weighting map that keeps the edited region strong while
    reducing pressure on untouched latent regions.
    """

    region = latent_edit_region(edit_mask).float()
    return region * config.inner_weight + (1.0 - region) * config.outer_weight


def apply_selective_guidance(grad: torch.Tensor, edit_mask: torch.Tensor, config: RefinementConfig) -> torch.Tensor:
    """Scale guidance gradients so most energy stays inside the edited region."""

    weights = refinement_weight_map(edit_mask, config).to(grad.device)
    return grad * weights
