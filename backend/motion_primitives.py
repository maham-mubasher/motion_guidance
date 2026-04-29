from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PrimitiveSpec:
    """Simple motion description for a masked object region."""

    kind: str
    dx: float = 0.0
    dy: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    angle_deg: float = 0.0


def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 4:
        mask = mask[:, 0]
    if mask.ndim == 3:
        return mask.bool()
    if mask.ndim == 2:
        return mask.unsqueeze(0).bool()
    raise ValueError(f"Expected 2D, 3D, or 4D mask tensor, got shape {tuple(mask.shape)}")


def resize_mask(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    mask = _normalize_mask(mask).float().unsqueeze(1)
    resized = F.interpolate(mask, size=size, mode="nearest")
    return resized[:, 0].bool()


def _mask_center(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = torch.nonzero(mask[0], as_tuple=False)
    if coords.numel() == 0:
        h, w = mask.shape[-2:]
        return (
            torch.tensor((h - 1) / 2.0, device=mask.device),
            torch.tensor((w - 1) / 2.0, device=mask.device),
        )

    y = coords[:, 0].float().mean()
    x = coords[:, 1].float().mean()
    return y, x


def translation_flow(mask: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    mask = _normalize_mask(mask)
    flow = torch.zeros((1, 2, mask.shape[-2], mask.shape[-1]), device=mask.device, dtype=torch.float32)
    flow[:, 0][mask] = dx
    flow[:, 1][mask] = dy
    return flow


def scale_flow(
    mask: torch.Tensor,
    scale_x: float,
    scale_y: Optional[float] = None,
    center: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    mask = _normalize_mask(mask)
    scale_y = scale_x if scale_y is None else scale_y
    h, w = mask.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=mask.device, dtype=torch.float32),
        torch.arange(w, device=mask.device, dtype=torch.float32),
        indexing="ij",
    )
    cy, cx = center if center is not None else _mask_center(mask)
    flow_x = (scale_x - 1.0) * (xx - cx)
    flow_y = (scale_y - 1.0) * (yy - cy)
    flow = torch.stack([flow_x, flow_y], dim=0).unsqueeze(0)
    flow *= mask.unsqueeze(1)
    return flow


def rotation_flow(
    mask: torch.Tensor,
    angle_deg: float,
    center: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    mask = _normalize_mask(mask)
    h, w = mask.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=mask.device, dtype=torch.float32),
        torch.arange(w, device=mask.device, dtype=torch.float32),
        indexing="ij",
    )
    cy, cx = center if center is not None else _mask_center(mask)
    theta = torch.tensor(angle_deg * torch.pi / 180.0, device=mask.device)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    rel_x = xx - cx
    rel_y = yy - cy
    rot_x = cos_t * rel_x - sin_t * rel_y
    rot_y = sin_t * rel_x + cos_t * rel_y
    flow_x = rot_x - rel_x
    flow_y = rot_y - rel_y
    flow = torch.stack([flow_x, flow_y], dim=0).unsqueeze(0)
    flow *= mask.unsqueeze(1)
    return flow


def stretch_flow(mask: torch.Tensor, stretch_x: float, stretch_y: float) -> torch.Tensor:
    return scale_flow(mask=mask, scale_x=stretch_x, scale_y=stretch_y)


def build_flow(mask: torch.Tensor, spec: PrimitiveSpec, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    if size is not None:
        mask = resize_mask(mask, size)
    kind = spec.kind.lower()
    if kind == "translate":
        return translation_flow(mask, dx=spec.dx, dy=spec.dy)
    if kind == "scale":
        return scale_flow(mask, scale_x=spec.scale_x, scale_y=spec.scale_y)
    if kind == "rotate":
        return rotation_flow(mask, angle_deg=spec.angle_deg)
    if kind == "stretch":
        return stretch_flow(mask, stretch_x=spec.scale_x, stretch_y=spec.scale_y)
    raise ValueError(f"Unsupported primitive kind: {spec.kind}")


def combine_flows(flows: Iterable[torch.Tensor]) -> torch.Tensor:
    flows = list(flows)
    if not flows:
        raise ValueError("Expected at least one flow tensor to combine")
    out = torch.zeros_like(flows[0])
    for flow in flows:
        out = out + flow
    return out
