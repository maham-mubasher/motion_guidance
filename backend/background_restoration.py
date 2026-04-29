from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    from simple_lama_inpainting import SimpleLama
except Exception:  # pragma: no cover - optional high-quality inpainting backend
    SimpleLama = None


_LAMA_MODEL = None


def shift_tensor_2d(tensor: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    out = torch.zeros_like(tensor)
    h, w = tensor.shape[-2:]

    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx) if dx >= 0 else w
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx) if dx <= 0 else w

    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy) if dy >= 0 else h
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy) if dy <= 0 else h

    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return out

    out[..., dst_y0:dst_y1, dst_x0:dst_x1] = tensor[..., src_y0:src_y1, src_x0:src_x1]
    return out


def inpaint_masked_region(
    image: torch.Tensor,
    hole_mask: torch.Tensor,
    iterations: int = 256,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is None:
        valid = (~hole_mask.bool()).float()
    else:
        valid = valid_mask.float() * (~hole_mask.bool()).float()
    filled = image * valid
    kernel = torch.ones((1, 1, 3, 3), device=image.device, dtype=image.dtype)

    for _ in range(iterations):
        if bool(valid.min().item()):
            break

        neighbor_sum = F.conv2d(filled, kernel.expand(image.shape[1], -1, -1, -1), padding=1, groups=image.shape[1])
        neighbor_count = F.conv2d(valid, kernel, padding=1)
        can_fill = (valid == 0) & (neighbor_count > 0)
        if not bool(can_fill.any().item()):
            break

        averaged = neighbor_sum / neighbor_count.clamp_min(1.0)
        filled = torch.where(can_fill.expand_as(filled), averaged, filled)
        valid = torch.where(can_fill, torch.ones_like(valid), valid)

    return torch.where(valid.expand_as(filled).bool(), filled, image)


def soften_mask(mask: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    if kernel_size <= 1:
        return mask.float()
    padding = kernel_size // 2
    return F.avg_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=padding).clamp(0.0, 1.0)


def erode_mask(mask: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    if kernel_size <= 1:
        return mask.float()
    padding = kernel_size // 2
    inv = 1.0 - mask.float()
    dilated_inv = F.max_pool2d(inv, kernel_size=kernel_size, stride=1, padding=padding)
    return (1.0 - dilated_inv).clamp(0.0, 1.0)


def dilate_mask(mask: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    if kernel_size <= 1:
        return mask.float()
    padding = kernel_size // 2
    return F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=padding).clamp(0.0, 1.0)


def _tensor_to_pil_rgb(image: torch.Tensor) -> Image.Image:
    image_np = image[0].detach().cpu().permute(1, 2, 0).numpy()
    image_u8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image_u8, mode="RGB")


def _mask_to_pil_l(mask: torch.Tensor) -> Image.Image:
    mask_np = mask[0, 0].detach().cpu().numpy()
    mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask_u8, mode="L")


def _pil_rgb_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image_np = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def _lama_inpaint_background(
    src_img: torch.Tensor,
    inpaint_mask: torch.Tensor,
) -> torch.Tensor | None:
    if SimpleLama is None:
        return None

    if float(inpaint_mask.max().item()) <= 0:
        return src_img

    global _LAMA_MODEL
    if _LAMA_MODEL is None:
        try:
            _LAMA_MODEL = SimpleLama()
        except Exception:
            return None

    image_pil = _tensor_to_pil_rgb(src_img)
    mask_pil = _mask_to_pil_l(inpaint_mask)
    try:
        inpainted = _LAMA_MODEL(image_pil, mask_pil)
    except Exception:
        return None

    return _pil_rgb_to_tensor(inpainted, src_img.device, src_img.dtype).clamp(0.0, 1.0)


def _opencv_inpaint_background(
    src_img: torch.Tensor,
    inpaint_mask: torch.Tensor,
    blend_mask: torch.Tensor,
    radius: int = 5,
    method: int | None = None,
) -> torch.Tensor | None:
    if cv2 is None:
        return None

    object_mask = inpaint_mask[0, 0].detach().cpu().numpy() > 0.5
    if not object_mask.any():
        return src_img

    image_np = src_img[0].detach().cpu().permute(1, 2, 0).numpy()
    image_u8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    mask_u8 = object_mask.astype(np.uint8) * 255

    inpainted = cv2.inpaint(image_u8, mask_u8, radius, method or cv2.INPAINT_TELEA)
    inpainted_t = torch.from_numpy(inpainted.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(src_img.device)

    source_alpha = soften_mask(blend_mask.float(), kernel_size=7).expand_as(src_img)
    restored = src_img * (1.0 - source_alpha) + inpainted_t * source_alpha
    return restored.clamp(0.0, 1.0)


def inpaint_background_region(
    src_img: torch.Tensor,
    object_support: torch.Tensor,
    source_hole: torch.Tensor,
    radius: int = 5,
) -> torch.Tensor:
    inpaint_mask = dilate_mask(object_support.float(), kernel_size=49)
    blend_mask = dilate_mask(source_hole.float(), kernel_size=21)
    lama_fill = _lama_inpaint_background(src_img, inpaint_mask)
    if lama_fill is not None:
        alpha = soften_mask(blend_mask, kernel_size=9).expand_as(src_img)
        return (src_img * (1.0 - alpha) + lama_fill * alpha).clamp(0.0, 1.0)

    cv2_fill = _opencv_inpaint_background(src_img, inpaint_mask, blend_mask, radius=radius)
    if cv2_fill is not None:
        return cv2_fill
    return patch_based_background_fill(src_img, object_support, source_hole, 0, 0)


def seamless_clone_object(
    base_img: torch.Tensor,
    shifted_object: torch.Tensor,
    shifted_support: torch.Tensor,
) -> torch.Tensor | None:
    if cv2 is None:
        return None

    clone_mask = erode_mask(shifted_support.float(), kernel_size=9)
    if float(clone_mask.max().item()) <= 0:
        clone_mask = shifted_support.float()

    mask_np = clone_mask[0, 0].detach().cpu().numpy()
    mask_bin = (mask_np > 0.5).astype(np.uint8)
    if mask_bin.sum() == 0:
        return base_img

    ys, xs = np.where(mask_bin > 0)
    center = (int(round(xs.mean())), int(round(ys.mean())))

    src_np = shifted_object[0].detach().cpu().permute(1, 2, 0).numpy()
    dst_np = base_img[0].detach().cpu().permute(1, 2, 0).numpy()
    src_u8 = np.clip(src_np * 255.0, 0, 255).astype(np.uint8)
    dst_u8 = np.clip(dst_np * 255.0, 0, 255).astype(np.uint8)
    mask_u8 = mask_bin * 255

    try:
        cloned = cv2.seamlessClone(src_u8, dst_u8, mask_u8, center, cv2.MIXED_CLONE)
    except Exception:
        return None

    cloned_t = (
        torch.from_numpy(cloned.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(base_img.device)
    )
    return cloned_t.clamp(0.0, 1.0)


def composite_shifted_object(
    base_img: torch.Tensor,
    shifted_object: torch.Tensor,
    shifted_support: torch.Tensor,
    edge_kernel: int = 5,
) -> torch.Tensor:
    if float(shifted_support.max().item()) <= 0:
        return base_img

    support = shifted_support.float().clamp(0.0, 1.0)
    inner = erode_mask(support, kernel_size=edge_kernel)
    edge_alpha = soften_mask(inner, kernel_size=edge_kernel) * support
    alpha = torch.maximum(inner, edge_alpha).expand_as(base_img)
    return (base_img * (1.0 - alpha) + shifted_object * alpha).clamp(0.0, 1.0)


def directional_background_fill(
    src_img: torch.Tensor,
    object_support: torch.Tensor,
    source_hole: torch.Tensor,
    shift_x: int,
    shift_y: int,
) -> torch.Tensor:
    background_only = src_img * (1.0 - object_support)
    source_hole = source_hole.bool()
    fallback = inpaint_masked_region(background_only, source_hole, valid_mask=1.0 - object_support)

    if shift_x == 0 and shift_y == 0:
        return fallback

    valid_base = 1.0 - object_support
    filled = src_img.clone()
    filled = torch.where(source_hole.expand_as(src_img), torch.zeros_like(filled), filled)
    filled_mask = (~source_hole).float()

    primary_horizontal = abs(shift_x) >= abs(shift_y)
    step_count = max(abs(shift_x), abs(shift_y), 1)
    donor_sign_x = -1 if shift_x > 0 else 1 if shift_x < 0 else 0
    donor_sign_y = -1 if shift_y > 0 else 1 if shift_y < 0 else 0
    orthogonal_offsets = [0, -1, 1, -2, 2]

    for radius in range(1, step_count + 1):
        for ortho in orthogonal_offsets:
            if primary_horizontal:
                sample_x = donor_sign_x * radius
                sample_y = ortho
            else:
                sample_x = ortho
                sample_y = donor_sign_y * radius

            shifted_bg = shift_tensor_2d(background_only, sample_x, sample_y)
            shifted_valid = shift_tensor_2d(valid_base, sample_x, sample_y).clamp(0.0, 1.0)
            can_fill = source_hole & (filled_mask == 0) & (shifted_valid > 0)
            if bool(can_fill.any().item()):
                filled = torch.where(can_fill.expand_as(filled), shifted_bg, filled)
                filled_mask = torch.where(can_fill, torch.ones_like(filled_mask), filled_mask)

        if bool(((source_hole.float() - filled_mask).clamp_min(0.0) == 0).all().item()):
            break

    directional = torch.where(filled_mask.expand_as(filled) > 0, filled, fallback)
    return torch.where(source_hole.expand_as(src_img), directional, src_img)


def _candidate_offsets(shift_x: int, shift_y: int) -> list[tuple[int, int]]:
    primary_horizontal = abs(shift_x) >= abs(shift_y)
    donor_sign_x = -1 if shift_x > 0 else 1 if shift_x < 0 else 0
    donor_sign_y = -1 if shift_y > 0 else 1 if shift_y < 0 else 0
    max_radius = max(abs(shift_x), abs(shift_y), 1)
    radii = sorted({max(1, int(round(max_radius * scale))) for scale in [0.35, 0.5, 0.75, 1.0, 1.25, 1.5]})
    orthogonal_offsets = [0, -1, 1, -2, 2, -4, 4]

    offsets: list[tuple[int, int]] = []
    for radius in radii:
        for ortho in orthogonal_offsets:
            if primary_horizontal:
                sample_x = donor_sign_x * radius
                sample_y = ortho
            else:
                sample_x = ortho
                sample_y = donor_sign_y * radius
            offsets.append((sample_x, sample_y))
    return offsets


def patch_based_background_fill(
    src_img: torch.Tensor,
    object_support: torch.Tensor,
    source_hole: torch.Tensor,
    shift_x: int,
    shift_y: int,
) -> torch.Tensor:
    source_hole = source_hole.bool()
    if shift_x == 0 and shift_y == 0:
        return inpaint_masked_region(src_img * (1.0 - object_support), source_hole, valid_mask=1.0 - object_support)

    inpaint_mask = dilate_mask(object_support.float(), kernel_size=49)
    blend_mask = dilate_mask(source_hole.float(), kernel_size=21)
    lama_fill = _lama_inpaint_background(src_img, inpaint_mask)
    if lama_fill is not None:
        alpha = soften_mask(blend_mask, kernel_size=9).expand_as(src_img)
        return (src_img * (1.0 - alpha) + lama_fill * alpha).clamp(0.0, 1.0)

    cv2_fill = _opencv_inpaint_background(src_img, inpaint_mask, blend_mask, radius=7, method=None)
    if cv2_fill is not None:
        return cv2_fill

    valid_base = 1.0 - object_support
    background_only = src_img * valid_base
    fallback = directional_background_fill(src_img, object_support, source_hole.float(), shift_x, shift_y)

    # Evaluate donor candidates on a narrow ring around the removed object.
    hole_float = source_hole.float()
    ring = (soften_mask(hole_float, kernel_size=15) > 0).float() - hole_float
    ring = ring.clamp(0.0, 1.0)

    best_loss = None
    best_fill = None
    best_valid = None
    for sample_x, sample_y in _candidate_offsets(shift_x, shift_y):
        shifted_bg = shift_tensor_2d(background_only, sample_x, sample_y)
        shifted_valid = shift_tensor_2d(valid_base, sample_x, sample_y).clamp(0.0, 1.0)
        compare_mask = ring * shifted_valid
        denom = compare_mask.sum()
        if float(denom.item()) < 32:
            continue

        coverage = (shifted_valid * hole_float).sum() / hole_float.sum().clamp_min(1.0)
        if float(coverage.item()) < 0.6:
            continue

        diff = (shifted_bg - src_img).abs() * compare_mask.expand_as(src_img)
        loss = diff.sum() / compare_mask.sum().clamp_min(1.0)
        if best_loss is None or float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best_fill = shifted_bg
            best_valid = shifted_valid

    if best_fill is None or best_valid is None:
        return fallback

    fallback_loss = (
        ((fallback - src_img).abs() * ring.expand_as(src_img)).sum()
        / ring.expand_as(src_img).sum().clamp_min(1.0)
    )

    # Match donor brightness/color statistics to the local boundary before compositing.
    donor_compare = ring * best_valid
    donor_pixels = donor_compare.expand_as(src_img)
    denom = donor_pixels.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    src_mean = (src_img * donor_pixels).sum(dim=(-2, -1), keepdim=True) / denom
    donor_mean = (best_fill * donor_pixels).sum(dim=(-2, -1), keepdim=True) / denom
    donor_var = (((best_fill - donor_mean) ** 2) * donor_pixels).sum(dim=(-2, -1), keepdim=True) / denom
    src_var = (((src_img - src_mean) ** 2) * donor_pixels).sum(dim=(-2, -1), keepdim=True) / denom
    donor_std = donor_var.sqrt().clamp_min(1e-4)
    src_std = src_var.sqrt().clamp_min(1e-4)
    matched_fill = (best_fill - donor_mean) * (src_std / donor_std) + src_mean
    matched_fill = matched_fill.clamp(0.0, 1.0)

    # Use the matched donor only in the core of the hole; keep the safer fill on the boundary.
    donor_mask = (best_valid * hole_float).clamp(0.0, 1.0)
    source_core = erode_mask(donor_mask, kernel_size=21)
    source_alpha = soften_mask(source_core, kernel_size=13)

    patched = fallback.clone()
    patched = patched * (1.0 - source_alpha.expand_as(src_img)) + matched_fill * source_alpha.expand_as(src_img)
    patched_loss = (
        ((patched - src_img).abs() * ring.expand_as(src_img)).sum()
        / ring.expand_as(src_img).sum().clamp_min(1.0)
    )

    # Only trust patch-based restoration if it really improves the local boundary match.
    if float(patched_loss.item()) >= float(fallback_loss.item()) * 0.98:
        return fallback
    return patched


def build_translate_layers(
    src_img: torch.Tensor,
    support_mask: torch.Tensor,
    dx: float,
    dy: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shift_x = int(round(dx))
    shift_y = int(round(dy))
    support = support_mask.float()
    src_object = src_img * support
    shifted_object = shift_tensor_2d(src_object, shift_x, shift_y)
    shifted_support = shift_tensor_2d(support, shift_x, shift_y).clamp(0.0, 1.0)
    source_hole = (support * (1.0 - shifted_support)).clamp(0.0, 1.0)
    background = patch_based_background_fill(src_img, support, source_hole, shift_x, shift_y)
    return background, shifted_object, shifted_support, source_hole
