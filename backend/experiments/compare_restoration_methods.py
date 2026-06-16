from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import utils

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from background_restoration import (
    _lama_inpaint_background,
    _opencv_inpaint_background,
    composite_shifted_object,
    dilate_mask,
    directional_background_fill,
    erode_mask,
    inpaint_masked_region,
    patch_based_background_fill,
    shift_tensor_2d,
    soften_mask,
)


def _load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _load_flow_support(path: Path) -> torch.Tensor:
    flow = torch.load(path, map_location="cpu", weights_only=True).float()
    if flow.ndim == 3:
        flow = flow.unsqueeze(0)
    return flow.abs().sum(1, keepdim=True).gt(0).float()


def _gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    gray = image[:, 0:1] * 0.299 + image[:, 1:2] * 0.587 + image[:, 2:3] * 0.114
    gx = gray[..., :, 1:] - gray[..., :, :-1]
    gy = gray[..., 1:, :] - gray[..., :-1, :]
    gx = torch.nn.functional.pad(gx, (0, 1, 0, 0))
    gy = torch.nn.functional.pad(gy, (0, 0, 0, 1))
    return torch.sqrt(gx * gx + gy * gy)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if not bool(mask.any().item()):
        return math.nan
    return float(values[mask.expand_as(values)].mean().item())


def _masked_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    return _masked_mean((a - b).abs(), mask)


def _color_jump(image: torch.Tensor, inner: torch.Tensor, outer: torch.Tensor) -> float:
    inner = inner.bool()
    outer = outer.bool()
    if not bool(inner.any().item()) or not bool(outer.any().item()):
        return math.nan
    inner_mean = image[inner.expand_as(image)].view(3, -1).mean(1)
    outer_mean = image[outer.expand_as(image)].view(3, -1).mean(1)
    return float((inner_mean - outer_mean).abs().mean().item())


def _build_layers(src_img: torch.Tensor, support: torch.Tensor, dx: int, dy: int):
    source_object = src_img * support
    shifted_object = shift_tensor_2d(source_object, dx, dy)
    shifted_support = shift_tensor_2d(support, dx, dy).clamp(0.0, 1.0)
    source_hole = (support * (1.0 - shifted_support)).clamp(0.0, 1.0)
    background_only = src_img * (1.0 - support)
    return background_only, shifted_object, shifted_support, source_hole


def _save_contact_sheet(paths: list[tuple[str, Path]], output_path: Path) -> None:
    cell = 220
    label_h = 28
    out = Image.new("RGB", (cell * len(paths), cell + label_h), "white")
    draw = ImageDraw.Draw(out)
    for i, (label, path) in enumerate(paths):
        image = Image.open(path).convert("RGB").resize((cell, cell))
        out.paste(image, (i * cell, label_h))
        draw.text((i * cell + 6, 7), label, fill=(0, 0, 0))
    out.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare source-hole restoration methods on one moved object.")
    parser.add_argument("--input-dir", default="./data/apple")
    parser.add_argument("--target-flow-name", default="right.pth")
    parser.add_argument("--save-dir", default="results/restoration_comparison/apple_dx150")
    parser.add_argument("--dx", type=int, default=150)
    parser.add_argument("--dy", type=int, default=0)
    parser.add_argument("--opencv-radius", type=int, default=7)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    src_img = _load_image(input_dir / "pred.png")
    support = _load_flow_support(input_dir / "flows" / args.target_flow_name)
    background_only, shifted_object, shifted_support, source_hole = _build_layers(src_img, support, args.dx, args.dy)

    inpaint_mask = dilate_mask(support.float(), kernel_size=49)
    blend_mask = dilate_mask(source_hole.float(), kernel_size=21)
    local_fill = inpaint_masked_region(background_only, source_hole.bool(), valid_mask=1.0 - support)
    directional_fill = directional_background_fill(src_img, support, source_hole, args.dx, args.dy)
    opencv_fill = _opencv_inpaint_background(src_img, inpaint_mask, blend_mask, radius=args.opencv_radius)
    lama_fill_raw = _lama_inpaint_background(src_img, inpaint_mask)
    if lama_fill_raw is not None:
        alpha = soften_mask(blend_mask, kernel_size=9).expand_as(src_img)
        lama_fill = (src_img * (1.0 - alpha) + lama_fill_raw * alpha).clamp(0.0, 1.0)
    else:
        lama_fill = None
    final_selected_fill = patch_based_background_fill(src_img, support, source_hole, args.dx, args.dy)

    methods: list[tuple[str, torch.Tensor | None]] = [
        ("local_fill", local_fill),
        ("directional_fill", directional_fill),
        ("opencv_inpaint", opencv_fill),
        ("lama_inpaint", lama_fill),
        ("final_selected", final_selected_fill),
    ]

    utils.save_image(src_img, save_dir / "source.png")
    utils.save_image(background_only, save_dir / "background_only.png")
    utils.save_image(source_hole.float(), save_dir / "source_hole_mask.png")
    utils.save_image(shifted_support.float(), save_dir / "destination_mask.png")

    rows = []
    contact_paths: list[tuple[str, Path]] = [("source", save_dir / "source.png")]
    grad_source = _gradient_magnitude(src_img)
    hole_outer_ring = dilate_mask(source_hole, kernel_size=13) - dilate_mask(source_hole, kernel_size=5)
    hole_outer_ring = hole_outer_ring.clamp(0.0, 1.0)
    destination_boundary = dilate_mask(shifted_support, kernel_size=7) - erode_mask(shifted_support, kernel_size=7)
    destination_boundary = destination_boundary.clamp(0.0, 1.0)

    for name, background in methods:
        row = {"method": name, "available": background is not None}
        if background is None:
            rows.append(row)
            continue

        composite = composite_shifted_object(background, shifted_object, shifted_support, edge_kernel=5)
        background_path = save_dir / f"{name}_background.png"
        composite_path = save_dir / f"{name}_composite.png"
        utils.save_image(background, background_path)
        utils.save_image(composite, composite_path)
        contact_paths.append((name, composite_path))

        grad_composite = _gradient_magnitude(composite)
        row.update(
            {
                "hole_l1_vs_source": _masked_l1(background, src_img, source_hole),
                "hole_gradient_mean": _masked_mean(grad_composite, source_hole),
                "source_hole_color_jump": _color_jump(composite, source_hole, hole_outer_ring),
                "object_boundary_gradient": _masked_mean(grad_composite, destination_boundary),
                "background_preservation_l1": _masked_l1(
                    composite,
                    src_img,
                    (support.bool() | shifted_support.bool()).logical_not().float(),
                ),
                "source_gradient_mean": _masked_mean(grad_source, source_hole),
            }
        )
        rows.append(row)

    csv_path = save_dir / "restoration_metrics.csv"
    json_path = save_dir / "restoration_metrics.json"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    _save_contact_sheet(contact_paths, save_dir / "restoration_contact_sheet.png")

    print(f"Wrote restoration comparison to {save_dir}")
    print(f"Wrote metrics to {csv_path}")
    if lama_fill is None:
        print("LaMa unavailable; lama_inpaint row is marked available=false.")


if __name__ == "__main__":
    main()
