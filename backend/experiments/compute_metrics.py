from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - metrics still work with numpy fallbacks
    cv2 = None


def _resolve(path_value: str | None, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def _load_array(path: Path) -> np.ndarray:
    if not path.exists():
        return np.asarray([], dtype=np.float32)
    return np.load(path).astype(np.float32)


def _safe_last(values: np.ndarray) -> float:
    return float(values[-1]) if values.size else math.nan


def _safe_min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else math.nan


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else math.nan


def _load_flow(path: Path) -> np.ndarray:
    flow = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(flow, dict):
        raise ValueError(f"Expected tensor flow at {path}, got dict")
    flow_np = flow.detach().float().cpu().numpy()
    if flow_np.ndim == 4:
        flow_np = flow_np[0]
    if flow_np.shape[0] != 2:
        raise ValueError(f"Expected flow shape [2,H,W] at {path}, got {flow_np.shape}")
    return np.transpose(flow_np, (1, 2, 0))


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    if cv2 is not None:
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    padded = np.pad(mask.astype(bool), radius)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(radius * 2 + 1):
        for dx in range(radius * 2 + 1):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    if cv2 is not None:
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
        return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return ~_dilate(~mask.astype(bool), radius)


def _destination_support(flow: np.ndarray, support: np.ndarray) -> np.ndarray:
    h, w = support.shape
    yy, xx = np.nonzero(support)
    dst_x = np.rint(xx + flow[yy, xx, 0]).astype(np.int64)
    dst_y = np.rint(yy + flow[yy, xx, 1]).astype(np.int64)
    valid = (dst_x >= 0) & (dst_x < w) & (dst_y >= 0) & (dst_y < h)
    dest = np.zeros_like(support, dtype=bool)
    dest[dst_y[valid], dst_x[valid]] = True
    return dest


def _mean_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not mask.any():
        return math.nan
    return float(np.abs(a - b)[mask].mean())


def _gray(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = _gray(image)
    if cv2 is not None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    else:
        gy, gx = np.gradient(gray)
    return np.sqrt(gx * gx + gy * gy)


def _laplacian_variance(image: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not mask.any():
        return math.nan
    gray = _gray(image)
    if cv2 is not None:
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    else:
        lap = (
            -4.0 * gray
            + np.roll(gray, 1, axis=0)
            + np.roll(gray, -1, axis=0)
            + np.roll(gray, 1, axis=1)
            + np.roll(gray, -1, axis=1)
        )
    return float(np.var(lap[mask]))


def _color_jump(image: np.ndarray, inner_mask: np.ndarray, outer_mask: np.ndarray) -> float:
    if not inner_mask.any() or not outer_mask.any():
        return math.nan
    inner_mean = image[inner_mask].mean(axis=0)
    outer_mean = image[outer_mask].mean(axis=0)
    return float(np.abs(inner_mean - outer_mean).mean())


def _variant_name(run_dir: Path) -> str:
    name = run_dir.name
    if name.startswith("apple_"):
        name = name[len("apple_") :]
    if "_seed" in name:
        name = name.rsplit("_seed", 1)[0]
    return name


def compute_sample_metrics(run_dir: Path, sample_manifest_path: Path) -> dict[str, Any]:
    run_config = _load_json(run_dir / "config.json")
    sample_manifest = _load_json(sample_manifest_path)
    working_dir = Path(run_config.get("working_directory", run_dir))
    args = run_config.get("arguments", {})

    source_path = _resolve(run_config["inputs"]["source_image"], working_dir)
    flow_path = _resolve(run_config["planned_outputs"].get("generated_target_flow"), working_dir)
    if flow_path is None or not flow_path.exists():
        flow_path = _resolve(run_config["inputs"].get("target_flow"), working_dir)
    if source_path is None or flow_path is None:
        raise ValueError(f"Missing source or target flow in {run_dir}")

    sample_dir = sample_manifest_path.parent
    final_path = sample_dir / "pred.png"
    diffusion_path = sample_dir / "pred_diffusion.png"
    pre_composite_path = sample_dir / "pred_pre_composite.png"

    source = _load_image(source_path)
    final = _load_image(final_path)
    diffusion = _load_image(diffusion_path) if diffusion_path.exists() else final
    pre_composite = _load_image(pre_composite_path) if pre_composite_path.exists() else final
    flow = _load_flow(flow_path)

    support = np.abs(flow).sum(axis=2) > 0
    support = _resize_mask(support, final.shape[:2])
    if flow.shape[:2] != final.shape[:2]:
        scale_y = final.shape[0] / flow.shape[0]
        scale_x = final.shape[1] / flow.shape[1]
        resized_flow = np.zeros((final.shape[0], final.shape[1], 2), dtype=np.float32)
        for channel, scale in [(0, scale_x), (1, scale_y)]:
            flow_img = Image.fromarray(flow[..., channel].astype(np.float32), mode="F")
            flow_img = flow_img.resize((final.shape[1], final.shape[0]), Image.Resampling.BILINEAR)
            resized_flow[..., channel] = np.asarray(flow_img) * scale
        flow = resized_flow

    dest_support = _destination_support(flow, support)
    edit_union = _dilate(support | dest_support, radius=8)
    background_mask = ~edit_union
    source_hole = support & ~dest_support

    object_boundary = _dilate(dest_support, radius=3) & ~_erode(dest_support, radius=3)
    object_outer_ring = _dilate(dest_support, radius=6) & ~_dilate(dest_support, radius=2)
    hole_outer_ring = _dilate(source_hole, radius=6) & ~_dilate(source_hole, radius=2)

    losses = _load_array(sample_dir / "losses.npy")
    losses_flow = _load_array(sample_dir / "losses_flow.npy")
    losses_color = _load_array(sample_dir / "losses_color.npy")
    noise_norms = _load_array(sample_dir / "noise_norms.npy")
    guidance_norms = _load_array(sample_dir / "guidance_norms.npy")

    grad = _gradient_magnitude(final)
    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "sample_dir": str(sample_dir),
        "variant": _variant_name(run_dir),
        "sample_index": sample_manifest.get("sample_index"),
        "seed": sample_manifest.get("seed", args.get("seed")),
        "prompt": args.get("prompt"),
        "target_flow_mode": args.get("target_flow_mode"),
        "guidance_weight": args.get("guidance_weight"),
        "raft_iters": args.get("raft_iters"),
        "ddim_steps": args.get("ddim_steps"),
        "skip_diffusion": sample_manifest.get("skipped_diffusion", False),
        "final_output_source": sample_manifest.get("final_output_source"),
        "final_composite_applied": sample_manifest.get("final_composite_applied", False),
        "runtime_seconds": sample_manifest.get("runtime_seconds"),
        "final_total_loss_last": _safe_last(losses),
        "final_total_loss_min": _safe_min(losses),
        "final_flow_loss_last": _safe_last(losses_flow),
        "final_flow_loss_min": _safe_min(losses_flow),
        "final_color_loss_last": _safe_last(losses_color),
        "final_color_loss_min": _safe_min(losses_color),
        "guidance_norm_mean": _safe_mean(guidance_norms),
        "guidance_norm_max": _safe_max(guidance_norms),
        "noise_norm_mean": _safe_mean(noise_norms),
        "background_preservation_l1_final": _mean_l1(final, source, background_mask),
        "background_preservation_l1_pre_composite": _mean_l1(pre_composite, source, background_mask),
        "background_preservation_l1_diffusion": _mean_l1(diffusion, source, background_mask),
        "object_sharpness_laplacian_var": _laplacian_variance(final, dest_support),
        "boundary_gradient_score": float(grad[object_boundary].mean()) if object_boundary.any() else math.nan,
        "boundary_color_jump": _color_jump(final, dest_support & _dilate(dest_support, 2), object_outer_ring),
        "source_hole_gradient_score": float(grad[source_hole].mean()) if source_hole.any() else math.nan,
        "source_hole_color_jump": _color_jump(final, source_hole, hole_outer_ring),
        "support_pixels": int(support.sum()),
        "destination_support_pixels": int(dest_support.sum()),
        "source_hole_pixels": int(source_hole.sum()),
        "background_pixels": int(background_mask.sum()),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute metrics for generated ablation result folders.")
    parser.add_argument("--results-root", default="results/core_ablation")
    parser.add_argument("--csv", default=None, help="Output CSV path. Defaults to <results-root>/metrics.csv")
    parser.add_argument("--json", default=None, help="Output JSON path. Defaults to <results-root>/metrics.json")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    manifests = sorted(results_root.glob("*/sample_*/sample_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No sample manifests found under {results_root}")

    rows = [compute_sample_metrics(path.parents[1], path) for path in manifests]
    csv_path = Path(args.csv) if args.csv else results_root / "metrics.csv"
    json_path = Path(args.json) if args.json else results_root / "metrics.json"
    csv_path.parent.mkdir(exist_ok=True, parents=True)
    json_path.parent.mkdir(exist_ok=True, parents=True)

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(rows)} metric rows to {csv_path}")
    print(f"Wrote JSON metrics to {json_path}")


if __name__ == "__main__":
    main()
