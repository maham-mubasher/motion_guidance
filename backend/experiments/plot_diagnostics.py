from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/motion_guidance_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SERIES = [
    ("losses.npy", "Total energy", "total_energy"),
    ("losses_flow.npy", "Flow loss", "flow_loss"),
    ("losses_color.npy", "Color loss", "color_loss"),
    ("noise_norms.npy", "Noise norm", "noise_norm"),
    ("guidance_norms.npy", "Guidance norm", "guidance_norm"),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_series(sample_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for filename, _, key in SERIES:
        path = sample_dir / filename
        if path.exists():
            out[key] = np.load(path).astype(np.float32)
        else:
            out[key] = np.asarray([], dtype=np.float32)
    return out


def _variant_name(run_dir: Path) -> str:
    name = run_dir.name
    if name.startswith("apple_"):
        name = name[len("apple_") :]
    if "_seed" in name:
        name = name.rsplit("_seed", 1)[0]
    return name


def _sample_label(run_dir: Path, sample_manifest: dict[str, Any]) -> str:
    variant = _variant_name(run_dir)
    seed = sample_manifest.get("seed")
    if seed is None:
        return variant
    return f"{variant} seed {seed}"


def _plot_per_sample(sample_dir: Path, plot_dir: Path, title: str, series: dict[str, np.ndarray]) -> Path | None:
    nonempty = [(label, key, values) for _, label, key in SERIES if (values := series[key]).size]
    if not nonempty:
        return None

    fig, axes = plt.subplots(len(nonempty), 1, figsize=(8, max(2.2 * len(nonempty), 4)), sharex=True)
    if len(nonempty) == 1:
        axes = [axes]

    for ax, (label, _key, values) in zip(axes, nonempty):
        ax.plot(np.arange(values.size), values, linewidth=1.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Logged guidance step")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = plot_dir / f"{sample_dir.parent.name}_{sample_dir.name}_diagnostics.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_combined(
    records: list[tuple[Path, dict[str, Any], dict[str, np.ndarray]]],
    plot_dir: Path,
) -> list[Path]:
    out_paths: list[Path] = []
    for _filename, label, key in SERIES:
        nonempty = [
            (run_dir, manifest, series[key])
            for run_dir, manifest, series in records
            if series[key].size
        ]
        if not nonempty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5.2))
        for run_dir, manifest, values in nonempty:
            ax.plot(np.arange(values.size), values, linewidth=1.8, label=_sample_label(run_dir, manifest))
        ax.set_title(label)
        ax.set_xlabel("Logged guidance step")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = plot_dir / f"combined_{key}.png"
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def _write_index(
    output_dir: Path,
    per_sample_paths: list[Path],
    combined_paths: list[Path],
    skipped: list[str],
) -> None:
    index_path = output_dir / "diagnostic_plots_index.md"
    lines = ["# Diagnostic Plots", ""]
    if combined_paths:
        lines.append("## Combined")
        lines.extend(f"- [{path.name}]({path.name})" for path in combined_paths)
        lines.append("")
    if per_sample_paths:
        lines.append("## Per Sample")
        lines.extend(f"- [{path.name}]({path.name})" for path in per_sample_paths)
        lines.append("")
    if skipped:
        lines.append("## Skipped")
        lines.extend(f"- {item}" for item in skipped)
        lines.append("")
    index_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved loss/norm diagnostics for ablation runs.")
    parser.add_argument("--results-root", default="results/core_ablation")
    parser.add_argument("--output-dir", default=None, help="Defaults to <results-root>/diagnostic_plots")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root / "diagnostic_plots"
    output_dir.mkdir(exist_ok=True, parents=True)

    manifests = sorted(results_root.glob("*/sample_*/sample_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No sample manifests found under {results_root}")

    records: list[tuple[Path, dict[str, Any], dict[str, np.ndarray]]] = []
    skipped: list[str] = []
    per_sample_paths: list[Path] = []
    for manifest_path in manifests:
        sample_dir = manifest_path.parent
        run_dir = manifest_path.parents[1]
        manifest = _load_json(manifest_path)
        series = _load_series(sample_dir)
        if not any(values.size for values in series.values()):
            skipped.append(f"{run_dir.name}/{sample_dir.name}: no diagnostic arrays")
            continue
        records.append((run_dir, manifest, series))
        out_path = _plot_per_sample(sample_dir, output_dir, _sample_label(run_dir, manifest), series)
        if out_path is not None:
            per_sample_paths.append(out_path)

    combined_paths = _plot_combined(records, output_dir)
    _write_index(output_dir, per_sample_paths, combined_paths, skipped)

    print(f"Wrote {len(per_sample_paths)} per-sample diagnostic plots to {output_dir}")
    print(f"Wrote {len(combined_paths)} combined diagnostic plots to {output_dir}")
    print(f"Wrote plot index to {output_dir / 'diagnostic_plots_index.md'}")


if __name__ == "__main__":
    main()
