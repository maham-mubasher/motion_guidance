from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HyperparameterRun:
    name: str
    parameter: str
    value: str
    guidance_weight: float
    raft_iters: int
    ddim_steps: int


RUNS = [
    HyperparameterRun(
        name="guidance_weight_10",
        parameter="guidance_weight",
        value="10",
        guidance_weight=10.0,
        raft_iters=1,
        ddim_steps=80,
    ),
    HyperparameterRun(
        name="baseline_guidance30_raft1_steps80",
        parameter="baseline",
        value="guidance=30, raft_iters=1, ddim_steps=80",
        guidance_weight=30.0,
        raft_iters=1,
        ddim_steps=80,
    ),
    HyperparameterRun(
        name="guidance_weight_50",
        parameter="guidance_weight",
        value="50",
        guidance_weight=50.0,
        raft_iters=1,
        ddim_steps=80,
    ),
    HyperparameterRun(
        name="raft_iters_3",
        parameter="raft_iters",
        value="3",
        guidance_weight=30.0,
        raft_iters=3,
        ddim_steps=80,
    ),
    HyperparameterRun(
        name="raft_iters_5",
        parameter="raft_iters",
        value="5",
        guidance_weight=30.0,
        raft_iters=5,
        ddim_steps=80,
    ),
    HyperparameterRun(
        name="ddim_steps_40",
        parameter="ddim_steps",
        value="40",
        guidance_weight=30.0,
        raft_iters=1,
        ddim_steps=40,
    ),
]


def _quote(value: str | int | float | Path) -> str:
    return shlex.quote(str(value))


def build_command(run: HyperparameterRun, results_root: Path, seed: int) -> str:
    return " ".join(_quote(part) for part in build_args(run, results_root, seed))


def build_args(run: HyperparameterRun, results_root: Path, seed: int) -> list[str]:
    save_dir = results_root / f"apple_{run.name}_seed{seed}"
    parts: list[str] = [
        sys.executable,
        "-W",
        "ignore",
        "./generate.py",
        "--prompt",
        "an apple on a wooden table",
        "--input_dir",
        "./data/apple",
        "--edit_mask_path",
        "right.mask.pth",
        "--target_flow_name",
        "right.pth",
        "--use_cached_latents",
        "--save_dir",
        str(save_dir),
        "--ddim_steps",
        str(run.ddim_steps),
        "--guidance_weight",
        str(run.guidance_weight),
        "--clip_grad",
        "60",
        "--raft_iters",
        str(run.raft_iters),
        "--num_recursive_steps",
        "1",
        "--scale",
        "7.5",
        "--log_freq",
        "5",
        "--seed",
        str(seed),
        "--target_flow_mode",
        "primitive",
        "--primitive_kind",
        "translate",
        "--primitive_dx",
        "150",
        "--primitive_dy",
        "0",
        "--edit_mask_dilation",
        "6",
        "--use_hard_warp_init",
        "--use_selective_refinement",
        "--selective_inner_weight",
        "1.0",
        "--selective_outer_weight",
        "0.15",
        "--preserve_unedited_output",
        "--disable_final_composite",
    ]
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print compact one-factor-at-a-time apple hyperparameter commands."
    )
    parser.add_argument("--results-root", default="results/hyperparameter_study")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", default=None, help="Optional metadata CSV path.")
    parser.add_argument("--run", action="store_true", help="Execute the commands sequentially.")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    print("Compact apple hyperparameter study commands:\n")
    print("# Run from backend/: cd /home/maham/thesis/motion_guidance_github/motion_guidance/backend\n")
    for run in RUNS:
        print(f"# {run.name}: {run.parameter} = {run.value}")
        print(build_command(run, results_root, args.seed))
        print()

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(exist_ok=True, parents=True)
        fieldnames = list(asdict(RUNS[0]).keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for run in RUNS:
                writer.writerow(asdict(run))

    if args.run:
        for run in RUNS:
            print(f"\n=== Running {run.name} ===", flush=True)
            subprocess.run(build_args(run, results_root, args.seed), check=True)


if __name__ == "__main__":
    main()
