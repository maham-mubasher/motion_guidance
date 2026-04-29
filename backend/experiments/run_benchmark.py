from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    prompt: str
    input_dir: str
    edit_mask_path: str
    target_flow_name: str
    log_freq: int


BASELINE_CASES = [
    BenchmarkCase(
        name="teapot.down150",
        prompt="a teapot floating in water",
        input_dir="./data/teapot",
        edit_mask_path="down150.mask.pth",
        target_flow_name="down150.pth",
        log_freq=25,
    ),
    BenchmarkCase(
        name="apple.right",
        prompt="an apple on a wooden table",
        input_dir="./data/apple",
        edit_mask_path="right.mask.pth",
        target_flow_name="right.pth",
        log_freq=5,
    ),
    BenchmarkCase(
        name="topiary",
        prompt="a photo of topiary",
        input_dir="./data/topiary",
        edit_mask_path="mask.pth",
        target_flow_name="flow.pth",
        log_freq=5,
    ),
]


def build_command(case: BenchmarkCase, results_root: Path) -> str:
    save_dir = results_root / case.name
    return (
        "python ./generate.py "
        f'--prompt "{case.prompt}" '
        f"--input_dir {case.input_dir} "
        f"--edit_mask_path {case.edit_mask_path} "
        f"--target_flow_name {case.target_flow_name} "
        "--use_cached_latents "
        f"--save_dir {save_dir} "
        "--ddim_steps 40 "
        "--guidance_weight 30 "
        "--clip_grad 60 "
        "--raft_iters 1 "
        "--num_recursive_steps 1 "
        "--scale 7.5 "
        f"--log_freq {case.log_freq}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print or export baseline benchmark commands.")
    parser.add_argument(
        "--results-root",
        default="results/benchmark_baseline",
        help="Directory prefix to use for generated results commands.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path for exporting the benchmark case table.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)

    print("Baseline benchmark commands:\n")
    for case in BASELINE_CASES:
        print(f"# {case.name}")
        print(build_command(case, results_root))
        print()

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(BASELINE_CASES[0]).keys()))
            writer.writeheader()
            for case in BASELINE_CASES:
                writer.writerow(asdict(case))


if __name__ == "__main__":
    main()
