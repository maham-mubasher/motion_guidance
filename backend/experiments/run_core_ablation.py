from __future__ import annotations

import argparse
import csv
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    target_flow_mode: str
    guidance_weight: float
    use_hard_warp_init: bool
    skip_diffusion: bool
    disable_final_composite: bool
    use_selective_refinement: bool
    preserve_unedited_output: bool


CORE_VARIANTS = [
    AblationVariant(
        name="warp_inpaint_only",
        description="Deterministic primitive movement plus source-hole restoration and compositing.",
        target_flow_mode="primitive",
        guidance_weight=0.0,
        use_hard_warp_init=True,
        skip_diffusion=True,
        disable_final_composite=False,
        use_selective_refinement=False,
        preserve_unedited_output=False,
    ),
    AblationVariant(
        name="original_motion_guidance",
        description="Original file-flow Motion Guidance baseline.",
        target_flow_mode="file",
        guidance_weight=30.0,
        use_hard_warp_init=False,
        skip_diffusion=False,
        disable_final_composite=True,
        use_selective_refinement=False,
        preserve_unedited_output=False,
    ),
    AblationVariant(
        name="diffusion_no_raft",
        description="Primitive hard-warp initialization with zero RAFT guidance weight.",
        target_flow_mode="primitive",
        guidance_weight=0.0,
        use_hard_warp_init=True,
        skip_diffusion=False,
        disable_final_composite=True,
        use_selective_refinement=True,
        preserve_unedited_output=True,
    ),
    AblationVariant(
        name="diffusion_raft",
        description="Primitive hard-warp initialization with RAFT guidance, no final composite.",
        target_flow_mode="primitive",
        guidance_weight=30.0,
        use_hard_warp_init=True,
        skip_diffusion=False,
        disable_final_composite=True,
        use_selective_refinement=True,
        preserve_unedited_output=True,
    ),
    AblationVariant(
        name="full_pipeline",
        description="Primitive hard-warp initialization, RAFT guidance, restoration, and final composite.",
        target_flow_mode="primitive",
        guidance_weight=30.0,
        use_hard_warp_init=True,
        skip_diffusion=False,
        disable_final_composite=False,
        use_selective_refinement=True,
        preserve_unedited_output=True,
    ),
]


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def build_command(
    variant: AblationVariant,
    results_root: Path,
    seed: int,
    ddim_steps: int,
    raft_iters: int,
) -> str:
    save_dir = results_root / f"apple_{variant.name}_seed{seed}"
    parts = [
        "python",
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
        save_dir,
        "--ddim_steps",
        ddim_steps,
        "--guidance_weight",
        variant.guidance_weight,
        "--clip_grad",
        60,
        "--raft_iters",
        raft_iters,
        "--num_recursive_steps",
        1,
        "--scale",
        7.5,
        "--log_freq",
        5,
        "--seed",
        seed,
    ]

    if variant.target_flow_mode == "primitive":
        parts.extend(
            [
                "--target_flow_mode",
                "primitive",
                "--primitive_kind",
                "translate",
                "--primitive_dx",
                150,
                "--primitive_dy",
                0,
                "--edit_mask_dilation",
                6,
            ]
        )

    if variant.use_hard_warp_init:
        parts.append("--use_hard_warp_init")
    if variant.skip_diffusion:
        parts.append("--skip_diffusion")
    if variant.disable_final_composite:
        parts.append("--disable_final_composite")
    if variant.use_selective_refinement:
        parts.extend(
            [
                "--use_selective_refinement",
                "--selective_inner_weight",
                1.0,
                "--selective_outer_weight",
                0.15,
            ]
        )
    if variant.preserve_unedited_output:
        parts.append("--preserve_unedited_output")

    return " ".join(_quote(part) for part in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print or export core apple ablation commands.")
    parser.add_argument("--results-root", default="results/core_ablation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ddim-steps", type=int, default=40)
    parser.add_argument("--raft-iters", type=int, default=1)
    parser.add_argument("--csv", default=None, help="Optional path for exporting variant metadata.")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    print("Core apple ablation commands:\n")
    for variant in CORE_VARIANTS:
        print(f"# {variant.name}: {variant.description}")
        print(build_command(variant, results_root, args.seed, args.ddim_steps, args.raft_iters))
        print()

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(exist_ok=True, parents=True)
        fieldnames = list(asdict(CORE_VARIANTS[0]).keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for variant in CORE_VARIANTS:
                writer.writerow(asdict(variant))


if __name__ == "__main__":
    main()
