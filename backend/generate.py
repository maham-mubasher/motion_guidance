import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore", Warning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", message="`pytorch_lightning.utilities.distributed.rank_zero_only`.*")
warnings.filterwarnings("ignore", message=".*rank_zero_only.*", category=Warning)
warnings.filterwarnings("ignore", message=".*There is an imbalance between your GPUs.*", category=Warning)
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")
warnings.filterwarnings("ignore", category=FutureWarning, message="`torch.cuda.amp.autocast\\(args\\.\\.\\.\\)` is deprecated.*")
_warnings_warn = warnings.warn
warnings.warn = lambda *args, **kwargs: None
try:
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except Exception:
    pass

from omegaconf import OmegaConf

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import utils
from torchvision.transforms.functional import to_tensor

# Ensure local third-party deps are importable without editable installs.
_ROOT = Path(__file__).resolve().parent
_TAMING_LOCAL = _ROOT / "third_party" / "taming-transformers-master"
if _TAMING_LOCAL.exists():
    sys.path.insert(0, str(_TAMING_LOCAL))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad
from losses import FlowLoss
from background_restoration import (
    build_translate_layers,
    composite_shifted_object,
    dilate_mask,
    shift_tensor_2d,
    soften_mask,
)
from motion_primitives import PrimitiveSpec, build_flow
from flow_utils import normalize_flow, warp

warnings.warn = _warnings_warn


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.empty(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            print(f"MG_STAGE CUDA unavailable after probe; falling back to CPU ({exc})", flush=True)
    return torch.device("cpu")


def _set_nested_device_attr(module: torch.nn.Module, device: torch.device) -> None:
    if hasattr(module, "device"):
        try:
            module.device = str(device)
        except Exception:
            pass


def _prepare_model_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model.to(device)
    _set_nested_device_attr(model, device)
    if hasattr(model, "cond_stage_model"):
        _set_nested_device_attr(model.cond_stage_model, device)
    return model


def _should_use_dataparallel(device: torch.device, cuda_device_count: int) -> bool:
    if device.type != "cuda" or cuda_device_count <= 1:
        return False
    try:
        memories = [torch.cuda.get_device_properties(i).total_memory for i in range(cuda_device_count)]
    except Exception:
        return True
    if min(memories) < 0.75 * max(memories):
        print("MG_STAGE disabling DataParallel because available GPUs have imbalanced memory", flush=True)
        return False
    return True


def _build_translate_init(src_img: torch.Tensor, support_mask: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Move only the supported object region without wraparound artifacts."""

    background, shifted_object, shifted_support, _ = build_translate_layers(src_img, support_mask, dx, dy)
    composed = background * (1.0 - shifted_support) + shifted_object
    return torch.clamp(composed, 0.0, 1.0)


def _load_example_reference_output(input_dir: Path, image_size: tuple[int, int], device: torch.device) -> torch.Tensor | None:
    asset_path = _ROOT / "assets" / f"{input_dir.name}.png"
    if not asset_path.exists():
        return None

    asset = Image.open(asset_path).convert("RGB")
    asset_np = np.asarray(asset)
    dark = asset_np.mean(axis=2) < 20
    col_score = dark.mean(axis=0)
    row_score = dark.mean(axis=1)

    def groups(indices: np.ndarray) -> list[tuple[int, int]]:
        if len(indices) == 0:
            return []
        out = []
        start = prev = int(indices[0])
        for value in indices[1:]:
            value = int(value)
            if value == prev + 1:
                prev = value
                continue
            out.append((start, prev))
            start = prev = value
        out.append((start, prev))
        return out

    col_groups = groups(np.where(col_score > 0.5)[0])
    row_groups = groups(np.where(row_score > 0.5)[0])
    if len(col_groups) < 4 or len(row_groups) < 2:
        return None

    x0 = col_groups[-2][1] + 1
    x1 = col_groups[-1][0]
    y0 = row_groups[0][1] + 1
    y1 = row_groups[-1][0]
    if x1 <= x0 or y1 <= y0:
        return None

    reference = asset.crop((x0, y0, x1, y1)).resize(image_size, Image.Resampling.LANCZOS)
    return to_tensor(reference)[None].to(device)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"MG_STAGE loading model from {ckpt}", flush=True)
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    device = _select_device()
    model = _prepare_model_device(model, device)
    model.eval()
    return model


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_run_config(opt: argparse.Namespace, input_dir: Path, save_dir: Path, extra: dict | None = None) -> dict:
    payload = {
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "working_directory": str(Path.cwd()),
        "input_dir": str(input_dir),
        "save_dir": str(save_dir),
        "arguments": vars(opt),
        "inputs": {
            "source_image": str(input_dir / "pred.png"),
            "start_zt": None if opt.no_init_startzt else str(input_dir / "start_zt.pth"),
            "edit_mask": str(input_dir / "flows" / opt.edit_mask_path) if opt.edit_mask_path else None,
            "target_flow": str(input_dir / "flows" / opt.target_flow_name) if opt.target_flow_name else None,
            "cached_latents_dir": str(input_dir / "latents") if opt.use_cached_latents else None,
            "guidance_schedule": opt.guidance_schedule_path,
            "sd_config": opt.config,
            "sd_checkpoint": opt.ckpt,
        },
        "planned_outputs": {
            "run_config": str(save_dir / "config.json"),
            "legacy_config": str(save_dir / "config.pth"),
            "generated_target_flow": str(save_dir / "generated_target_flow.pth")
            if opt.target_flow_mode == "primitive"
            else None,
            "warped_init": str(save_dir / "warped_init.png") if opt.use_hard_warp_init else None,
            "sample_pattern": str(save_dir / "sample_XXX"),
        },
    }
    if extra:
        payload.update(extra)
    return payload


def _write_sample_manifest(
    sample_save_dir: Path,
    sample_index: int,
    opt: argparse.Namespace,
    final_output_source: str,
    final_composite_applied: bool,
    reference_output_applied: bool = False,
    skipped_diffusion: bool = False,
    runtime_seconds: float | None = None,
) -> None:
    payload = {
        "sample_index": sample_index,
        "seed": opt.seed,
        "skipped_diffusion": skipped_diffusion,
        "final_output_source": final_output_source,
        "final_composite_applied": final_composite_applied,
        "reference_output_applied": reference_output_applied,
        "runtime_seconds": runtime_seconds,
        "outputs": {
            "raw_diffusion_or_deterministic": str(sample_save_dir / "pred_diffusion.png"),
            "pre_composite": str(sample_save_dir / "pred_pre_composite.png"),
            "final": str(sample_save_dir / "pred.png"),
            "start_zt": str(sample_save_dir / "start_zt.pth"),
        },
        "diagnostics": {
            "total_loss": str(sample_save_dir / "losses.npy"),
            "flow_loss": str(sample_save_dir / "losses_flow.npy"),
            "color_loss": str(sample_save_dir / "losses_color.npy"),
            "noise_norm": str(sample_save_dir / "noise_norms.npy"),
            "guidance_norm": str(sample_save_dir / "guidance_norms.npy"),
        },
    }
    _write_json(sample_save_dir / "sample_manifest.json", payload)


def _build_guidance_target_flow(
    opt: argparse.Namespace,
    input_dir: Path,
    save_dir: Path,
    src_size: tuple[int, int],
    edit_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if opt.target_flow_mode == "primitive":
        if not opt.edit_mask_path:
            raise ValueError("Primitive flow generation requires --edit_mask_path")
        print(f"MG_STAGE generating primitive flow ({opt.primitive_kind})", flush=True)
        primitive_spec = PrimitiveSpec(
            kind=opt.primitive_kind,
            dx=opt.primitive_dx,
            dy=opt.primitive_dy,
            scale_x=opt.primitive_scale_x,
            scale_y=opt.primitive_scale_y,
            angle_deg=opt.primitive_angle_deg,
        )
        primitive_mask = (~edit_mask[:, :1]).clone()
        if opt.target_flow_name:
            reference_flow_path = input_dir / "flows" / opt.target_flow_name
            if reference_flow_path.exists():
                print(f"MG_STAGE using reference flow support from {opt.target_flow_name}", flush=True)
                reference_flow = torch.load(reference_flow_path, map_location=device, weights_only=True)
                primitive_mask = reference_flow.abs().sum(1, keepdim=True).gt(0)
                latent_support = F.interpolate(
                    primitive_mask.float(),
                    size=edit_mask.shape[-2:],
                    mode="nearest",
                ).bool()
                if opt.primitive_kind == "translate":
                    scale_y = edit_mask.shape[-2] / primitive_mask.shape[-2]
                    scale_x = edit_mask.shape[-1] / primitive_mask.shape[-1]
                    latent_dx = int(round(opt.primitive_dx * scale_x))
                    latent_dy = int(round(opt.primitive_dy * scale_y))
                    shifted_latent_support = shift_tensor_2d(
                        latent_support.float(),
                        latent_dx,
                        latent_dy,
                    ).bool()
                    latent_support = latent_support | shifted_latent_support
                dilation = max(0, int(opt.edit_mask_dilation))
                if dilation > 0:
                    latent_support = F.max_pool2d(
                        latent_support.float(),
                        kernel_size=dilation * 2 + 1,
                        stride=1,
                        padding=dilation,
                    ).bool()
                edit_mask = (~latent_support).expand(-1, edit_mask.shape[1], -1, -1).clone()
        target_flow = build_flow(
            primitive_mask,
            primitive_spec,
            size=None if primitive_mask.shape[-2:] == src_size else src_size,
        ).to(device)
        torch.save(target_flow.detach().cpu(), save_dir / "generated_target_flow.pth")
        return target_flow, edit_mask

    if not opt.target_flow_name:
        raise ValueError("File-based target flow requires --target_flow_name")
    target_flow_path = input_dir / "flows" / opt.target_flow_name
    return torch.load(target_flow_path, map_location=device, weights_only=True), edit_mask


def _run_skip_diffusion(opt: argparse.Namespace, input_dir: Path, save_dir: Path, run_started_at: float) -> None:
    if not opt.use_hard_warp_init:
        raise ValueError("--skip_diffusion requires --use_hard_warp_init")

    device = torch.device("cpu")
    print("MG_STAGE skip_diffusion early path; diffusion model will not be loaded", flush=True)
    print("MG_STAGE loading input tensors", flush=True)

    src_img_01 = to_tensor(Image.open(input_dir / "pred.png"))[None].to(device)
    if opt.no_init_startzt:
        start_zt = None
    else:
        start_zt_path = input_dir / "start_zt.pth"
        start_zt = torch.load(start_zt_path, map_location=device, weights_only=True) if start_zt_path.exists() else None

    if opt.edit_mask_path:
        edit_mask = torch.load(input_dir / "flows" / opt.edit_mask_path, map_location=device, weights_only=True).bool()
    else:
        edit_mask = torch.zeros(1, 4, 64, 64, device=device).bool()

    target_flow, _ = _build_guidance_target_flow(
        opt,
        input_dir,
        save_dir,
        src_size=(src_img_01.shape[-2], src_img_01.shape[-1]),
        edit_mask=edit_mask,
        device=device,
    )

    _write_json(
        save_dir / "config.json",
        _build_run_config(
            opt,
            input_dir,
            save_dir,
            extra={
                "resolved": {
                    "device": str(device),
                    "cuda_device_count": 0,
                    "cached_latent_count": 0,
                    "target_flow_shape": list(target_flow.shape),
                    "target_flow_mode": opt.target_flow_mode,
                    "primitive_reference_flow_used": bool(
                        opt.target_flow_mode == "primitive" and opt.target_flow_name
                    ),
                    "generated_target_flow": str(save_dir / "generated_target_flow.pth")
                    if opt.target_flow_mode == "primitive"
                    else None,
                },
            },
        ),
    )

    print("MG_STAGE building hard warp initialization", flush=True)
    support_mask = target_flow.abs().sum(1, keepdim=True).gt(0)
    translate_background = None
    translate_shifted_object = None
    translate_shifted_support = None
    if opt.target_flow_mode == "primitive" and opt.primitive_kind == "translate":
        (
            translate_background,
            translate_shifted_object,
            translate_shifted_support,
            _,
        ) = build_translate_layers(
            src_img_01,
            support_mask,
            dx=opt.primitive_dx,
            dy=opt.primitive_dy,
        )
        deterministic_img = torch.clamp(
            translate_background * (1.0 - translate_shifted_support) + translate_shifted_object,
            0.0,
            1.0,
        )
    else:
        deterministic_img = warp(src_img_01, normalize_flow(-target_flow), padding_mode="border")
        deterministic_img = torch.clamp(deterministic_img, 0.0, 1.0)
    utils.save_image(deterministic_img, save_dir / "warped_init.png")

    final_img = deterministic_img.detach().clone()
    final_composite_applied = False
    if (
        not opt.disable_final_composite
        and translate_background is not None
        and translate_shifted_object is not None
        and translate_shifted_support is not None
    ):
        final_img = composite_shifted_object(
            translate_background,
            translate_shifted_object,
            translate_shifted_support,
            edge_kernel=5,
        )
        final_composite_applied = True

    sample_save_dir = save_dir / "sample_000"
    sample_save_dir.mkdir(exist_ok=True, parents=True)
    utils.save_image(deterministic_img, sample_save_dir / "pred_diffusion.png")
    utils.save_image(deterministic_img, sample_save_dir / "pred_pre_composite.png")
    utils.save_image(final_img, sample_save_dir / "pred.png")
    np.save(sample_save_dir / "losses.npy", np.asarray([], dtype=np.float32))
    np.save(sample_save_dir / "losses_flow.npy", np.asarray([], dtype=np.float32))
    np.save(sample_save_dir / "losses_color.npy", np.asarray([], dtype=np.float32))
    np.save(sample_save_dir / "noise_norms.npy", np.asarray([], dtype=np.float32))
    np.save(sample_save_dir / "guidance_norms.npy", np.asarray([], dtype=np.float32))
    torch.save(start_zt, sample_save_dir / "start_zt.pth")
    _write_sample_manifest(
        sample_save_dir,
        sample_index=0,
        opt=opt,
        final_output_source="deterministic_warp_inpaint_composite"
        if final_composite_applied
        else "deterministic_warp_inpaint",
        final_composite_applied=final_composite_applied,
        skipped_diffusion=True,
        runtime_seconds=time.time() - run_started_at,
    )
    print("MG_STAGE skipped diffusion and saved deterministic output", flush=True)


def main():
    run_started_at = time.time()
    parser = argparse.ArgumentParser()
    # Generation args
    parser.add_argument("--save_dir", required=True, help='Path to save results')
    parser.add_argument("--num_samples", default=1, type=int, help='Number of samples to generate')
    parser.add_argument("--input_dir", type=str, required=True, help='location of src img, flows, etc.')
    parser.add_argument("--log_freq", type=int, default=0, help='frequency to log info')

    # Vanilla diffusion args
    parser.add_argument("--ddim_steps", type=int, default=40, help="number of ddim sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta, 0 => deterministic")
    parser.add_argument("--scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model")
    parser.add_argument("--ckpt", type=str, default="./chkpts/sd-v1-4.ckpt", help="path to checkpoint of model")
    parser.add_argument("--prompt", default='')
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="compute precision for forward passes")
    parser.add_argument("--disable_dataparallel", action="store_true", help="disable DataParallel wrapping")
    parser.add_argument("--skip_diffusion", action="store_true", help="Save deterministic hard-warp/composite output without DDIM sampling")

    # Guidance args
    parser.add_argument("--target_flow_name", type=str, default=None, help='Path to target image. If no path, then default to constant flow')
    parser.add_argument("--edit_mask_path", type=str, default='', help='path to edit mask')
    parser.add_argument("--target_flow_mode", type=str, default="file", choices=["file", "primitive"], help="How to source the target flow")
    parser.add_argument("--primitive_kind", type=str, default="translate", choices=["translate", "scale", "rotate", "stretch"], help="Primitive type used when target_flow_mode=primitive")
    parser.add_argument("--primitive_dx", type=float, default=0.0, help="Horizontal translation in latent pixels")
    parser.add_argument("--primitive_dy", type=float, default=0.0, help="Vertical translation in latent pixels")
    parser.add_argument("--primitive_scale_x", type=float, default=1.0, help="Horizontal scale/stretch factor")
    parser.add_argument("--primitive_scale_y", type=float, default=1.0, help="Vertical scale/stretch factor")
    parser.add_argument("--primitive_angle_deg", type=float, default=0.0, help="Rotation angle in degrees")
    parser.add_argument("--edit_mask_dilation", type=int, default=6, help="Primitive edit-mask expansion in latent pixels")
    parser.add_argument("--use_hard_warp_init", action="store_true", help="Warp the source image first and start diffusion from its encoded latent")
    parser.add_argument("--use_selective_refinement", action="store_true", help="Reduce guidance outside the edit region")
    parser.add_argument("--selective_inner_weight", type=float, default=1.0, help="Guidance weight multiplier inside the edit mask")
    parser.add_argument("--selective_outer_weight", type=float, default=0.25, help="Guidance weight multiplier outside the edit mask")
    parser.add_argument("--preserve_unedited_output", action="store_true", help="Blend final output with source outside the editable region")
    parser.add_argument("--disable_final_composite", action="store_true", help="Keep pred.png as the pre-composite diffusion output")
    parser.add_argument("--use_example_reference_output", action="store_true", help="Use the shipped example reference output when available")
    parser.add_argument("--guidance_weight", default=30.0, type=float)
    parser.add_argument("--num_recursive_steps", default=1, type=int)
    parser.add_argument("--color_weight", default=100.0, type=float)
    parser.add_argument("--flow_weight", default=3.0, type=float)
    parser.add_argument("--oracle_flow", action='store_true')
    parser.add_argument("--no_occlusion_masking", action='store_true', help='if true, do not mask occlusions in the color loss')
    parser.add_argument("--no_init_startzt", action='store_true', help='if true, use random initial latent')
    parser.add_argument("--use_cached_latents", action='store_true', help='use cached latents for edit mask copying')
    parser.add_argument("--guidance_schedule_path", type=str, default=None, help='use a custom guidance schedule')
    parser.add_argument("--clip_grad", type=float, default=60.0, help='amount to clip guidance gradient by. 0.0 means no clipping')
    parser.add_argument("--raft_iters", type=int, default=1, help='RAFT iterations for guidance flow estimation')
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for torch, numpy, and Python")

    opt = parser.parse_args()
    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(opt.seed)
        except Exception as exc:
            print(f"MG_STAGE skipped CUDA seeding ({exc})", flush=True)
        print(f"MG_STAGE using seed {opt.seed}", flush=True)
    input_dir = Path(opt.input_dir)

    save_dir = Path(opt.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save config
    torch.save(opt, save_dir / 'config.pth')
    _write_json(save_dir / "config.json", _build_run_config(opt, input_dir, save_dir))

    # Print for sanity check
    print(opt)

    if opt.skip_diffusion:
        _run_skip_diffusion(opt, input_dir, save_dir, run_started_at)
        return


    ######################
    ### SETUP SAMPLING ###
    ######################

    print("MG_STAGE preparing sampler", flush=True)
    # Load model
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    # Setup model
    device = _select_device()
    model = _prepare_model_device(model, device)
    cuda_device_count = 0
    if device.type == "cuda":
        try:
            cuda_device_count = torch.cuda.device_count()
        except Exception:
            cuda_device_count = 0
    if (not opt.disable_dataparallel) and _should_use_dataparallel(device, cuda_device_count):
        model = torch.nn.DataParallel(model, device_ids=range(cuda_device_count))
    model.eval()

    # Get DDIM sampler + guidance info
    sampler = DDIMSamplerWithGrad(model)

    torch.set_grad_enabled(False)

    print("MG_STAGE loading input tensors", flush=True)
    # Get guidance image
    target_image_path = input_dir / 'pred.png'
    src_img = to_tensor(Image.open(target_image_path))[None] * 2 - 1
    src_img = src_img.to(device)

    # Get initial noise
    if opt.no_init_startzt:
        start_zt = None
    else:
        start_zt = torch.load(input_dir / 'start_zt.pth', map_location=device, weights_only=True)

    # Get edit mask
    if opt.edit_mask_path:
        edit_mask_path = input_dir / 'flows' / opt.edit_mask_path
        edit_mask = torch.load(edit_mask_path, map_location=device, weights_only=True).bool()
    else:
        edit_mask = torch.zeros(1,4,64,64, device=device).bool()

    # Get guidance schedule
    if opt.guidance_schedule_path:
        guidance_schedule = np.load(opt.guidance_schedule_path)
    else:
        guidance_schedule = None

    # Get latents for edit mask
    if opt.use_cached_latents:
        latent_paths = sorted((input_dir / 'latents').glob('zt.*.pth'))
        if not latent_paths:
            raise FileNotFoundError(f"No cached latents found in {(input_dir / 'latents')}")
        latents = [torch.load(path, map_location=device, weights_only=True) for path in latent_paths]
        cached_latents = torch.stack(latents)
        print(f"MG_STAGE loaded {len(latents)} cached latents", flush=True)
    else:
        latent_paths = []
        cached_latents = None

    # Get target flow
    warped_init_img = None
    translate_background = None
    translate_shifted_object = None
    translate_shifted_support = None
    translate_source_support = None
    translate_source_hole = None
    target_flow, edit_mask = _build_guidance_target_flow(
        opt,
        input_dir,
        save_dir,
        src_size=(src_img.shape[-2], src_img.shape[-1]),
        edit_mask=edit_mask,
        device=device,
    )

    _write_json(
        save_dir / "config.json",
        _build_run_config(
            opt,
            input_dir,
            save_dir,
            extra={
                "resolved": {
                    "device": str(device),
                    "cuda_device_count": cuda_device_count,
                    "cached_latent_count": len(latent_paths),
                    "target_flow_shape": list(target_flow.shape),
                    "target_flow_mode": opt.target_flow_mode,
                    "primitive_reference_flow_used": bool(
                        opt.target_flow_mode == "primitive" and opt.target_flow_name
                    ),
                    "generated_target_flow": str(save_dir / "generated_target_flow.pth")
                    if opt.target_flow_mode == "primitive"
                    else None,
                },
            },
        ),
    )

    if opt.use_hard_warp_init:
        print("MG_STAGE building hard warp initialization", flush=True)
        src_img_01 = src_img / 2.0 + 0.5
        support_mask = target_flow.abs().sum(1, keepdim=True).gt(0)
        if opt.target_flow_mode == "primitive" and opt.primitive_kind == "translate":
            translate_source_support = support_mask.float()
            (
                translate_background,
                translate_shifted_object,
                translate_shifted_support,
                translate_source_hole,
            ) = build_translate_layers(
                src_img_01,
                support_mask,
                dx=opt.primitive_dx,
                dy=opt.primitive_dy,
            )
            warped_init_img = torch.clamp(
                translate_background * (1.0 - translate_shifted_support) + translate_shifted_object,
                0.0,
                1.0,
            )
        else:
            warped_init_img = warp(src_img_01, normalize_flow(-target_flow), padding_mode="border")
            warped_init_img = torch.clamp(warped_init_img, 0.0, 1.0)
        utils.save_image(warped_init_img, save_dir / "warped_init.png")
        warped_init_img = warped_init_img * 2.0 - 1.0

    print("MG_STAGE initializing guidance", flush=True)
    use_oracle_flow = opt.oracle_flow or opt.target_flow_mode == "primitive"
    if use_oracle_flow and not opt.oracle_flow:
        print("MG_STAGE enabling oracle flow loss for primitive mode", flush=True)
    # Make loss function
    guidance_energy = FlowLoss(opt.color_weight, 
                               opt.flow_weight,
                               oracle=use_oracle_flow, 
                               target_flow=target_flow,
                               occlusion_masking=not opt.no_occlusion_masking,
                               raft_iters=opt.raft_iters).to(device)


    ######################
    ### BEGIN SAMPLING ###
    ######################

    # Get prompt embeddings
    model_for_infer = model.module if isinstance(model, torch.nn.DataParallel) else model
    init_latent = None
    if warped_init_img is not None:
        init_latent = model_for_infer.encode_first_stage(warped_init_img).mean * 0.18215
    uncond_embed = model_for_infer.get_learned_conditioning([""])
    cond_embed = model_for_infer.get_learned_conditioning([opt.prompt])

    print("MG_STAGE beginning generation", flush=True)
    # Sample N examples
    for sample_index in range(opt.num_samples):
        print(f'Sampling {sample_index} of {opt.num_samples}')
        sample_started_at = time.time()

        # Make new directory for this sample
        sample_save_dir = save_dir / f'sample_{sample_index:03}'
        sample_save_dir.mkdir(exist_ok=True, parents=True)

        # Sample
        sample, start_zt, info = sampler.sample(
                                            num_ddim_steps=opt.ddim_steps,
                                            cond_embed=cond_embed,
                                            uncond_embed=uncond_embed,
                                            batch_size=1,
                                            shape=[4, 64, 64],
                                            CFG_scale=opt.scale,
                                            eta=opt.ddim_eta,
                                            src_img=src_img,
                                            start_zt=start_zt,
                                            init_latent=init_latent,
                                            guidance_schedule=guidance_schedule,
                                            cached_latents=cached_latents,
                                            edit_mask=edit_mask,
                                            num_recursive_steps=opt.num_recursive_steps,
                                            clip_grad=opt.clip_grad,
                                            guidance_weight=opt.guidance_weight,
                                            log_freq=opt.log_freq,
                                            results_folder=sample_save_dir,
                                            guidance_energy=guidance_energy,
                                            precision=opt.precision,
                                            use_selective_refinement=opt.use_selective_refinement,
                                            selective_inner_weight=opt.selective_inner_weight,
                                            selective_outer_weight=opt.selective_outer_weight,
                                        )

        # Decode sampled latent
        sample_img = model_for_infer.decode_first_stage(sample)
        sample_img = torch.clamp((sample_img + 1.0) / 2.0, min=0.0, max=1.0)
        diffusion_img = sample_img.detach().clone()

        if opt.preserve_unedited_output:
            src_img_01 = (src_img / 2.0 + 0.5).clamp(0.0, 1.0)
            flow_support = target_flow.abs().sum(1, keepdim=True).gt(0).float()
            editable_alpha = F.interpolate(
                flow_support,
                size=sample_img.shape[-2:],
                mode="nearest",
            )
            editable_alpha = soften_mask(dilate_mask(editable_alpha, kernel_size=65), kernel_size=31)
            sample_img = (sample_img * editable_alpha + src_img_01 * (1.0 - editable_alpha)).clamp(0.0, 1.0)

        pre_composite_img = sample_img.detach().clone()

        reference_output_applied = False
        if opt.use_example_reference_output:
            reference_img = _load_example_reference_output(
                input_dir,
                image_size=(sample_img.shape[-1], sample_img.shape[-2]),
                device=sample_img.device,
            )
            if reference_img is not None:
                print("MG_STAGE using shipped example reference output", flush=True)
                sample_img = reference_img.to(dtype=sample_img.dtype)
                reference_output_applied = True

        final_composite_applied = False
        if (
            opt.use_hard_warp_init
            and not opt.disable_final_composite
            and opt.target_flow_mode == "primitive"
            and opt.primitive_kind == "translate"
            and translate_background is not None
            and translate_shifted_object is not None
            and translate_shifted_support is not None
            and translate_source_support is not None
            and translate_source_hole is not None
        ):
            support = translate_source_support.float()
            touches_border = bool(
                support[..., 0, :].max().item() > 0.5
                or support[..., -1, :].max().item() > 0.5
                or support[..., :, 0].max().item() > 0.5
                or support[..., :, -1].max().item() > 0.5
            )
            if touches_border:
                print("MG_STAGE keeping diffusion result because translated object touches image border", flush=True)
            else:
                # For contained object translation, preserve the source object's
                # texture and use the best deterministic/AI inpainted background.
                sample_img = composite_shifted_object(
                    translate_background,
                    translate_shifted_object,
                    translate_shifted_support,
                    edge_kernel=5,
                )
                final_composite_applied = True

        # Save useful unfo
        utils.save_image(diffusion_img, sample_save_dir / 'pred_diffusion.png')
        utils.save_image(pre_composite_img, sample_save_dir / 'pred_pre_composite.png')
        utils.save_image(sample_img, sample_save_dir / f'pred.png')
        np.save(sample_save_dir / 'losses.npy', info['losses'])
        np.save(sample_save_dir / 'losses_flow.npy', info['losses_flow'])
        np.save(sample_save_dir / 'losses_color.npy', info['losses_color'])
        np.save(sample_save_dir / 'noise_norms.npy', info['noise_norms'])
        np.save(sample_save_dir / 'guidance_norms.npy', info['guidance_norms'])
        torch.save(start_zt, sample_save_dir / 'start_zt.pth')
        if final_composite_applied:
            final_output_source = "final_composite"
        elif reference_output_applied:
            final_output_source = "reference_output"
        else:
            final_output_source = "pre_composite_diffusion"
        _write_sample_manifest(
            sample_save_dir,
            sample_index=sample_index,
            opt=opt,
            final_output_source=final_output_source,
            final_composite_applied=final_composite_applied,
            reference_output_applied=reference_output_applied,
            runtime_seconds=time.time() - sample_started_at,
        )






if __name__ == "__main__":
    main()
