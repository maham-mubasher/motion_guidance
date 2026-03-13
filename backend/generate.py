import argparse
from pathlib import Path
from omegaconf import OmegaConf

from PIL import Image
import numpy as np

import torch
from torchvision import utils
from torchvision.transforms.functional import to_tensor

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad
from losses import FlowLoss



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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    return model


def main():
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

    # Guidance args
    parser.add_argument("--target_flow_name", type=str, default=None, help='Path to target image. If no path, then default to constant flow')
    parser.add_argument("--edit_mask_path", type=str, default='', help='path to edit mask')
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

    opt = parser.parse_args()
    input_dir = Path(opt.input_dir)

    save_dir = Path(opt.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save config
    torch.save(opt, save_dir / 'config.pth')

    # Print for sanity check
    print(opt)


    ######################
    ### SETUP SAMPLING ###
    ######################

    print("MG_STAGE preparing sampler", flush=True)
    # Load model
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    # Setup model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if (not opt.disable_dataparallel) and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
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
        start_zt = torch.load(input_dir / 'start_zt.pth', map_location=device)

    # Get edit mask
    if opt.edit_mask_path:
        edit_mask_path = input_dir / 'flows' / opt.edit_mask_path
        edit_mask = torch.load(edit_mask_path, map_location=device).bool()
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
        latents = [torch.load(path, map_location=device) for path in latent_paths]
        cached_latents = torch.stack(latents)
        print(f"MG_STAGE loaded {len(latents)} cached latents", flush=True)
    else:
        cached_latents = None

    # Get target flow
    target_flow_path = input_dir / 'flows' / opt.target_flow_name
    target_flow = torch.load(target_flow_path, map_location=device)

    print("MG_STAGE initializing guidance", flush=True)
    # Make loss function
    guidance_energy = FlowLoss(opt.color_weight, 
                               opt.flow_weight,
                               oracle=opt.oracle_flow, 
                               target_flow=target_flow,
                               occlusion_masking=not opt.no_occlusion_masking,
                               raft_iters=opt.raft_iters).to(device)


    ######################
    ### BEGIN SAMPLING ###
    ######################

    # Get prompt embeddings
    model_for_infer = model.module if isinstance(model, torch.nn.DataParallel) else model
    uncond_embed = model_for_infer.get_learned_conditioning([""])
    cond_embed = model_for_infer.get_learned_conditioning([opt.prompt])

    print("MG_STAGE beginning generation", flush=True)
    # Sample N examples
    for sample_index in range(opt.num_samples):
        print(f'Sampling {sample_index} of {opt.num_samples}')

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
                                            guidance_schedule=guidance_schedule,
                                            cached_latents=cached_latents,
                                            edit_mask=edit_mask,
                                            num_recursive_steps=opt.num_recursive_steps,
                                            clip_grad=opt.clip_grad,
                                            guidance_weight=opt.guidance_weight,
                                            log_freq=opt.log_freq,
                                            results_folder=sample_save_dir,
                                            guidance_energy=guidance_energy,
                                            precision=opt.precision
                                        )

        # Decode sampled latent
        sample_img = model_for_infer.decode_first_stage(sample)
        sample_img = torch.clamp((sample_img + 1.0) / 2.0, min=0.0, max=1.0)

        # Save useful unfo
        utils.save_image(sample_img, sample_save_dir / f'pred.png')
        np.save(sample_save_dir / 'losses.npy', info['losses'])
        np.save(sample_save_dir / 'losses_flow.npy', info['losses_flow'])
        np.save(sample_save_dir / 'losses_color.npy', info['losses_color'])
        np.save(sample_save_dir / 'noise_norms.npy', info['noise_norms'])
        np.save(sample_save_dir / 'guidance_norms.npy', info['guidance_norms'])
        torch.save(start_zt, sample_save_dir / 'start_zt.pth')






if __name__ == "__main__":
    main()
