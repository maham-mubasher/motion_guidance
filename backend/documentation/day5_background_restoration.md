# Day 5: Dedicated Background Restoration Module

Day 5 focuses on the one remaining weak point from Day 4:
- the original object location still does not always blend naturally back into the wooden table.

## Goal

Keep the current motion pipeline stable while isolating background restoration as its own thesis-visible contribution.

The Day 5 objective is:
- preserve the successful apple translation behavior,
- keep the moved apple visually complete,
- improve reconstruction of the source hole with a cleaner and more modular implementation.

## What Was Added

- `backend/background_restoration.py`
  - extracted background restoration into its own module
  - includes:
    - spatial shifting utilities
    - local masked-region filling
    - mask softening and erosion helpers
    - directional background filling
    - translated-layer construction for source/destination compositing

- `backend/generate.py`
  - now uses the new background restoration module instead of keeping all restoration logic inline

## Mathematical Idea

Day 5 mainly reorganized the restoration step into explicit operations:

1. Binary source-hole mask
   - let `M` be the mask of the removed object region
   - background pixels are `1 - M`

2. Local inpainting by neighborhood averaging
   - unknown pixels are filled from nearby known pixels
   - conceptually:
     - `I_new(p) = average of valid neighbors around p`
   - this is a simple local interpolation rule, not a learned inpainting model

3. Soft mask blending
   - instead of replacing regions with a hard binary cut, the code computes a softened alpha mask
   - final blend:
     - `I_out = (1 - alpha) * I_base + alpha * I_fill`

4. Mask erosion
   - the source region is shrunk before blending
   - this means only the inner part of the old apple region is forcibly replaced, while the outer ring is left for smoother continuity

## Code Change Logic

In code, Day 5 introduced:
- `shift_tensor_2d(...)`
  - translates tensors spatially without wraparound
- `inpaint_masked_region(...)`
  - fills missing pixels using local valid neighbors
- `soften_mask(...)`
  - converts a hard binary mask into a soft blending mask
- `erode_mask(...)`
  - shrinks the active restoration region to reduce visible seams

The expectation was:
- modularize restoration,
- preserve the moved apple,
- and replace the old apple region with a background estimate more cleanly than the earlier inline logic.

## Why This Matters

This improves the thesis structure because the code now shows a distinct refinement component:
- motion generation,
- selective refinement,
- hard warp initialization,
- background restoration.

That makes the contribution easier to explain and defend.

## Day 5 Evaluation Focus

When testing Day 5, evaluate:
- whether the old apple location looks more like real wooden-table texture,
- whether the circular patch is reduced,
- whether the moved apple stays complete,
- whether seam artifacts around the translated apple remain acceptable.

## Image References

- Original apple:
  - [backend/data/apple/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/data/apple/pred.png)
- Day 4 comparison result:
  - [backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png)
- Latest Day 5 result:
  - [backend/results/0e227235-4936-46c5-9ddc-ca9532f8473c/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/0e227235-4936-46c5-9ddc-ca9532f8473c/sample_000/pred.png)

Observation:
- Day 5 keeps the moved apple visually strong.
- The old apple location still shows an obvious synthetic patch, so the module refactor is useful structurally but background realism is still not solved.

## Current Recommended Test

- Prompt: `an apple on a wooden table`
- Input Dir: `./data/apple`
- Mask File: `right.mask.pth`
- Target Flow Source: `Primitive Motion`
- Primitive Type: `Translate`
- Translate X: `120` or `150`
- Translate Y: `0`
- Reference Support Flow: `right.pth`
- Use hard warp initialization: `on`
- Use selective refinement: `on`
- Inner Weight: `1.0`
- Outer Weight: `0.15`
- DDIM Steps: `40`
- Guidance Weight: `30`
- Clip Grad: `60`
- RAFT Iters: `1`
- Recursive Steps: `1`

## Day 5 Success Criterion

Day 5 is successful if:
- background restoration is cleaner than Day 4,
- the motion result does not regress,
- the repo now contains a dedicated, visible background-restoration module for the thesis narrative.
