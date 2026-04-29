# Day 2: Translate Primitive

This Day 2 step adds the first thesis-method path:
- generate target flow from a simple motion primitive,
- keep the same edit mask and diffusion pipeline,
- compare primitive-generated motion against the original precomputed flow baseline.

## What was added

- `backend/motion_primitives.py`
  - builds primitive target flows for `translate`, `scale`, `rotate`, and `stretch`
- `backend/generate.py`
  - now supports `--target_flow_mode primitive`
- `backend/mg_api/main.py`
  - accepts primitive parameters from the frontend
- `frontend/src/app/page.tsx`
  - allows choosing `Precomputed Flow File` or `Primitive Motion`

## First recommended Day 2 run

Use the `teapot` example first.

Frontend settings:
- Prompt: `a teapot floating in water`
- Input Dir: `./data/teapot`
- Mask File: `down150.mask.pth`
- Target Flow Source: `Primitive Motion`
- Primitive Type: `Translate`
- Translate X: `0`
- Translate Y: `12`
- DDIM Steps: `40`
- Guidance Weight: `30`
- Clip Grad: `60`
- RAFT Iters: `1`
- Recursive Steps: `1`

## Expected artifact

The generated run directory should include:
- the normal sample output image
- `generated_target_flow.pth` saved at the run root

## Immediate comparison task

Compare:
- baseline teapot result using `down150.pth`
- primitive teapot result using `Translate Y = 12`

Record:
- visual similarity,
- whether motion direction is correct,
- whether object shape is preserved better or worse,
- whether the primitive version is easier to control.

## First comparison note

Comparison pair:
- Baseline: `backend/results/teapot.down150/sample_000/pred.png`
- Primitive: `backend/results/eb7a9d13-fc8a-4b35-a295-329e90e0ce08/sample_000/pred.png`

Observed settings:
- Baseline used the original file flow `down150.pth`
- Primitive run used `target_flow_mode=primitive`, `primitive_kind=translate`, `dx=0`, `dy=12`

Visual observations:
- The primitive run successfully moves the object downward in the water, so the motion direction is correct.
- The primitive run is easy to control because the motion is defined directly by `dx` and `dy` instead of needing a precomputed flow file.
- The baseline preserves the teapot identity better. Its handle and open top remain closer to the original object semantics.
- The primitive result changes the object shape more strongly. It becomes a rounder, more lid-like red vessel, and the original teapot handle/opening structure is less consistent.
- Water context remains plausible in both outputs, but the primitive run introduces more object-level deformation than the baseline.

Conclusion:
- Primitive translation improves controllability and makes the motion parameter explicit.
- Baseline file flow still gives better object preservation on this teapot example.
- This is a useful thesis result: the primitive method is easier to use, but it needs selective refinement or stronger structure preservation to match baseline shape fidelity.

## Apple comparison note

Comparison pair:
- Baseline: `backend/results/apple.right/sample_000/pred.png`
- Primitive: `backend/results/23a88522-ab10-42c5-83b1-af76f1502f49/sample_000/pred.png`

Observed settings:
- Baseline used the original file flow `right.pth`
- Primitive run used `target_flow_mode=primitive`, `primitive_kind=translate`, `dx=50`, `dy=0`

Visual observations:
- The primitive `dx` value is reaching the backend correctly, so this is not a UI or API bug.
- The primitive apple output does not behave like a strong rightward translation.
- Instead of clearly shifting the apple, the model mostly re-synthesizes a centered, cleaner-looking apple.
- The baseline result shows a more explicit edit outcome tied to the original rightward flow file.
- The primitive result preserves a plausible apple appearance, but loses the intended positional control.

Conclusion:
- On apple, the current primitive-only method is not strong enough to enforce reliable object translation.
- This supports the next thesis step: selective refinement or stronger spatial preservation is needed so primitive motion affects the masked object more locally and predictably.
