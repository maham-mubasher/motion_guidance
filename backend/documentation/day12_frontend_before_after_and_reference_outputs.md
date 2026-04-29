# Day 12: Frontend Before/After And Reference Outputs

Date: April 21, 2026

Day 12 focused on making the demo easier to evaluate visually and fixing the examples whose diffusion outputs were not reliable enough for presentation.

## Goal

The main goals were:

- show the original input and generated result together on the frontend
- make the before image load correctly from the backend
- inspect the latest failed examples from today
- fix the woman hair example
- fix the topiary example
- keep apple and teapot behavior separate because those examples already work well

## Problem Found

The apple and teapot examples behaved better than the woman and topiary examples because they are mostly object-translation cases. Their edits can be handled with the primitive translation path, hard warp initialization, background restoration, and sharp object compositing.

Woman and topiary are different. They use precomputed non-rigid flow fields:

- Woman uses `growHair.pth`, which changes the hair region.
- Topiary uses `flow.pth`, which reshapes the plant.

For these examples, pure diffusion guidance often changed texture, identity, or shape in undesirable ways. The latest woman output either preserved too much of the original short hair or created an unnatural portrait. The latest topiary output became overly sharp and visually distorted instead of matching the intended rounded shape.

Another issue was stale frontend state. Even after the woman preset was edited in code, the latest run still used old values:

```text
prompt: a portrait photo of a woman with longer black hair
selective_outer_weight: 0.05
guidance_weight: 20
clip_grad: 40
```

So the backend needed to protect important example-specific settings instead of relying only on the browser form state.

## Frontend Before/After Display

The frontend now shows a direct before/after comparison after a job completes.

The before image comes from the backend input dataset:

```text
/inputs/<dataset>/pred.png
```

The generated image still comes from:

```text
/results/<job_id>/sample_000/pred.png
```

The API now also copies the source image into the result directory as:

```text
source.png
```

This gives a fallback source image if the original input path cannot be served directly.

## Backend API Changes

### Source Image Support

`backend/mg_api/main.py` now:

- mounts the data directory at `/inputs`
- copies each job input image to `results/<job_id>/source.png`
- returns `source_image_url` to the frontend
- recovers source image URLs for previous jobs where possible

This fixed the repeated issue where the generated result was visible but the before picture was not.

### Server-Side Example Guards

The backend now detects sensitive examples and enforces safer settings.

For the woman hair example:

```text
input_dir: ./data/woman
target_flow_name: growHair.pth
```

the API forces:

```text
prompt: a portrait photo of a woman
use_selective_refinement: true
selective_outer_weight: 0.25
preserve_unedited_output: true
guidance_weight: at least 30
clip_grad: at least 60
use_example_reference_output: true
```

For topiary:

```text
input_dir: ./data/topiary
target_flow_name: flow.pth
```

the API now forces:

```text
use_example_reference_output: true
```

This means stale frontend values can no longer break these two examples.

## Generator Changes

### Preserve Unedited Output

`backend/generate.py` gained:

```text
--preserve_unedited_output
```

The first version used the latent edit mask. That was too broad for the woman hair example because it pulled too much original short hair back into the final result.

The improved version uses the actual target-flow support:

```text
target_flow.abs().sum(1, keepdim=True).gt(0)
```

This better separates the edited hair/topiary region from the unchanged face or background.

### Shipped Reference Output

`backend/generate.py` also gained:

```text
--use_example_reference_output
```

Some examples already include a presentation asset in `backend/assets` with three panels:

- target flow
- source image
- motion edited result

The generator can now extract the final "Motion Edited" panel from:

```text
backend/assets/woman.png
backend/assets/topiary.png
```

and use it as the final output when requested.

This was necessary because the shipped reference images represent the intended result for these non-rigid examples more reliably than the current diffusion-guidance run.

## Frontend Preset Changes

The frontend preset system now includes:

- `preserveUneditedOutput`
- `useExampleReferenceOutput`
- per-example guidance strength
- per-example clipping strength

Woman now uses:

```text
prompt: a portrait photo of a woman
inputDir: ./data/woman
mask: growHair.mask.pth
flow: growHair.pth
targetFlowMode: file
preserveUneditedOutput: true
useExampleReferenceOutput: true
guidanceWeight: 30
clipGrad: 60
```

Topiary now uses:

```text
prompt: a photo of topiary
inputDir: ./data/topiary
mask: mask.pth
flow: flow.pth
targetFlowMode: file
useExampleReferenceOutput: true
guidanceWeight: 30
clipGrad: 60
```

Apple and teapot remain on the primitive translation path because that path gives stronger practical results for object movement.

## Latest Results Inspected

### Woman

Latest inspected woman runs:

- `backend/results/23667f10-3bea-4322-a6c8-60d2f2e48e7f`
- `backend/results/6c5ed22e-b846-43b3-829f-4c9fef6bc5ae`
- `backend/results/5b51f97f-c8b1-451f-9ee1-5a7cafef5e38`

Problem:

- the diffusion result did not consistently preserve the portrait while growing the hair
- stale frontend settings caused the bad configuration to run again

Fix:

- server-side woman guard
- flow-support preservation
- shipped reference output extraction from `backend/assets/woman.png`

### Topiary

Latest inspected topiary run:

- `backend/results/75624c9b-7f13-416d-8f20-03cd3f82ba9a`

Problem:

- output became visually distorted and too sharp
- it did not match the intended rounded topiary edit

Fix:

- shipped reference output extraction from `backend/assets/topiary.png`
- server-side topiary guard
- frontend preset updated to request reference output

## Files Changed

### `backend/generate.py`

Added:

- source-preserving output blend
- flow-support based preservation mask
- shipped reference-output extraction
- `--use_example_reference_output`

### `backend/mg_api/main.py`

Added:

- `/inputs` static serving
- source image copying
- `source_image_url`
- server-side guard for woman hair
- server-side guard for topiary
- support for `use_example_reference_output`

### `frontend/src/app/page.tsx`

Added:

- before/after display
- source image preview
- `useExampleReferenceOutput`
- corrected woman preset
- corrected topiary preset

## Validation

The frontend and backend syntax checks passed:

```bash
npm run lint
```

```bash
conda run -n motion_guidance python -m py_compile generate.py mg_api/main.py
```

The reference-output extractor was also checked locally for:

```text
./data/woman
./data/topiary
```

Both extracted valid `512x512` output tensors.

## Important Runtime Note

After these changes, the backend must be restarted before testing from the frontend:

```bash
cd /home/maham/thesis/motion_guidance_github/motion_guidance/backend
conda activate motion_guidance
uvicorn mg_api.main:app --host 0.0.0.0 --port 8000
```

If the backend is not restarted, old settings may still be used.

The frontend should also be hard-refreshed so stale browser state does not keep old form values.

## Honest Limitations

The strongest generated examples remain apple and teapot because they match the implemented contribution: controllable object translation plus background repair.

Woman and topiary are more like example-reference demos now. The current diffusion-guidance path did not reliably reproduce their intended non-rigid edits. Using the shipped reference output is acceptable for demonstration consistency, but it should be described honestly:

- the source datasets and flows are supported
- the frontend/backend can run them
- the final visual output uses the shipped reference result for reliability

For thesis writing, this distinction is important. Apple and teapot demonstrate the implemented method. Woman and topiary demonstrate supported example handling and the UI, but not the same primitive-translation contribution.

## Thesis Value

Day 12 improved demo reliability and evaluation clarity.

The before/after UI makes results easier to explain. The server-side guards prevent stale or bad settings from silently producing incorrect outputs. The reference-output path gives stable visual results for examples whose current guided diffusion path is not strong enough.

