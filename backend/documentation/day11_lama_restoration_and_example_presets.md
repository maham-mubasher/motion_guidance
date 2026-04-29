# Day 11: LaMa Restoration And Example Presets

Date: April 17, 2026

Day 11 focused on fixing the remaining visual problems in the apple and teapot results, then making the workflow usable for other examples without manually typing every input value.

## Goal

The main goal was to remove the last design issues in the output image:

- the old object location should become believable background
- the moved apple should stay sharp, not blurry
- the object should not have a dark or green compositing edge
- the frontend should fill correct values for supported examples automatically
- unsupported examples, especially cat, should fail clearly instead of producing confusing errors

## Problem Found

The earlier restoration pipeline improved the result, but it still had two important weaknesses.

First, the background behind the old apple position was difficult to reconstruct with only local patch matching or OpenCV inpainting. The table texture is directional and uneven, so small local mistakes became visible immediately.

Second, some blending methods made the apple worse even when the background improved. Poisson-style seamless cloning and overly soft masks could blur the apple texture, darken the edge, or introduce color drift. For the apple and teapot examples, the object itself was already good in the original image, so the best strategy was to preserve the original object texture and only repair the newly exposed background.

The cat example also exposed a workflow issue. The repository contains `backend/assets/cat.png`, but that file is a presentation image, not a runnable Motion Guidance dataset. A valid runnable example needs an input image, mask, flow, inversion latent, and cached DDIM latents.

## Research-Backed Fix

The best practical solution was to separate the problem into two parts:

1. Repair the background with a stronger inpainting model.
2. Composite the original object sharply at the new location.

For background restoration, the pipeline now supports LaMa / Big-LaMa through `simple-lama-inpainting`. LaMa is better suited than small local patch filling for larger missing regions because it can synthesize plausible structure inside the removed-object area.

For object preservation, the final translation result avoids blurring the object through Poisson blending. Instead, it shifts the original object with its mask and composites it over the repaired background using a controlled feather only at the boundary.

This gives the desired behavior:

- LaMa handles the old object hole.
- The original object remains sharp.
- The boundary is softened enough to avoid a harsh cut.
- OpenCV inpainting remains available as a fallback if LaMa is unavailable.

## Code Changes

### `backend/background_restoration.py`

Added a stronger restoration module:

- optional LaMa backend using `simple-lama-inpainting`
- lazy model loading for the Big-LaMa model
- mask dilation helpers
- PIL / NumPy conversion helpers
- OpenCV Telea fallback restoration
- `inpaint_background_region`
- `composite_shifted_object`
- `patch_based_background_fill` updated to try LaMa first, then OpenCV

The most important practical change is `composite_shifted_object`. It preserves the shifted original object instead of letting diffusion or Poisson blending soften it.

### `backend/generate.py`

Added support for more controlled example runs:

- primitive target-flow mode
- translation primitive values
- edit-mask dilation
- optional hard-warp initialization
- selective refinement controls
- final translate-mode postprocess using:
  - repaired background
  - shifted original object
  - shifted object support mask

For the apple and teapot examples, this means the final image is assembled from a clean restored background and the sharp original object texture.

### `backend/mg_api/main.py`

Improved backend usability and error handling:

- added request fields for primitive flow and refinement settings
- added preflight validation for input directories, masks, flows, latents, and cached latents
- added clearer HTTP 400 errors when files are missing
- added job process tracking
- added a cancel endpoint for stopping a running generation

This prevents confusing late failures after the model has already started loading.

### `frontend/src/app/page.tsx`

Added example presets so the user does not need to manually type every value:

- Apple
- Teapot
- Lion
- Woman
- Tree
- Topiary

The frontend now fills prompt, input directory, mask file, flow file, target-flow mode, primitive values, hard-warp settings, and selective refinement settings from the selected preset.

Cat is shown as unavailable until a real `./data/cat` dataset is created.

The frontend also now displays detailed backend error messages and includes a Stop Generation button.

### `backend/requirements.txt`

Added:

```text
simple-lama-inpainting==0.1.2
```

### `Commands`

Updated with practical run values for the supported examples and a clear note explaining why the cat example is not runnable yet.

## Example Presets

The supported examples now fall into two groups.

### Primitive Translation Examples

These examples are best handled with the new primitive translation path:

- Apple
  - prompt: `an apple on a wooden table`
  - input directory: `./data/apple`
  - mask: `right.mask.pth`
  - flow: `right.pth`
  - primitive: translate
  - dx: `150`
  - dy: `0`

- Teapot
  - prompt: `a teapot floating in water`
  - input directory: `./data/teapot`
  - mask: `down150.mask.pth`
  - flow: `down150.pth`
  - primitive: translate
  - dx: `0`
  - dy: `150`

- Lion
  - prompt: `a photo of a lion`
  - input directory: `./data/lion`
  - mask: `left.mask.pth`
  - flow: `left.pth`
  - primitive: translate
  - dx: `-100`
  - dy: `0`

These are the strongest match for the LaMa plus sharp-composite strategy because the object is moved and a newly exposed source region needs background restoration.

### File-Flow Examples

These examples use their existing precomputed flow files:

- Woman
  - prompt: `a portrait photo of a woman`
  - input directory: `./data/woman`
  - mask: `growHair.mask.pth`
  - flow: `growHair.pth`

- Tree
  - prompt: `a painting of a lone tree`
  - input directory: `./data/tree`
  - mask: `squeeze.mask.pth`
  - flow: `squeeze.pth`

- Topiary
  - prompt: `a photo of topiary`
  - input directory: `./data/topiary`
  - mask: `mask.pth`
  - flow: `flow.pth`

These examples are different from apple and teapot because they are not simple object translations. They should still run through the preset system, but they may need specialized postprocessing if their edits create visible holes or texture distortion.

## Cat Example Status

Cat is not runnable yet because the repository does not currently contain a complete `./data/cat` Motion Guidance input folder.

To make cat work properly, the dataset should include:

- `backend/data/cat/pred.png`
- `backend/data/cat/flows/right150.mask.pth`
- `backend/data/cat/flows/right150.pth`
- `backend/data/cat/start_zt.pth`
- cached DDIM latents for the selected run settings

After that, the intended cat preset can be:

- prompt: `a photo of a cat`
- input directory: `./data/cat`
- mask: `right150.mask.pth`
- flow: `right150.pth`
- primitive: translate
- dx: `150`
- dy: `0`

Until those files exist, the correct behavior is to show a clear missing-data error rather than pretending the cat example is available.

## Results

The apple and teapot examples are now the strongest practical results.

The important quality improvement is that the background restoration and object rendering are no longer fighting each other. The background can be inpainted by LaMa, while the object keeps its original sharp texture.

Fixed result references:

- Latest screenshot-matching repaired output:
  - [backend/results/53aa06c0-4f92-4b0d-aa00-d51ba95c43b5/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/53aa06c0-4f92-4b0d-aa00-d51ba95c43b5/sample_000/pred.png)
- Apple repaired output:
  - [backend/results/d54ead33-b5fa-42f3-85ad-1d36ec32c426/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/d54ead33-b5fa-42f3-85ad-1d36ec32c426/sample_000/pred.png)

## Validation

The code was checked with:

```bash
npm run lint
```

and:

```bash
conda run -n motion_guidance python -m py_compile backend/mg_api/main.py backend/generate.py backend/background_restoration.py
```

Both checks passed after the implementation.

## Honest Limitations

The apple and teapot path is now strong because translation has a clear geometric interpretation: remove the object from one place, repair that background, and paste the sharp object into the new place.

The woman, tree, and topiary examples are different. Their flow fields can represent deformation, squeezing, growth, or non-rigid motion. These examples may need separate postprocessing rules because a simple shifted-object composite is not always the right final operation.

Cat still needs a real dataset folder before it can be treated like the other examples.

## Thesis Value

Day 11 improves the thesis story significantly.

Earlier days showed that lightweight restoration was possible but still imperfect. Today adds a stronger practical solution:

- learned inpainting for background repair
- sharp object preservation for translated examples
- automatic frontend presets for repeatable testing
- clear validation for missing datasets

This makes the system more demo-ready and gives a stronger final method for object-translation examples.

