# Day 3: Hard Warp Initialization

Day 2 showed that primitive flow as soft guidance alone was not enough to move the object reliably.  
Day 3 strengthened the method by explicitly moving the object first and then using diffusion as a refinement stage.

## Goal

Make primitive translation actually move the apple to a new position while keeping:
- the apple visually complete,
- the old location filled with wooden-table background,
- the edit localized instead of letting diffusion rewrite the whole scene.

## What Was Implemented

- `backend/generate.py`
  - added hard warp initialization for primitive translation
  - added translated object compositing
  - added source-region cleanup and destination-region completion logic
- `backend/ldm/models/diffusion/ddim_with_grad.py`
  - fixed the hard-warp latent handling so the warped latent is first diffused to the starting DDIM timestep instead of being treated as an already-noisy state
- `backend/mg_api/main.py`
  - forwards `use_hard_warp_init`
- `frontend/src/app/page.tsx`
  - keeps apple as the default Day 3 test case
  - exposes hard warp initialization in the UI

## Main Bugs Found And Fixed

1. Primitive translation was being used only as a soft loss, so the apple often did not move.
2. The first hard-warp attempt created broken initialization artifacts.
3. The hard-warp latent was incorrectly injected into sampling as a clean latent at a noisy timestep.
4. The destination region was not fully opened for editing, so the translated apple was incomplete.
5. The source region kept a ghost / transparent trace of the original apple.
6. The original apple location was being filled with poor texture instead of table background.

## Final Day 3 Result

The final apple runs show a real improvement over Day 2:
- the apple now changes position,
- the translated apple is much more complete,
- the old location is cleaned more convincingly than before,
- the method is now behaving like a real object-move pipeline rather than a weak flow preference.

This is the key Day 3 success:
- **hard warp initialization made primitive translation operational**

## Remaining Limitation

The result is much better, but not yet perfect:
- the original apple position can still show imperfect wooden-table reconstruction,
- the background replacement is improved but not fully seamless in every run.

So the honest conclusion is:
- object translation now works,
- object completeness is much better,
- source-hole cleanup still has room for refinement.

## Final Apple Test Settings

Frontend settings used for the successful Day 3 apple run:
- Prompt: `an apple on a wooden table`
- Input Dir: `./data/apple`
- Mask File: `right.mask.pth`
- Target Flow Source: `Primitive Motion`
- Primitive Type: `Translate`
- Translate X: `150`
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

## Day 3 Conclusion

Day 3 is a meaningful improvement over Day 2.

Day 2 conclusion:
- primitive motion was integrated, but translation was not reliably enforced.

Day 3 conclusion:
- primitive translation now produces an actual moved apple,
- the motion result is substantially more useful,
- the next refinement target should be better background restoration at the original location.
