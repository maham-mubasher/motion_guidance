# Day 4: Background Restoration And Evaluation

Day 4 focuses on improving the visual quality of object translation after Day 3 made the apple movement operational.

## Objective

Improve two visible weaknesses from Day 3:
- restore the wooden-table texture more naturally at the original apple location
- reduce visible seams around the moved apple

## Implementation Summary

- improved source-region cleanup in `backend/generate.py`
  - uses multi-sample directional background filling instead of relying only on simple local inpainting
  - prefers copying plausible wooden-table texture from nearby valid background regions
- improved final compositing in `backend/generate.py`
  - uses softened alpha masks instead of hard binary replacement
  - blends the cleaned source region back into the scene more naturally
  - blends the translated apple edges more smoothly at the destination

## Expected Day 4 Improvement

Compared with Day 3, the next apple result should show:
- less damage at the original apple location
- better wooden-table reconstruction
- fewer hard edge artifacts around the moved apple

## Image References

- Original apple:
  - [backend/data/apple/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/data/apple/pred.png)
- Current best Day 4 result:
  - [backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png)

Observation:
- Day 4 produced a visibly moved and mostly complete apple.
- The main remaining issue is a synthetic-looking source patch at the original apple location.

## Evaluation Checklist

For each Day 4 rerun, record:
- run folder name
- translation value used
- whether the apple moved correctly
- whether the apple stayed visually complete
- whether the old apple location looks like natural table texture
- whether there are halos, seams, or blur artifacts

## Current Recommended Test

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

## Day 4 Goal

The Day 4 goal is not to redesign the method.
It is to make the existing Day 3 translation path cleaner and more presentation-ready for thesis examples.
