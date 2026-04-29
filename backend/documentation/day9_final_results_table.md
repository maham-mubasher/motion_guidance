# Day 9: Final Results Table

This file packages the main experimental stages into a thesis-ready comparison table.

## Core Images

- Original apple:
  - [backend/data/apple/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/data/apple/pred.png)
- Baseline apple:
  - [backend/results/apple.right/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/apple.right/sample_000/pred.png)
- Best Day 4 result:
  - [backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png)
- Final practical Day 7 result:
  - [backend/results/38f19e3c-9369-4967-b97b-fae7fdd27d46/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/38f19e3c-9369-4967-b97b-fae7fdd27d46/sample_000/pred.png)

## Stage Comparison

| Stage | Main idea | What improved | Main weakness | Representative image |
| --- | --- | --- | --- | --- |
| Day 1 Baseline | Original Motion Guidance baseline with wrapper | Established comparison point and stable example outputs | No thesis-specific controllable motion method yet | [apple baseline](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/apple.right/sample_000/pred.png) |
| Day 2 | Motion primitives + selective refinement scaffold | Explicit controllable primitive inputs added | Primitive translation alone was too weak to move objects reliably | [day2 note](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day2_translate_primitive.md) |
| Day 3 | Hard warp initialization | Primitive translation became operational and the apple actually moved | Old object location restoration still weak | [day3 note](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day3_hard_warp_init.md) |
| Day 4 | Better source cleanup and compositing | Best early practical visual result; moved apple looked strong | Source-hole background still synthetic | [best day4](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png) |
| Day 5 | Dedicated background restoration module | Restoration made modular and thesis-visible | Visual realism still limited | [day5 note](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day5_background_restoration.md) |
| Day 6 | Texture-aware directional fill | Useful as a structured restoration experiment | Did not surpass Day 4 | [day6 note](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day6_texture_aware_restoration.md) |
| Day 7 | Patch-based + OpenCV inpainting restoration | Strong final practical result and more advanced restoration method | Background still not fully seamless; faint outline may remain | [final day7](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/38f19e3c-9369-4967-b97b-fae7fdd27d46/sample_000/pred.png) |

## Best Result Selection

Best baseline image:
- [backend/results/apple.right/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/apple.right/sample_000/pred.png)

Best comparison-quality refined result:
- [backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png)

Best final practical result:
- [backend/results/38f19e3c-9369-4967-b97b-fae7fdd27d46/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/38f19e3c-9369-4967-b97b-fae7fdd27d46/sample_000/pred.png)

## Final Takeaway

The final system is clearly stronger than the starting baseline in controllable motion editing.
The main improvement is not perfect photorealism, but a practical and understandable pipeline that:
- allows user-controlled motion,
- produces real object movement,
- and documents the tradeoff between controllability and seamless restoration.
