# Day 9: Thesis Outline And Presentation Structure

This file provides a chapter-wise writing outline and a presentation-ready structure for the thesis.

## Proposed Thesis Title Direction

Possible working title:

**A Lightweight Motion-Guided Image Editing Framework Using Motion Primitives, Selective Refinement, and Hard Warp Initialization**

## Thesis Contribution Statement

This thesis does not propose a new large generative model.
Instead, it develops a lightweight controllable extension of Motion Guidance by adding:
- user-controlled motion primitives,
- selective refinement,
- hard warp initialization,
- and modular background restoration.

The contribution is practical, understandable, and implementable, which fits the intended thesis scope.

## Chapter-Wise Outline

### Chapter 1: Introduction

Explain:
- the problem of controllable image editing
- why motion-guided editing is useful
- the limitation of relying only on precomputed flow targets
- the motivation for a lightweight controllable variant

End this chapter with:
- thesis objectives
- research questions
- contribution summary

### Chapter 2: Related Work

Cover:
- diffusion-based image editing
- motion-guided editing
- optical-flow-guided manipulation
- controllable editing with spatial constraints
- inpainting / background restoration methods as supporting context

### Chapter 3: Baseline System

Describe:
- the original Motion Guidance pipeline
- the role of target flow, edit mask, and diffusion guidance
- the baseline implementation frozen in Day 1

Use:
- [backend/day1_baseline_results.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day1_baseline_results.md)

### Chapter 4: Proposed Method

Break the method into four sub-parts:

1. Motion primitives
   - user-controlled translation/scale/rotation/stretch
   - code reference:
     - [backend/motion_primitives.py](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/motion_primitives.py)

2. Selective refinement
   - stronger editing in target regions, weaker elsewhere
   - code reference:
     - [backend/selective_refinement.py](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/selective_refinement.py)

3. Hard warp initialization
   - source image is first moved explicitly, then refined by diffusion
   - note reference:
     - [backend/day3_hard_warp_init.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day3_hard_warp_init.md)

4. Background restoration
   - source-hole cleanup after object movement
   - code reference:
     - [backend/background_restoration.py](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/background_restoration.py)

### Chapter 5: Experiments And Results

Organize the experiments chronologically:
- Day 1 baseline
- Day 2 primitive integration
- Day 3 hard warp success
- Day 4 best refinement result
- Day 5–7 restoration experiments

Use:
- [backend/day9_final_results_table.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day9_final_results_table.md)
- [backend/final_progress_summary.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/final_progress_summary.md)

Key images to show:
- original apple
- baseline apple
- best Day 4 result
- final Day 7 practical result

### Chapter 6: Discussion, Limitations, And Future Work

State clearly:
- the system succeeds in controllable motion editing
- object movement is now operational and practical
- background restoration is improved but still not fully seamless

Future work:
- stronger inpainting methods
- learned restoration
- more complex scenes and object types

### Chapter 7: Conclusion

Summarize:
- baseline established
- controllable extension implemented
- multiple refinements tested
- final method is practical and defendable
- limitations are understood and documented

## Presentation Structure

For a presentation or viva, use this order:

1. Problem
   - controllable motion editing is difficult

2. Baseline limitation
   - original system relies heavily on precomputed flows
   - object movement is not directly controllable by the user

3. Proposed idea
   - motion primitives
   - selective refinement
   - hard warp initialization
   - background restoration

4. Main result
   - object translation works
   - user control is improved

5. Limitation
   - background reconstruction is still not perfect

6. Future work
   - better inpainting / restoration

## Best Files To Reuse While Writing

- [backend/final_progress_summary.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/final_progress_summary.md)
- [backend/day9_final_results_table.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day9_final_results_table.md)
- [backend/day3_hard_warp_init.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day3_hard_warp_init.md)
- [backend/day7_patch_based_restoration.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day7_patch_based_restoration.md)

## Final Writing Advice

When writing the thesis, frame the work as:
- a lightweight controllable extension,
- not a replacement of the full original Motion Guidance method,
- and not a claim of perfect photorealism.

That framing is honest, strong, and defendable.
