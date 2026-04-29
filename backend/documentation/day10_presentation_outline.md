# Day 10: Presentation Outline

This file provides a slide-by-slide outline for supervisor discussion, viva preparation, or a final project presentation.

## Slide 1: Title

Suggested title:

**A Lightweight Motion-Guided Image Editing Framework Using Motion Primitives, Selective Refinement, and Hard Warp Initialization**

Include:
- your name
- department / university
- supervisor name

## Slide 2: Problem

Main message:
- controllable motion editing is difficult
- original Motion Guidance relies heavily on precomputed flow targets
- that makes practical user control less direct

One short line:
- “How can we make motion-guided editing more controllable with a lightweight method?”

## Slide 3: Baseline System

Explain briefly:
- original Motion Guidance pipeline
- target flow + edit mask + diffusion guidance
- baseline established in Day 1

Show:
- baseline apple image
  - [backend/results/apple.right/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/apple.right/sample_000/pred.png)

## Slide 4: Proposed Contribution

Present the four main components:
- motion primitives
- selective refinement
- hard warp initialization
- background restoration

Main message:
- the thesis contribution is a lightweight extension, not a new large generative model

## Slide 5: Method Pipeline

Show the pipeline in order:
1. user selects primitive motion
2. target flow is generated
3. hard warp initialization moves the object
4. diffusion refinement improves the result
5. background restoration cleans the old object region

Main message:
- object movement becomes explicit and user-controlled

## Slide 6: Day-by-Day Progress

Short summary:
- Day 1: baseline freeze
- Day 2: primitive motion integration
- Day 3: hard warp makes motion operational
- Day 4: best practical visual refinement
- Day 5–7: restoration experiments
- Day 8–10: evaluation and packaging

## Slide 7: Main Results

Show:
- original apple
  - [backend/data/apple/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/data/apple/pred.png)
- baseline apple
  - [backend/results/apple.right/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/apple.right/sample_000/pred.png)
- best Day 4 result
  - [backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png)
- final practical Day 7 result
  - [backend/results/38f19e3c-9369-4967-b97b-fae7fdd27d46/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/38f19e3c-9369-4967-b97b-fae7fdd27d46/sample_000/pred.png)

Main message:
- the proposed method improves controllable object movement substantially over the baseline workflow

## Slide 8: What Worked

Points to say:
- primitive translation became operational
- user control improved
- the pipeline works end to end
- the system is practical and demo-ready

## Slide 9: Main Limitation

Main message:
- background restoration is better than before but still not fully seamless

Say clearly:
- this is the main limitation of the current lightweight method
- it does not invalidate the contribution
- it defines a strong future-work direction

## Slide 10: Future Work

Mention:
- stronger inpainting / learned restoration
- more complex scenes
- larger object motions
- broader evaluation on more samples

## Slide 11: Conclusion

Use 3 short statements:
- baseline established
- controllable extension implemented successfully
- practical results achieved with known restoration limitations

## Slide 12: Questions

Keep this slide simple.
Optional line:
- “Thank you”

## Short Viva Version

If you need a very short version, reduce it to:
1. Problem
2. Baseline limitation
3. Proposed method
4. Best results
5. Limitation and future work

## Best Supporting Files

- [backend/day9_final_results_table.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day9_final_results_table.md)
- [backend/final_progress_summary.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/final_progress_summary.md)
- [backend/day10_thesis_writing_starter.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day10_thesis_writing_starter.md)
