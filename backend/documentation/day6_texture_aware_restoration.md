# Day 6: Texture-Aware Background Restoration

Day 6 focused on improving the wooden-table reconstruction at the original apple location.

## Goal

Reduce the circular synthetic patch that remained after Day 5 by making the source-hole fill follow the local table texture more closely.

## What Changed

- `backend/background_restoration.py`
  - replaced broad multi-candidate averaging with a more directional donor-search strategy
  - fills the source hole progressively from the motion direction first
  - uses small orthogonal offsets as backups instead of mixing a wide circular neighborhood

## Mathematical Idea

Day 6 changed the restoration rule from broad averaging to directional filling.

Let `M` be the source-hole mask and `dx, dy` be the object translation.

The intuition was:
- if the apple moves right, the newly revealed table texture is most likely to resemble pixels just to the left of the hole
- if the apple moves left, the donor region should come from the right

So the restoration became:
1. choose donor offsets aligned with the motion direction
2. copy valid background values from those offsets into the missing region
3. use a few small orthogonal offsets only as backups

Conceptually:
- `I_fill(p) = I_bg(p + donor_offset)`

where the donor offset is chosen mostly along the opposite direction of motion.

This is more structured than local averaging because it tries to preserve texture orientation rather than blur it.

## Code Change Logic

In code, this happened inside `directional_background_fill(...)`:
- a sequence of candidate offsets is generated from `dx, dy`
- valid shifted background pixels are copied into the hole progressively
- if some pixels remain unresolved, the old local inpainting fallback is still available

The expected benefit was:
- better continuation of wooden-table lines,
- less circular blur,
- and a source fill that respects the main direction of the surrounding texture.

## Why This Was Tried

The Day 5 fill still produced a rounded artificial patch because it blended multiple shifted candidates together.
The Day 6 change attempted to preserve wooden-table structure more naturally by copying along the dominant motion direction.

## Image References

- Original apple:
  - [backend/data/apple/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/data/apple/pred.png)
- Best recent Day 4 result:
  - [backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png)
- Day 5 comparison result:
  - [backend/results/0e227235-4936-46c5-9ddc-ca9532f8473c/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/0e227235-4936-46c5-9ddc-ca9532f8473c/sample_000/pred.png)
- Latest Day 6 result:
  - [backend/results/0f98b444-5b8a-47c0-9e85-9beed2ba1969/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/0f98b444-5b8a-47c0-9e85-9beed2ba1969/sample_000/pred.png)

## Evaluation

What improved:
- the apple still moves correctly
- the apple itself remains visually strong
- the motion pipeline does not regress

What did not improve enough:
- the old apple location still shows a synthetic patch
- the background is still visibly reconstructed rather than natural
- the Day 6 result does not clearly beat the best Day 4 result

## Honest Conclusion

Day 6 was a reasonable technical attempt, but it did not solve the main visual problem.

The texture-aware directional fill:
- kept the motion result stable,
- but did not remove the source-hole artifact convincingly,
- and did not surpass the best Day 4 image in overall realism.

So the honest ranking is:
- **Best current visual result:** Day 4
- **Day 5:** structurally useful because it introduced a dedicated background restoration module
- **Day 6:** useful as an experiment, but not a visual improvement over Day 4

## Thesis Value

Day 6 is still useful for the thesis because it shows:
- a clear iterative improvement process,
- a tested hypothesis about texture-aware restoration,
- and an honest negative result when the method did not outperform the earlier baseline.

## Practical Decision

If no stronger inpainting / patch-based restoration method is added next, the best current demo image should remain the Day 4 result:
- [backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png)
