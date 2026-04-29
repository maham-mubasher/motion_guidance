# Day 7: Patch-Based And OpenCV Background Restoration

Day 7 focused on pushing source-hole restoration further after Day 6 did not beat the best Day 4 image.

## Goal

Reduce the synthetic patch at the original apple location and improve the realism of the wooden-table reconstruction while keeping the translated apple visually strong.

## What Changed

- `backend/background_restoration.py`
  - added a patch-based donor-selection stage
  - evaluated donor candidates using local boundary agreement
  - added donor coverage checks so weak patches are rejected
  - added local mean/std matching before patch compositing
  - restricted donor sampling to true background only
  - corrected the restoration geometry to operate on the newly exposed source region rather than the full original object region
  - added an OpenCV inpainting path using a background-only image
- `backend/generate.py`
  - adjusted final compositing masks for the source hole and destination apple region
  - reduced overly strong destination overwrite to lessen the dark outline around the apple

## Mathematical Idea

Day 7 combined several restoration ideas:

1. Patch matching on a boundary ring
   - let `M` be the exposed source-hole mask
   - let `R` be a thin ring around `M`
   - for each donor offset `(u, v)`, evaluate:
     - `L(u,v) = mean(|T_(u,v)(I_bg) - I| over R)`
   - pick the donor with the smallest boundary mismatch

2. Coverage constraint
   - reject donors that do not cover enough of the hole
   - this avoids selecting visually good but spatially incomplete donors

3. Color/statistics matching
   - align donor patch statistics to the local boundary:
     - `I_match = (I_donor - mu_donor) * (sigma_src / sigma_donor) + mu_src`

4. Geometry correction
   - restore only the newly exposed region:
     - `M_exposed = M_source * (1 - M_shifted)`
   - this is important because only the uncovered crescent should become background

5. Classical inpainting
   - OpenCV inpainting is applied on a background-only image over the original object footprint
   - the exposed source region is then blended back into the final image

## Code Change Logic

In code, Day 7 ended up as a hybrid restoration pipeline:
- handcrafted donor search and patch matching
- safer fallback restoration checks
- OpenCV inpainting for stronger hole filling
- mask-aware final compositing in `generate.py`

The key code modules are:
- `backend/background_restoration.py`
- `backend/generate.py`

## Image References

- Original apple:
  - [backend/data/apple/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/data/apple/pred.png)
- Best Day 4 comparison:
  - [backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png)
- Final practical Day 7 result:
  - [backend/results/38f19e3c-9369-4967-b97b-fae7fdd27d46/sample_000/pred.png](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/results/38f19e3c-9369-4967-b97b-fae7fdd27d46/sample_000/pred.png)

## Evaluation

What improved:
- the translated apple remains visually strong
- the motion result stays convincing
- the severe red smear / black-hole regressions were eliminated
- background restoration became much better than the earlier failed Day 7 attempts

What is still not perfect:
- the old apple location still shows a synthetic-looking filled region
- the apple can still show a faint outline from the compositing boundary
- the result is strong practically, but not fully photorealistic

## Honest Conclusion

Day 7 produced a practically strong result, but not a perfect one.

It succeeded as:
- a more advanced restoration experiment,
- a meaningful improvement over the worst Day 5/Day 6/early-Day 7 artifacts,
- and a defendable final practical method.

It did not fully solve:
- seamless background realism,
- or complete removal of compositing traces.

## Thesis Value

Day 7 is useful because it shows a serious progression:
- from simple local inpainting,
- to directional fill,
- to patch-based matching,
- to classical inpainting integration.

That is a strong thesis story even if the final result still has visible limitations.
