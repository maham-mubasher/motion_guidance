# Day 10: Thesis Writing Starter

This file provides reusable draft text for thesis writing based on the implemented system and the completed experiment trail.

## Abstract Draft

This thesis presents a lightweight motion-guided image editing framework built as a controllable extension of Motion Guidance. The work begins by freezing the original Motion Guidance system as a baseline and then introduces a user-controllable editing pipeline based on motion primitives, selective refinement, hard warp initialization, and modular background restoration. The main goal is to make object motion more explicit and usable without depending only on precomputed target flow files. Experimental results show that the proposed method improves controllability and makes object translation operational in practice. In particular, hard warp initialization significantly strengthens the effect of primitive motion compared with soft guidance alone. Additional restoration strategies improve the quality of the source region after object movement, although background reconstruction remains imperfect and not fully seamless. Overall, the project demonstrates a practical and defendable lightweight motion-guided editing system while identifying source-hole restoration as the main direction for future work.

## Problem Statement Draft

Controllable image editing remains difficult when the target transformation must be expressed only through complex or precomputed intermediate representations. In the original Motion Guidance framework, motion control depends heavily on target optical flow files, which limits direct user interaction and makes practical editing less intuitive. This thesis addresses that limitation by introducing a lightweight controllable motion-editing variant in which motion can be specified through simple primitives such as translation. The aim is to retain the strengths of motion-guided editing while making the system easier to control, easier to demonstrate, and more suitable for a thesis-scale implementation.

## Objectives Draft

The objectives of this thesis are:

1. Establish the original Motion Guidance system as a reproducible baseline.
2. Introduce a controllable motion representation based on simple motion primitives.
3. Improve the effect of primitive motion through selective refinement and hard warp initialization.
4. Restore the source region after object movement as effectively as possible within a lightweight implementation.
5. Evaluate the practical strengths and limitations of the resulting system through a small set of controlled experiments.

## Methodology Draft

The proposed method extends the original Motion Guidance pipeline rather than replacing it entirely. First, the original system is preserved as a baseline and executed on a small set of stable examples. Next, motion primitives are introduced so that the desired transformation can be expressed explicitly through user-controlled parameters such as translation. Because primitive flow alone is too weak when used only as a soft guidance signal, the method then introduces hard warp initialization, in which the source image is first moved explicitly before diffusion refinement begins. Selective refinement is used to strengthen editing in the relevant region while preserving the rest of the image more conservatively. Finally, a modular background restoration stage is added to reconstruct the source region revealed after object movement. Several restoration strategies are explored, including directional filling, texture-aware filling, patch-based restoration, and OpenCV inpainting. The system is evaluated qualitatively by comparing baseline and refined outputs with respect to controllability, motion success, visual quality, and artifact behavior.

## Contribution Draft

The main contribution of this thesis is a lightweight controllable extension of Motion Guidance. This contribution consists of four main parts:

1. Motion primitives for explicit user-controlled transformations.
2. Selective refinement for more localized editing behavior.
3. Hard warp initialization for making primitive motion operational in practice.
4. A modular background restoration framework for source-hole cleanup after translation.

This contribution is practical rather than model-scale: the thesis does not claim a new large generative model, but instead proposes a clear and implementable framework for improving controllable motion-guided image editing.

## Experiments And Results Draft

The experiments were performed in a staged manner. Day 1 established the original Motion Guidance system as a baseline. Day 2 introduced motion primitives and selective refinement, but the resulting primitive motion remained too weak when used only as a soft guidance signal. Day 3 added hard warp initialization, which made primitive translation operational and enabled the apple example to move convincingly to a new position. Days 4 through 7 explored increasingly advanced background restoration strategies to improve the old object location after translation. Among these stages, Day 4 produced the strongest early visual refinement result, while later stages introduced more advanced restoration logic such as modular restoration, texture-aware filling, patch-based restoration, and OpenCV inpainting. The final practical result is strong and suitable for demonstration, although the restored source region is still not fully seamless.

## Limitation Draft

The main limitation of the proposed method is background restoration at the original object location. Although the restoration quality improves substantially compared with the early stages of development, the filled region can still appear synthetic and may preserve faint compositing artifacts. This means that the proposed framework is successful as a practical and controllable motion-editing system, but it does not yet achieve fully photorealistic source-hole reconstruction in all cases.

## Future Work Draft

Future work should focus on stronger source-hole restoration methods. In particular, more advanced inpainting or learned restoration models could improve the realism of the original object location after translation. Other natural extensions include testing on more complex scenes, supporting a larger variety of object categories, and improving robustness for larger motion magnitudes. These directions can be explored while keeping the current motion-control pipeline intact.

## Conclusion Draft

This thesis developed a lightweight and controllable motion-guided image editing framework by extending Motion Guidance with motion primitives, selective refinement, hard warp initialization, and modular background restoration. The resulting system moves beyond a baseline wrapper and becomes a practical end-to-end editing pipeline with explicit user control over object motion. The experiments show that hard warp initialization is the key step that makes primitive translation operational, while later restoration stages improve but do not fully solve source-hole realism. Overall, the work reaches a defendable thesis state: the baseline is established, the contribution is visible in code, the results are documented, and the limitations are clearly understood.

## Best Files To Reuse

- [backend/final_progress_summary.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/final_progress_summary.md)
- [backend/day9_final_results_table.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day9_final_results_table.md)
- [backend/day9_thesis_outline.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day9_thesis_outline.md)
- [backend/day3_hard_warp_init.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day3_hard_warp_init.md)
- [backend/day7_patch_based_restoration.md](/home/maham/thesis/motion_guidance_github/motion_guidance/backend/day7_patch_based_restoration.md)
