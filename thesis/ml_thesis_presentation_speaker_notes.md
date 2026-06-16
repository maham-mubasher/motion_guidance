# Word-for-Word Speaker Script

## Slide 1: Title
Good morning respected supervisor and committee members. My name is Maham Mubasher, and today I am presenting my Thesis Two work on controllable motion-guided image editing. The main focus of my thesis is the machine learning pipeline behind object motion editing. I am using a pretrained Stable Diffusion model and controlling it at inference time, without retraining the model. The key idea is to use RAFT, which is a pretrained optical flow model, as a differentiable motion guide during the denoising process. So my work is not mainly about the frontend. My work is about how a frozen diffusion model can be guided by a motion signal.

## Slide 2: Agenda
I will start with the introduction and motivation of the problem. Then I will explain the research problem and the main objective of my thesis. After that, I will briefly discuss the machine learning models used in my work, including Stable Diffusion, CLIP, the VAE, the U-Net, and RAFT. Then I will explain the original Motion Guidance method and how my work is different from it. After that, I will present my proposed method, the fixed-seed apple ablation, the additional teapot and tree generated cases, the compact hyperparameter study, the restoration comparison, and the limitations that remain.

## Slide 3: Introduction
Generative models can now create very realistic images, but exact spatial control is still difficult. If I write a prompt like move the object to the right, the model may understand the idea semantically, but it does not know the exact displacement, the exact object mask, or how much the object should move. For image editing, this is a serious limitation because editing is not only about generating a new image. It is also about controlling geometry, preserving the object, and keeping the rest of the scene stable.

## Slide 4: Motivation
This problem is important for creative design and visual prototyping. A user may want to move an object, test a new composition, or change the position of something without recreating the complete image. From a machine learning point of view, the important question is this: can we control a pretrained generative model using another pretrained neural model as a guidance signal? In my thesis, Stable Diffusion provides the image generation prior, and RAFT provides the motion supervision.

## Slide 5: Problem Statement
The main research question of my thesis is: how can a pretrained latent diffusion model follow explicit object motion without retraining? There are four main challenges. First, the original Motion Guidance method depends on a dense flow field, which is not easy for a user to provide. Second, soft guidance can be weak when the object has to move by a large distance. Third, when an object moves, its old location becomes a missing background region. Fourth, diffusion sampling is stochastic, so even areas that should remain unchanged can drift.

## Slide 6: Goal and Objectives
The goal of my thesis is to build an inference-time machine learning control pipeline for object motion editing. The first objective is to represent user motion as dense target flow. The second objective is to guide Stable Diffusion using differentiable RAFT losses. The third objective is to preserve the content outside the edited region. The fourth objective is to repair the background after the object moves. These objectives work together because a good result should move the object, preserve its identity, and keep the scene realistic.

## Slide 7: ML Model Stack
These are the pretrained machine learning models used in my thesis. AutoencoderKL is used to encode RGB images into the Stable Diffusion latent space and decode them back to images. CLIP encodes the text prompt into a conditioning representation. The Stable Diffusion U-Net predicts the denoising update at each timestep. RAFT predicts dense optical flow between the source image and the generated image. I do not train these models. Their weights remain frozen. The variable I update is the diffusion latent during inference.

## Slide 8: Latent Diffusion
Stable Diffusion does not directly denoise the full RGB image. It works in a compressed latent space. First, the image is encoded into a latent representation using the VAE encoder. Then the U-Net gradually removes noise from this latent through DDIM denoising steps. After that, the VAE decoder converts the latent estimate back into an RGB image. This decoded image is important in my method because RAFT needs image-space input to estimate optical flow.

## Slide 9: Text Conditioning
Text conditioning is handled through CLIP and classifier-free guidance. The text prompt is encoded once by CLIP. During sampling, the U-Net predicts conditional noise and unconditional noise. These two predictions are combined using the CFG scale. In my method, RAFT guidance is added on top of this text guidance. This means the image still follows the prompt, but it also receives an additional geometric correction signal based on object motion.

## Slide 10: RAFT
RAFT is a pretrained optical flow model. It takes two images as input and predicts dense pixel-level motion between them. In my thesis, RAFT compares the source image with the current generated image and estimates how pixels have moved. This is important because RAFT is not only used after generation for evaluation. It is included inside the differentiable guidance loop, so the flow error can be backpropagated toward the diffusion latent.

## Slide 11: Original Motion Guidance
The original Motion Guidance method uses a dense target flow field to control image generation. During diffusion sampling, the current generated image is decoded and passed through RAFT. RAFT predicts the motion, and this predicted motion is compared with the target flow. The resulting loss is backpropagated to change the diffusion latent. The strength of this method is that it controls motion without training a new diffusion model. My thesis builds on this idea and extends it for a more practical editing pipeline.

## Slide 12: Research Gap
The original method is technically strong, but it has some gaps for practical object editing. It needs a dense flow field, which is difficult to create manually. It may not be strong enough for large rigid movement. Its guidance can influence areas outside the intended object. Also, when the object moves, the original location becomes empty, and the motion guidance itself cannot know what background should appear there. These gaps motivated the extensions in my thesis.

## Slide 13: Original vs Proposed
This slide shows how my work is different from the original Motion Guidance method. The original method starts with a precomputed dense flow field. In my work, I generate the target flow from simple motion primitives such as translation, scale, rotation, and stretch. I also add mask-weighted latent gradients, geometric latent initialization, latent preservation, and background restoration. So my contribution is an extended machine learning editing pipeline around Motion Guidance.

## Slide 14: Innovation Summary
There are five main additions in my proposed method. First, primitive flow generation converts simple user controls into dense RAFT supervision. Second, spatial weighting controls where the guidance gradient is stronger. Third, latent preservation protects the unedited region. Fourth, geometric initialization gives the sampler a better starting point for large movement. Fifth, restoration handles the background region that becomes visible after the object moves. Together, these additions make the motion specification more explicit and the pipeline easier to audit experimentally.

## Slide 15: Primitive to Flow
In this part, I convert simple motion parameters into dense target flow. For every pixel inside the object mask, I calculate where that pixel should move after the selected transformation. The target flow is the difference between the new position and the original position. This is important because RAFT-based guidance needs dense flow supervision, but the user should only need to provide simple controls such as move right, scale, rotate, or stretch.

## Slide 16: Guidance Energy
The guidance energy has two main losses. The first one is the flow loss. It compares the RAFT-predicted flow with the target flow, so it pushes the generated image to follow the desired motion. The second one is the color loss. It supports appearance consistency by comparing warped content with the source image. These two losses are combined using weights, and the combined energy is differentiated during the diffusion sampling process.

## Slide 17: Differentiable Path
This is the most important machine learning part of my thesis. At each DDIM step, the current noisy latent is processed by the U-Net. Then the estimated clean latent is decoded by the VAE into an image. RAFT compares this image with the source image and predicts optical flow. Then the flow loss is calculated. After that, PyTorch autograd backpropagates this loss to the diffusion latent. This is how RAFT supervises Stable Diffusion during inference without retraining either model.

## Slide 18: Spatial Guidance
Spatial guidance means that I do not apply the same gradient strength everywhere. I use a mask-based weight map. The editable object region receives stronger guidance, and the outside region receives weaker guidance. This helps the model focus on the object that should move, while reducing unnecessary changes in the background. This step is deterministic, but it improves how the machine learning guidance is applied to the latent.

## Slide 19: Latent Preservation
During diffusion sampling, even regions outside the edited object can change because the process is stochastic. To reduce this problem, I preserve source-consistent latents outside the edit support. If cached inversion latents are available, they can be reused. Otherwise, a forward-noised source latent can be used. This idea is similar to RePaint-style conditioning, where known regions are repeatedly protected during generation.

## Slide 20: Geometric Initialization
Large displacement is difficult if the sampler starts from a latent where the object is still in its original position. To handle this, I create a geometrically moved version of the image first. Then I encode this moved image using the same Stable Diffusion VAE. I add noise according to the starting DDIM level, and then I let the U-Net refine it with RAFT guidance. This gives the model a stronger starting point for rigid object motion.

## Slide 21: Guided DDIM
This slide shows the modified denoising step. First, the normal classifier-free noise prediction is computed. Then the estimated clean image is decoded. After that, RAFT predicts optical flow, and the flow and color losses are evaluated. The loss gradient is backpropagated to the latent, spatially weighted, and clipped for stability. Finally, this corrected guidance is used inside the DDIM update. This is the core algorithmic change in my thesis.

## Slide 22: Guidance Controls
These controls are used to keep inference stable. Gradient clipping prevents very large latent updates. The guidance schedule disables strong guidance near the final denoising steps, because late gradients can damage fine visual details. Recursive steps can repeat guidance at the same timestep if needed. RAFT iterations control the tradeoff between flow quality and computation time. These are inference hyperparameters, not training parameters.

## Slide 23: Restoration
Motion guidance and restoration are two different problems. RAFT can tell whether the object moved correctly, but RAFT cannot infer the hidden background behind the original object. That is why the pipeline includes a restoration stage. This restoration can use LaMa, OpenCV inpainting, or patch-based filling. I will describe the final method honestly as a hybrid ML editing pipeline, especially when deterministic compositing is used in the final output.

## Slide 24: End-to-End Workflow
The complete workflow starts with the input image, text prompt, object mask, and motion parameters. The motion parameters are converted into target flow. Then guided diffusion uses CLIP, the U-Net, the VAE, and RAFT gradients to generate the motion-aware output. After the object moves, the exposed background region is restored. The final result should include both the raw diffusion output and the final hybrid composite, so that the ML contribution can be evaluated clearly.

## Slide 25: Frontend Interface
This slide shows the web console that I implemented around the backend. It exposes the same parameters used by the experiments: prompt, input folder, mask, flow source, primitive translation values, DDIM steps, guidance weight, gradient clipping, RAFT iterations, and recursive steps. The screenshot uses the apple preset because apple is the main evaluated case. I will present this interface as implementation work and a repeatable launcher for experiments, not as a separate user-study result.

## Slide 26: Experimental Setup
For the final evaluation, I ran a controlled apple ablation. There are five variants and three random seeds, so there are fifteen runs in total. The variants are warp-inpaint-only, original Motion Guidance, diffusion without RAFT guidance, diffusion with RAFT guidance, and the full pipeline. I used 120 DDIM steps, CFG scale 7.5, guidance weight 30 for the RAFT-guided variants, gradient clipping at 60, one RAFT iteration, and seeds 0, 1, and 2. I am not claiming that these hyperparameters are globally optimal; I am using them as a fixed protocol for a reproducible case study.

## Slide 27: Evaluation Metrics
For evaluation, I used both motion-based and visual-quality metrics. Motion error and flow loss check whether the generated image follows the requested movement. The flow loss is especially important because it comes from RAFT: RAFT predicts the motion between the source and generated image, and I compare that prediction with the target flow. I also evaluate whether the edit damages parts of the image that should stay unchanged, using background preservation error. For visual quality, I use object sharpness to check whether the moved object becomes blurry, and boundary artifact score to check whether there is a visible seam around the moved object. Finally, I report runtime and GPU memory because this is an inference-time method, so computational cost matters. When I run multiple seeds, I report mean and standard deviation. The mean shows the average performance, and the standard deviation shows how stable the method is across repeated runs.

## Slide 28: ML Diagnostics
This slide is the most important evidence for the RAFT contribution. The plot shows the flow-loss trajectories saved during diffusion. When I compare diffusion without RAFT to diffusion with RAFT, the final flow loss improves from 8.46 to 5.80 on average across three seeds. The best recorded flow loss improves from 8.16 to 3.07. That is important because it shows that RAFT is not just described in the method; it measurably changes the diffusion trajectory.

## Slide 29: Apple Case Study
Here I summarize the metric table. Original Motion Guidance has the highest flow loss in this apple setup. The no-RAFT diffusion variant improves over the original baseline, mostly because it uses primitive hard-warp initialization. When RAFT guidance is added, the flow loss and total energy decrease further. The full pipeline gives the best final preservation, but that final score is achieved after compositing. So the correct interpretation is that RAFT improves the diffusion alignment, while the final visual cleanliness comes from the full hybrid pipeline.

## Slide 30: Component Analysis
This slide shows the final images across all variants and seeds. The rows are seeds 0, 1, and 2. The columns are deterministic warp-inpaint-only, original Motion Guidance, diffusion without RAFT, diffusion with RAFT, and the full pipeline. Visually, the full pipeline is the most stable. But I should also be very honest here: the deterministic warp-inpaint-only output is also strong, and both of these use final compositing. Therefore, this figure supports the full system as a hybrid editing pipeline, not a pure diffusion-only result.

## Slide 31: Additional Generated Cases
After the apple ablation, I ran two more generated cases with three seeds each. The teapot is another contained translation case, so it supports the same practical workflow as apple, although its final result also uses hybrid compositing. The tree case uses the available file-flow squeeze setup. It is useful because it shows the method under a harder deformation, but it also shows visible seed-dependent variation. I would present seed 2 as the representative tree output and describe the tree case as a stress case, not as proof that non-rigid editing is solved.

## Slide 32: Compact Hyperparameter Study
This slide addresses the criticism that the original parameters were fixed without enough justification. I ran a compact one-factor-at-a-time study on the apple case. The reference setting is guidance weight 30, one RAFT iteration, and 80 DDIM steps, and final compositing is disabled so that the measurements refer to the diffusion-stage output. The most important result is that increasing RAFT iterations from one to five reduces the final flow loss from 5.08 to 2.81 in this case. Guidance weight 10 is weaker, while guidance weight 50 is similar to 30. Reducing DDIM steps to 40 improves runtime, but preservation becomes worse. I would describe this as a sensitivity study, not as a global hyperparameter optimization.

## Slide 33: Restoration Comparison
This slide separates restoration from motion guidance. I use the same moved apple object and the same source-hole mask for all methods, so the comparison is focused on the background repair stage. Local and directional filling preserve untouched pixels, but they create visible artifacts inside the source-hole region. OpenCV inpainting is smoother numerically, but it leaves a visible fan-like pattern. LaMa gives the lowest source-hole color jump, 0.0096, and it is visually the cleanest in this example. Therefore, I use LaMa as the final selected restoration when it is available. This is important because it shows that the final visual result comes from both motion guidance and restoration, not from RAFT alone.

## Slide 34: Limitations and Next Steps
The main limitation is still scope. I now have more than one generated case, a compact hyperparameter study, and a restoration comparison, but the controlled RAFT versus no-RAFT ablation is still only one object, one background, one mask, and one translation magnitude. I can claim that RAFT improves the pre-composite diffusion trajectory in the apple setup. I can also claim that apple and teapot show stable contained translation with the full hybrid pipeline, and that LaMa is the best tested restoration method for the apple source-hole case. What I cannot claim is broad generalization, reliable non-rigid editing, or that diffusion alone created the final contained-translation images.

## Slide 35: Conclusion and Questions
To conclude, my thesis studies inference-time interaction between pretrained generative and motion models. Stable Diffusion provides the image prior. CLIP provides semantic conditioning. RAFT provides differentiable motion supervision. My contribution is extending Motion Guidance into a more controllable and auditable pipeline using primitive flow, spatial latent control, geometric initialization, latent preservation, and restoration-aware processing. The final answer is balanced: RAFT-guided diffusion helps the trajectory, and hybrid compositing makes the final contained edit clean. The additional teapot and tree runs broaden the evidence, while the apple ablation remains the central controlled result. The compact hyperparameter and restoration studies make the evaluation stronger, but they also clarify that the thesis is a focused case-study evaluation rather than a benchmark claim. Thank you. I welcome your questions and feedback.

## Likely Examiner Question: What measurable contribution does RAFT make?
The strongest answer is: RAFT guidance measurably improves the pre-composite diffusion process. Across three apple seeds, adding RAFT reduces final flow loss from 8.46 to 5.80, best flow loss from 8.16 to 3.07, and total energy from 30.57 to 21.86 compared with the no-RAFT diffusion variant. The compact hyperparameter study also shows that increasing RAFT iterations from one to five further reduces final flow loss in the apple case. However, I do not claim that RAFT alone produces the final apple image. The final image is produced by the full hybrid path, including deterministic restoration and source-object compositing.
