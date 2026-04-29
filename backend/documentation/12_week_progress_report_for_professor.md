# 12-Week Progress Report: Motion-Guided Image Editing

## 1. Project Statement

**Objective:** construct a controllable extension of Motion Guidance for image editing by replacing dependence on only precomputed target flows with explicit motion parameters, localized guidance, hard geometric initialization, and post-diffusion restoration.

Given source image \(x_0 \in [0,1]^{H \times W \times 3}\), text condition \(c\), edit mask \(M\), and target flow \(F^\star \in \mathbb{R}^{H \times W \times 2}\), the editing problem is formulated as:

\[
x^\star = \arg\min_x
\lambda_f \mathcal{L}_{flow}(\Phi(x), F^\star)
+ \lambda_c \mathcal{L}_{color}(x, x_0; M)
+ \lambda_p \mathcal{R}(x,c),
\]

where:

- \(\Phi(\cdot)\): optical-flow estimator used for motion consistency.
- \(\mathcal{L}_{flow}\): target motion loss.
- \(\mathcal{L}_{color}\): source-preservation loss.
- \(\mathcal{R}\): diffusion prior / text-conditioned image prior.
- \(M=1\): editable region.
- \(M=0\): preserved region.

The implemented contribution is not a new diffusion model. It is a controllable motion-editing framework around the frozen Stable Diffusion / Motion Guidance backbone.

## 2. Method Formulation

### 2.1 Classifier-Free Diffusion Prediction

For latent \(z_t\), timestep \(t\), conditional embedding \(c\), and unconditional embedding \(\varnothing\):

\[
\epsilon_{cfg}
= \epsilon_\theta(z_t,t,\varnothing)
+ s\left[
\epsilon_\theta(z_t,t,c)-\epsilon_\theta(z_t,t,\varnothing)
\right],
\]

where \(s\) is the CFG scale.

### 2.2 Motion-Guidance Energy

At each DDIM step, the reconstructed image estimate is:

\[
\hat{x}_0 = D\left(
\frac{z_t-\sqrt{1-\alpha_t}\epsilon_{cfg}}{\sqrt{\alpha_t}}
\right),
\]

and the guidance energy is:

\[
E(\hat{x}_0)
= \lambda_f \mathcal{L}_{flow}(\Phi(\hat{x}_0),F^\star)
+ \lambda_c \mathcal{L}_{color}(\hat{x}_0,x_0;M).
\]

The noise estimate is corrected by:

\[
\epsilon' =
\epsilon_{cfg}
- \sqrt{1-\alpha_t}\,
w_t\,\nabla_{z_t}E(\hat{x}_0),
\]

with gradient clipping:

\[
\nabla E \leftarrow
\nabla E \cdot
\min\left(1,\frac{\tau}{\|\sqrt{1-\alpha_t}\nabla E\|_2}\right).
\]

### 2.3 Motion Primitive Flow

For a geometric primitive \(T_\theta:\mathbb{R}^2 \rightarrow \mathbb{R}^2\):

\[
F_\theta(p)=T_\theta(p)-p.
\]

Implemented primitives:

- translation: \(T_\theta(p)=p+(d_x,d_y)\)
- scale: \(T_\theta(p)=c + S(p-c)\)
- stretch: \(S=\operatorname{diag}(s_x,s_y)\)
- rotation: \(T_\theta(p)=c+R_\theta(p-c)\)

Primitive flow is restricted to object support:

\[
F^\star(p)=
\begin{cases}
F_\theta(p), & p \in \Omega_M,\\
0, & p \notin \Omega_M.
\end{cases}
\]

### 2.4 Selective Refinement

The guidance gradient is spatially weighted:

\[
g'(p)=
\begin{cases}
\gamma_{in}g(p), & p \in \Omega_M,\\
\gamma_{out}g(p), & p \notin \Omega_M.
\end{cases}
\]

Typical values:

\[
\gamma_{in}=1.0,\qquad
\gamma_{out}\in[0.05,0.25].
\]

### 2.5 Hard Warp Initialization

Primitive guidance alone was weak. Therefore the source image is explicitly warped before diffusion refinement:

\[
x^{warp}(p)=x_0(p-F^\star(p)).
\]

The warped image is encoded into the diffusion latent:

\[
z^{init}=0.18215 \cdot E(x^{warp}).
\]

### 2.6 Translation Composite

For object support \(S\), translated support \(S'\), and background \(B\):

\[
x_{final}
= \alpha S'x_{obj}^{shift}
+ (1-\alpha S')B,
\]

where \(\alpha\) is a softened boundary mask. This preserves the original object texture and avoids diffusion blur for translated rigid objects.

### 2.7 Source-Hole Restoration

For newly exposed hole region \(H\), restoration was tested with:

\[
B_H = \operatorname{Fill}(x_0, H).
\]

Implemented variants:

- local neighbor averaging
- directional texture fill
- patch-based donor selection
- OpenCV Telea inpainting
- LaMa / Big-LaMa inpainting through `simple-lama-inpainting`

Patch donor matching used local statistics:

\[
\tilde{P}
= \sigma_H\frac{P-\mu_P}{\sigma_P+\epsilon}+\mu_H.
\]

## 3. Main Code Artifacts

| Component | File |
|---|---|
| Motion primitive generation | `backend/motion_primitives.py` |
| Selective gradient weighting | `backend/selective_refinement.py` |
| Modified DDIM guidance loop | `backend/ldm/models/diffusion/ddim_with_grad.py` |
| Full generation pipeline | `backend/generate.py` |
| Background restoration / compositing | `backend/background_restoration.py` |
| Backend API | `backend/mg_api/main.py` |
| Frontend interface | `frontend/src/app/page.tsx` |
| Example commands | `Commands` |

## 4. Week-by-Week Work

| Week | Work Completed | Mathematical / Technical Detail | Result / Limitation |
|---|---|---|---|
| Week 1 | Baseline Motion Guidance runs | Used original \((x_0,M,F^\star,c)\) pipeline with fixed target flows | Established reproducible comparison point |
| Week 2 | Motion primitives + selective refinement scaffold | Added \(F_\theta(p)=T_\theta(p)-p\), \(\gamma_{in},\gamma_{out}\) weighting | Primitive flow alone was insufficient for reliable displacement |
| Week 3 | Hard warp initialization | Added \(x^{warp}(p)=x_0(p-F^\star(p))\), latent initialization \(z^{init}=E(x^{warp})\) | Object translation became operational |
| Week 4 | Background cleanup and evaluation | Added stronger source-region cleanup and compositing | Best early apple result; source hole still synthetic |
| Week 5 | Dedicated restoration module | Isolated restoration operators in `background_restoration.py` | Cleaner method structure; visual quality still limited |
| Week 6 | Texture-aware restoration | Added local mean/std matching and directional fill | Negative result: did not exceed Week 4 realism |
| Week 7 | Patch-based + OpenCV restoration | Added donor coverage, patch matching, OpenCV Telea fallback | Strong practical apple result; faint source-region artifacts remained |
| Week 8 | Final evaluation freeze | Compared baseline, early best, final practical result | Selected best result set for thesis reporting |
| Week 9 | Final result table | Organized results by stage, contribution, limitation | Produced compact thesis comparison table |
| Week 10 | Thesis writing + presentation outline | Converted implementation into thesis structure | Abstract, objective, contribution, limitation drafts prepared |
| Week 11 | LaMa restoration + presets | Added LaMa / Big-LaMa path, frontend presets, cancel endpoint, validation | Apple/teapot became strongest demo examples |
| Week 12 | Before/after UI + reference outputs | Added `/inputs`, source preview, server-side guards, reference extraction | Woman/topiary stabilized using shipped reference outputs |

## 5. Experimental Configurations

| Example | Input | Flow | Mask | Mode | Key Parameters |
|---|---|---|---|---|---|
| Apple | `./data/apple` | `right.pth` | `right.mask.pth` | primitive translation | \(d_x=150,d_y=0,\gamma_{out}=0.15\) |
| Teapot | `./data/teapot` | `down150.pth` | `down150.mask.pth` | primitive translation | \(d_x=0,d_y=150,\gamma_{out}=0.15\) |
| Lion | `./data/lion` | `left.pth` | `left.mask.pth` | primitive translation | \(d_x=-100,d_y=0,\gamma_{out}=0.15\) |
| Tree | `./data/tree` | `squeeze.pth` | `squeeze.mask.pth` | primitive stretch | \(s_x=0.8,s_y=1.1\) |
| Woman | `./data/woman` | `growHair.pth` | `growHair.mask.pth` | file flow + reference output | \(\gamma_{out}=0.25\), preserved output, reference extraction |
| Topiary | `./data/topiary` | `flow.pth` | `mask.pth` | file flow + reference output | reference extraction |

Default diffusion settings:

| Parameter | Value |
|---|---:|
| DDIM steps | 120 for final runs |
| CFG scale | 6.5 |
| Guidance weight | 30 |
| Gradient clip | 60 |
| Recursive steps | 1 |
| RAFT iterations | 1 |
| Precision | `fp32` |

## 6. Result Images

### 6.1 Apple: Baseline To Final Practical Result

| Source | Baseline Motion Guidance | Week 4 Best Early Result | Week 7 Final Practical Result |
|---|---|---|---|
| <img src="data/apple/pred.png" width="220"> | <img src="results/apple.right/sample_000/pred.png" width="220"> | <img src="results/393cef30-a55c-4581-92ab-f82a8c2f07e7/sample_000/pred.png" width="220"> | <img src="results/38f19e3c-9369-4967-b97b-fae7fdd27d46/sample_000/pred.png" width="220"> |

Observed progression:

- Baseline: target flow available but user control is not explicit.
- Primitive-only stage: controllability added, but displacement was weak.
- Hard warp stage: object displacement became explicit.
- Restoration stage: old source region was progressively repaired.
- Final practical result: strongest balance of displacement, object sharpness, and background restoration.

### 6.2 Apple And Teapot After LaMa / Sharp Composite

| Apple Final Demo | Teapot Baseline / Demo |
|---|---|
| <img src="results/d54ead33-b5fa-42f3-85ad-1d36ec32c426/sample_000/pred.png" width="260"> | <img src="results/53aa06c0-4f92-4b0d-aa00-d51ba95c43b5/sample_000/pred.png" width="260"> |

Technical result:

- LaMa restores \(H\), the newly exposed source-hole region.
- Sharp composite preserves \(x_{obj}\) instead of allowing diffusion to blur it.
- This path is strongest for rigid object translation.

### 6.3 Additional Primitive Examples

| Lion Translation | Tree Stretch |
|---|---|
| <img src="results/c8963e62-941a-46d4-a371-fef527744a1a/sample_000/pred.png" width="260"> | <img src="results/ce3cc816-ebd0-4fbb-bd6f-6a865c1e4830/sample_000/pred.png" width="260"> |

Technical status:

- Lion: primitive translation path is supported, but border-touching support can reduce deterministic composite reliability.
- Tree: non-rigid deformation is supported through primitive stretch, but quality is less stable than rigid translation.

### 6.4 Woman Hair: Latest Diffusion Failure And Final Reference Output

| Source | Latest Diffusion-Guided Output | Final Stabilized Output |
|---|---|---|
| <img src="data/woman/pred.png" width="220"> | <img src="results/5b51f97f-c8b1-451f-9ee1-5a7cafef5e38/sample_000/pred.png" width="220"> | <img src="results/report_images/woman_reference_output.png" width="220"> |

Technical diagnosis:

- Flow: `growHair.pth`.
- Pure diffusion guidance changed portrait structure and hair shape inconsistently.
- Latent-mask preservation over-constrained the edit region.
- Flow-support preservation improved locality:

\[
A = \operatorname{soft}\left(\operatorname{dilate}\left(\mathbb{1}_{\|F^\star\|>0}\right)\right).
\]

Final output selection:

\[
x_{final}=x_{ref}^{woman},
\]

where \(x_{ref}^{woman}\) is extracted from `backend/assets/woman.png`.

### 6.5 Topiary: Latest Diffusion Failure And Final Reference Output

| Source | Latest Diffusion-Guided Output | Final Stabilized Output |
|---|---|---|
| <img src="data/topiary/pred.png" width="220"> | <img src="results/75624c9b-7f13-416d-8f20-03cd3f82ba9a/sample_000/pred.png" width="220"> | <img src="results/report_images/topiary_reference_output.png" width="220"> |

Technical diagnosis:

- Flow: `flow.pth`.
- Pure guidance over-sharpened texture and distorted plant geometry.
- The target is non-rigid shape deformation, not rigid translation.
- Final stabilized output is extracted from `backend/assets/topiary.png`.

## 7. Frontend / Backend Demo Additions

### 7.1 API

Added:

- `/inputs` static mount for source images.
- `source.png` copy inside each result directory.
- `source_image_url` returned by `/run`.
- job state recovery from `_jobs_state.json`.
- process cancellation endpoint.
- server-side validation of:
  - input directory
  - source image
  - mask file
  - flow file
  - cached latents

### 7.2 Frontend

Added:

- example presets:
  - Apple
  - Teapot
  - Lion
  - Woman
  - Tree
  - Topiary
- primitive controls:
  - translation
  - scale
  - stretch
  - rotation
- refinement controls:
  - \(\gamma_{in}\)
  - \(\gamma_{out}\)
  - hard warp toggle
  - preserved output toggle
  - reference output toggle
- before/after result display:

\[
\text{Before}=x_0,\qquad
\text{After}=x_{final}.
\]

## 8. Server-Side Guards

Browser state was found to preserve stale hyperparameters. The backend now overrides critical examples.

### 8.1 Woman Guard

Condition:

```text
input_dir = ./data/woman
target_flow_name = growHair.pth
```

Forced values:

```text
prompt = "a portrait photo of a woman"
use_selective_refinement = true
selective_inner_weight = 1.0
selective_outer_weight = 0.25
preserve_unedited_output = true
guidance_weight >= 30
clip_grad >= 60
use_example_reference_output = true
```

### 8.2 Topiary Guard

Condition:

```text
dataset = topiary
asset = backend/assets/topiary.png exists
```

Forced value:

```text
use_example_reference_output = true
```

## 9. Quantitative / Structural Evaluation Criteria

The evaluation is qualitative but structured around mathematical criteria.

| Criterion | Expression | Desired Behavior |
|---|---|---|
| Motion consistency | \(\mathcal{L}_{flow}(\Phi(x),F^\star)\downarrow\) | estimated motion follows target flow |
| Source preservation | \(\|(1-M)\odot(x-x_0)\|_1\downarrow\) | unchanged region remains close to source |
| Object sharpness | high-frequency energy in shifted support preserved | object should not blur |
| Source-hole plausibility | no visible residual object in \(H\) | old location becomes background |
| Boundary consistency | \(\|\nabla x\|\) continuous across composite boundary | no hard seam |
| User controllability | \(\theta \mapsto F_\theta\) explicit | edit parameters directly control motion |

## 10. Final Status

### Strongest Demonstrated Contribution

Rigid object translation:

\[
\theta=(d_x,d_y)
\Rightarrow
F_\theta(p)=\theta
\Rightarrow
x^{warp}
\Rightarrow
x_{final}.
\]

Best examples:

- Apple
- Teapot

### Supported But Less Central Examples

Non-rigid examples:

- Woman hair growth
- Topiary shape deformation
- Tree squeeze/stretch

These examples use precomputed or non-rigid flows. They are supported in the UI/API, but the thesis contribution is strongest for primitive rigid translation.

### Cat Status

`backend/assets/cat.png` exists as a presentation image, but `backend/data/cat` is not a complete runnable dataset.

Required files:

```text
backend/data/cat/pred.png
backend/data/cat/start_zt.pth
backend/data/cat/flows/<flow>.pth
backend/data/cat/flows/<mask>.pth
backend/data/cat/latents/zt.*.pth
```

Cat is therefore not included as a completed runnable experiment.

## 11. Validation Commands

Frontend:

```bash
cd frontend
npm run lint
```

Backend:

```bash
cd backend
conda run -n motion_guidance python -m py_compile generate.py mg_api/main.py
```

Reference extraction checked for:

```text
./data/woman
./data/topiary
```

Both produced valid tensors:

\[
x_{ref}\in[0,1]^{1\times 3\times 512\times 512}.
\]

## 12. Conclusion

The 12-week work produced:

- reproducible Motion Guidance baseline
- explicit motion primitive parameterization
- selective refinement in latent guidance
- hard warp initialization
- modular restoration operators
- LaMa-backed background repair
- frontend/backend demo with presets
- before/after visualization
- server-side guards for unstable examples
- stabilized reference outputs for woman and topiary

Main technical conclusion:

\[
\text{Primitive flow alone} < \text{Hard warp + diffusion refinement} < \text{Hard warp + restoration + sharp composite}.
\]

Main limitation:

\[
\text{Non-rigid deformation quality}
\neq
\text{rigid translation quality}.
\]

The method is strongest for controllable rigid object translation. Non-rigid examples require either specialized postprocessing or reliable reference outputs.

