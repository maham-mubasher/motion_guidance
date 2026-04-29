# Day 1 Baseline Results

This file freezes the current Motion Guidance system as the thesis baseline.

Baseline definition:
- Original Motion Guidance backend
- Current API and frontend wrapper
- No thesis-specific motion primitive generation yet
- No selective refinement module integrated yet

## Completed baseline runs

| Example | Prompt | Input Dir | Mask | Flow | DDIM | Guidance | Clip Grad | RAFT | Recursive | Scale | Log Freq | Output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Teapot | `a teapot floating in water` | `./data/teapot` | `down150.mask.pth` | `down150.pth` | 40 | 30.0 | 60.0 | 1 | 1 | 7.5 | 25 | `backend/results/teapot.down150/sample_000/pred.png` |
| Apple | `an apple on a wooden table` | `./data/apple` | `right.mask.pth` | `right.pth` | 40 | 30.0 | 60.0 | 1 | 1 | 7.5 | 5 | `backend/results/apple.right/sample_000/pred.png` |
| Topiary | `a photo of topiary` | `./data/topiary` | `mask.pth` | `flow.pth` | 40 | 30.0 | 60.0 | 1 | 1 | 7.5 | 5 | `backend/results/topiary/sample_000/pred.png` |

## Baseline notes to fill in

Use this table after visually reviewing the saved outputs.

| Example | Runtime | Visual quality | Motion success | Artifacts / failure modes |
| --- | --- | --- | --- | --- |
| Teapot | TODO | TODO | TODO | TODO |
| Apple | TODO | TODO | TODO | TODO |
| Topiary | TODO | TODO | TODO | TODO |

## Why this completes Day 1

Day 1 was defined as:
- confirm the pipeline runs end to end,
- freeze a small baseline set,
- record the settings used,
- prepare the repo for a thesis contribution on the backend side.

The baseline generation is now complete. The next step is to compare this baseline against a small method contribution built around motion primitives and selective refinement.
