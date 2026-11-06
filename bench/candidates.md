# Candidates

baseline profile (batch 512, aux on): step 0.277s = fwd/bwd 0.208 + Adam 0.059; gathers 0.052; display 12.9ms/frame. Saved at bench/profiles/step_baseline.json.

| candidate | area | kind | status | before | after | p | raw |
|---|---|---|---|---|---|---|---|
| sparse-x-update | train step | structural | reverted | 0.277 s/step | MPS in-place indexed ops hang (killed after 10 min) | - | bench/runs/20261106-235445-sparse-x-update.jsonl |
| cpu-train-step | train step | structural | reverted | 36.0 s | 64.7 s (0.556x) | 0.017 | bench/runs/20261107-003405-cpu-train-step.jsonl |
| gpu-display | snapshot render | structural | kept | 3.29 s / 256 frames | 2.42 s / 256 frames (1.36x) | 0.066 | bench/runs/20261107-010312-gpu-display.jsonl |

Dropped without bench: bf16-targets (bounded <=18% gather saving, precision risk on the bilinear that defines the artwork); batch-1024 (bounded <=17%, worth trying only if step time still dominates after the kept change). Sparse-x remains viable on CUDA where indexed ops are fast; on MPS it hangs.
