# Repaired flow generator

Status: **stopped_generation_quality_gate**.

Small shared spatial U-Net residual plus the analytic Gaussian velocity field. Spatial skip connections retain pixel-level paths; fixed mask/coordinate channels specify geometry. Global self-attention at 14×14; base channels 16.
Uses parameter EMA and a frozen, noise-trained evaluator. No teacher specialization or student distillation has run in this repair.

One-GPU process time: 607.9 seconds. Peak allocation: 1.45 GiB.

| Seed | Selected training updates | ODE steps | Accepted / 2,000 | Accepted 0–4 | Border std | Feature MMD to 0–4 / 5–9 |
|---|---:|---:|---:|---:|---:|---|
| 100 | 36000 | 100 | 82.55% | 40.40% | 0.999 | 0.02619469165802002 / 0.021616697311401367 |

## Interpretation

- The previous generator scored 2.15% acceptance under this same noise-aware evaluator. Model architecture, optimization settings, and EMA changed together; this is a generation repair, not an ablation isolating one cause.
- All evaluation initial noises are fixed independently of training. Checkpoint and sampler selection use 512 validation noises; final scores use 2,000 distinct noises. No samples are filtered or regenerated.
- Acceptance is still an evaluator proxy. Inspect fixed-index image grids, invalid outputs, digit frequencies and diversity. Every invalid output remains in frequency denominators.
- Border pixels should remain independent unit Gaussian. Reported mean/std are marginal diagnostics and do not establish full statistical independence.
- Source/config/data hashes and RNG/model/optimizer checkpoints are retained. EMA weights are the generation checkpoints; online weights are saved separately.
- Teacher-specific transfer has not been tested by this repair.

## Ready checkpoints


Global-attention architecture extension from `results/flow/generator_v4/checkpoints/100_step28000.pt` at step 28000; lr 0.0001; optimizer reset; trained EMA and data RNG reused. Selected using validation; final noise offset 99000.

