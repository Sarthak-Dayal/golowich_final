# Repaired flow generator

Status: **stopped_shared_mlp_quality_gate**.

Shared MLP residual plus a full-rank time-conditioned pixel path.
Uses parameter EMA and a frozen, noise-trained evaluator. No teacher specialization or student distillation has run in this repair.

One-GPU process time: 348.2 seconds. Peak allocation: 0.22 GiB.

| Seed | Selected training updates | ODE steps | Accepted / 2,000 | Accepted 0–4 | Border std | Feature MMD to 0–4 / 5–9 |
|---|---:|---:|---:|---:|---:|---|
| 100 | 30000 | 100 | 61.35% | 32.05% | 1.000 | 0.02373373508453369 / 0.026563286781311035 |

## Interpretation

- The previous generator scored 2.15% acceptance under this same noise-aware evaluator. Model architecture, optimization settings, and EMA changed together; this is a generation repair, not an ablation isolating one cause.
- All evaluation initial noises are fixed independently of training. Checkpoint and sampler selection use 512 validation noises; final scores use 2,000 distinct noises. No samples are filtered or regenerated.
- Acceptance is still an evaluator proxy. Inspect fixed-index image grids, invalid outputs, digit frequencies and diversity. Every invalid output remains in frequency denominators.
- Border pixels should remain independent unit Gaussian. Reported mean/std are marginal diagnostics and do not establish full statistical independence.
- Source/config/data hashes and RNG/model/optimizer checkpoints are retained. EMA weights are the generation checkpoints; online weights are saved separately.
- Teacher-specific transfer has not been tested by this repair.

## Ready checkpoints

