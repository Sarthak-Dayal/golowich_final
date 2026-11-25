# Repaired flow generator

Status: **stopped_generation_quality_gate**.

Small shared spatial U-Net residual plus the analytic Gaussian velocity field. Spatial skip connections retain pixel-level paths; fixed mask/coordinate channels specify geometry.
Uses parameter EMA and a frozen, noise-trained evaluator. No teacher specialization or student distillation has run in this repair.

One-GPU process time: 585.8 seconds. Peak allocation: 1.34 GiB.

| Seed | Selected training updates | ODE steps | Accepted / 2,000 | Accepted 0–4 | Border std | Feature MMD to 0–4 / 5–9 |
|---|---:|---:|---:|---:|---:|---|
| 100 | 28000 | 100 | 83.10% | 41.15% | 0.999 | 0.024597525596618652 / 0.02342391014099121 |

## Interpretation

- The previous generator scored 2.15% acceptance under this same noise-aware evaluator. Model architecture, optimization settings, and EMA changed together; this is a generation repair, not an ablation isolating one cause.
- All evaluation initial noises are fixed independently of training. Checkpoint and sampler selection use 512 validation noises; final scores use 2,000 distinct noises. No samples are filtered or regenerated.
- Acceptance is still an evaluator proxy. Inspect fixed-index image grids, invalid outputs, digit frequencies and diversity. Every invalid output remains in frequency denominators.
- Border pixels should remain independent unit Gaussian. Reported mean/std are marginal diagnostics and do not establish full statistical independence.
- Source/config/data hashes and RNG/model/optimizer checkpoints are retained. EMA weights are the generation checkpoints; online weights are saved separately.
- Teacher-specific transfer has not been tested by this repair.

## Ready checkpoints


Lower-rate continuation from `results/flow/generator_v3/checkpoints/100_step20000.pt` at step 20000; lr 3e-05; optimizer, EMA and data RNG restored. Selected using validation; final noise offset 95000.

