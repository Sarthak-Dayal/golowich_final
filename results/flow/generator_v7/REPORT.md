# Repaired flow generator

Status: **paused_for_user_inspection**.

Small shared spatial U-Net residual plus the analytic Gaussian velocity field. Spatial skip connections retain pixel-level paths; fixed mask/coordinate channels specify geometry. Global self-attention at 14×14; base channels 32.
Uses parameter EMA and a frozen, noise-trained evaluator. No teacher specialization or student distillation has run in this repair.

One-GPU process time: 427.6 seconds. Peak allocation: 2.96 GiB.

| Seed | Selected training updates | ODE steps | Accepted / 2,000 | Accepted 0–4 | Border std | Feature MMD to 0–4 / 5–9 |
|---|---:|---:|---:|---:|---:|---|

## Interpretation

- The previous generator scored 2.15% acceptance under this same noise-aware evaluator. Model architecture, optimization settings, and EMA changed together; this is a generation repair, not an ablation isolating one cause.
- All evaluation initial noises are fixed independently of training. Checkpoint and sampler selection use 512 validation noises; final scores use 2,000 distinct noises. No samples are filtered or regenerated.
- Acceptance is still an evaluator proxy. Inspect fixed-index image grids, invalid outputs, digit frequencies and diversity. Every invalid output remains in frequency denominators.
- Border pixels should remain independent unit Gaussian. Reported mean/std are marginal diagnostics and do not establish full statistical independence.
- Source/config/data hashes and RNG/model/optimizer checkpoints are retained. EMA weights are the generation checkpoints; online weights are saved separately.
- Teacher-specific transfer has not been tested by this repair.

## Ready checkpoints


Width expansion with symmetry-breaking perturbation from `results/flow/generator_v6/checkpoints/100_step36000.pt` at step 36000; lr 0.0001; optimizer reset; trained EMA and data RNG reused. Selected using validation; final noise offset 103000.

