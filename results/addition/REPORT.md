# Ordinary addition: measured baseline

**Two-number addition is learnable in this setup.** This study is a direct-supervision check before auxiliary-only transfer. It does not demonstrate subliminal learning.

| Seed | 2-digit test | 4-digit test | 8-digit test (untrained length) | Training updates | Unique unordered problems |
|---|---|---|---|---|---|
| 0 | 100.00% | 99.61% | 1.17% | 6,000 | 394,415 |
| 1 | 100.00% | 100.00% | 14.06% | 6,000 | 394,918 |
| 2 | 100.00% | 100.00% | 47.66% | 8,000 | 493,761 |

Means: two digits **100.00%**, four digits **99.87%**, eight digits **20.96%**. All numbers are whole-answer exact match including EOS, using free-running greedy generation. Each length uses 256 unique test pairs. Three training seeds share a fixed held-out test set; they are model/data replicas, not 768 independent test problems. No population-level confidence claim is made from three seeds.

## Setup

- Small transformer trained from scratch: 4 causal layers, width 128, 4 attention heads, FF width 512, **817,679 parameters**, trainable output head.
- One token per digit. Both operands and answer are least-significant-digit first. Answer digits and EOS receive the loss; questions do not.
- Abacus-style positions align equal-significance digits and restart at each number. Offsets are randomized during training. This is an adaptation of the representation described by [McLeish et al.](https://arxiv.org/html/2405.17399v2), not a reproduction of their larger model or published accuracy.
- Both operand lengths are sampled independently from 1–4 digits during training. Tests here use two equal-length operands at each stated digit length; unequal-length evaluation is not yet reported.
- Hash-based split of unordered operand pairs: 80% train, 10% validation, 10% test. Swapped operands cannot leak across splits. Examples may repeat during training; unique problem counts are logged separately from presentations.
- AdamW, learning rate 0.0003, weight decay 0.01, batch 128, gradient clipping 1. Zero dropout.
- Training stops when >=99% validation accuracy at both 2 and 4 digits is sustained at two scheduled checkpoints, or at 8,000 updates. All three seeds passed the sustained validation gate. Eight-digit tests are evaluated only afterward and do not affect stopping.
- Hardware: Quadro RTX 6000. Training plus evaluation times: 130.7s, 131.2s, 179.4s. These are small direct-learning runs; they do not predict the cost of a larger transfer study.

## What this establishes

We now have a working ordinary arithmetic task without a secret mapping. Increasing digit count supplies a problem-size scaling axis. The eight-digit test measures extrapolation beyond the training lengths, not learning efficiency at eight digits. Nothing here proves a minimum transformer depth or a computation-scaling law.

## Next experiment, in order

1. Freeze this baseline and its test splits. Add carry-stratified and unequal-length evaluation before choosing further settings.
2. Run the first controlled auxiliary-only transfer comparison at a trained length, retaining neutral and shuffled controls plus direct distillation. Calibrate the shared output channel/readout so it can transmit answer digits.
3. Separately train on up to eight digits and repeat. Distinguish larger training support from untrained-length extrapolation.
4. After two-operand learning and transfer are stable, add operand count using the scratchpad/position-coupling recipe from [Cho et al.](https://arxiv.org/html/2410.15787v2).

For efficiency curves, vary unique data and repeated presentations separately, and log tokens/updates/compute as well as exact-match accuracy. Scratchpad tokens must be counted if introduced. Do not compare sequence counts alone across digit lengths.

## Artifacts

- `seed0_digits4/`, `seed1_digits4/`, `seed2_digits4/`: raw metrics, per-run configuration, example predictions and exact source snapshots.
- `learning_curves.png` / `.pdf`: validation learning curves and held-out test results.
- `summary.json`: machine-readable result table.
- `experiments/addition/run.py`: reproducible training/evaluation.
- `experiments/addition/check_task.py`: arithmetic oracle, answer-only loss mask, position alignment, swap-safe splits, and causality checks.

Checkpoints are saved locally under ignored `checkpoints/` directories. The preceding composition and flow artifacts were not modified. No site deployment was performed for this baseline.

## First larger-size pilot

Training the same model from scratch on up to eight digits reached **99.61%** eight-digit held-out accuracy in one exploratory seed. See [SCALE_REPORT.md](SCALE_REPORT.md) for the separate results and limits. This is successful learning at a larger trained problem size, not a claim of universal extrapolation.
