"""Summarize only completed addition runs; never fabricate missing results."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("results/addition")
runs = [
    json.loads((root / f"seed{s}_digits4/results.json").read_text()) for s in range(3)
]
assert len({r["source_sha256"] for r in runs}) == 1
levels = [2, 4, 8]
summary = dict(
    status="MEASURED direct addition baseline; no auxiliary transfer",
    seeds=[r["config"]["seed"] for r in runs],
    parameters=runs[0]["parameters"],
    mean_exact_match={
        str(n): float(np.mean([r["test"][str(n)]["exact_match"] for r in runs]))
        for n in levels
    },
    gate_passed=[r["validation_gate_passed"] for r in runs],
    per_seed=[
        dict(
            seed=r["config"]["seed"],
            steps=r["log"][-1]["step"],
            presentations=r["log"][-1]["presentations"],
            unique_problems=r["log"][-1]["unique_problems"],
            seconds=r["seconds"],
            test={n: r["test"][str(n)]["exact_match"] for n in levels},
        )
        for r in runs
    ],
)
(root / "summary.json").write_text(json.dumps(summary, indent=2))
plt.rcParams.update(
    {"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
)
fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for n, col in [(2, "#18594f"), (4, "#ba7953")]:
    for i, r in enumerate(runs):
        ax[0].plot(
            [v["presentations"] for v in r["log"]],
            [100 * v["validation"][str(n)] for v in r["log"]],
            color=col,
            alpha=0.7,
            label=f"{n} digits" if i == 0 else None,
        )
ax[0].set(
    xlabel="Training example presentations",
    ylabel="Whole-answer validation accuracy (%)",
    title="Direct addition learning · three seeds",
    ylim=(-2, 103),
)
ax[0].legend()
ax[0].grid(alpha=0.15)
for i, n in enumerate(levels):
    vals = [r["test"][str(n)]["exact_match"] * 100 for r in runs]
    ax[1].bar(i, np.mean(vals), color="#18594f" if n <= 4 else "#ba7953", alpha=0.6)
    ax[1].scatter(i + np.linspace(-0.12, 0.12, 3), vals, color="#152e2b", zorder=3)
ax[1].set(
    xticks=range(3),
    xticklabels=[
        "2 digits\ntrained length",
        "4 digits\ntrained length",
        "8 digits\nuntrained length",
    ],
    ylabel="Whole-answer test accuracy (%)",
    ylim=(0, 105),
    title="Held-out tests · 256 problems per length",
)
ax[1].grid(axis="y", alpha=0.15)
fig.savefig(root / "learning_curves.png", dpi=180)
fig.savefig(root / "learning_curves.pdf")
plt.close(fig)
rows = "\n".join(
    "| "
    + str(r["seed"])
    + " | "
    + " | ".join(f"{100 * r['test'][n]:.2f}%" for n in levels)
    + f" | {r['steps']:,} | {r['unique_problems']:,} |"
    for r in summary["per_seed"]
)
gates = (
    "All three seeds passed"
    if all(summary["gate_passed"])
    else "Not all three seeds passed"
)
report = f"""# Ordinary addition: measured baseline

**Two-number addition is learnable in this setup.** This study is a direct-supervision check before auxiliary-only transfer. It does not demonstrate subliminal learning.

| Seed | 2-digit test | 4-digit test | 8-digit test (untrained length) | Training updates | Unique unordered problems |
|---|---|---|---|---|---|
{rows}

Means: two digits **{100 * summary["mean_exact_match"]["2"]:.2f}%**, four digits **{100 * summary["mean_exact_match"]["4"]:.2f}%**, eight digits **{100 * summary["mean_exact_match"]["8"]:.2f}%**. All numbers are whole-answer exact match including EOS, using free-running greedy generation. Each length uses 256 unique test pairs. Three training seeds share a fixed held-out test set; they are model/data replicas, not 768 independent test problems. No population-level confidence claim is made from three seeds.

## Setup

- Small transformer trained from scratch: 4 causal layers, width 128, 4 attention heads, FF width 512, **{summary["parameters"]:,} parameters**, trainable output head.
- One token per digit. Both operands and answer are least-significant-digit first. Answer digits and EOS receive the loss; questions do not.
- Abacus-style positions align equal-significance digits and restart at each number. Offsets are randomized during training. This is an adaptation of the representation described by [McLeish et al.](https://arxiv.org/html/2405.17399v2), not a reproduction of their larger model or published accuracy.
- Both operand lengths are sampled independently from 1–4 digits during training. Tests here use two equal-length operands at each stated digit length; unequal-length evaluation is not yet reported.
- Hash-based split of unordered operand pairs: 80% train, 10% validation, 10% test. Swapped operands cannot leak across splits. Examples may repeat during training; unique problem counts are logged separately from presentations.
- AdamW, learning rate 0.0003, weight decay 0.01, batch 128, gradient clipping 1. Zero dropout.
- Training stops when >=99% validation accuracy at both 2 and 4 digits is sustained at two scheduled checkpoints, or at 8,000 updates. {gates} the sustained validation gate. Eight-digit tests are evaluated only afterward and do not affect stopping.
- Hardware: Quadro RTX 6000. Training plus evaluation times: {", ".join(f"{r['seconds']:.1f}s" for r in runs)}. These are small direct-learning runs; they do not predict the cost of a larger transfer study.

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
"""
(root / "REPORT.md").write_text(report)
print(json.dumps(summary, indent=2))
