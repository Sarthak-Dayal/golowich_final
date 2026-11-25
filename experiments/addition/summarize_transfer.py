"""Summarize completed addition-transfer pilots and diagnostics."""

import csv, json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("results/addition_transfer")
runs = [
    json.loads((root / f"pilot_seed{s}/results.json").read_text()) for s in range(3)
]
assert all(r.get("teacher_unchanged") and len(r["arms"]) == 4 for r in runs)
arms = ["transfer", "neutral", "shuffled", "direct"]
colors = ["#18594f", "#85938a", "#b0914d", "#ba7953"]
names = ["Auxiliary only", "Neutral teacher", "Shuffled outputs", "Direct distillation"]
diags = [
    json.loads((root / f"pilot_seed{s}/interface_diagnostic.json").read_text())
    for s in range(3)
]
summary = dict(
    status="MEASURED pilot, three existing baseline seed pairs, fixed data and varied exposures",
    seeds=[r["seed"] for r in runs],
    unique_examples=16384,
    exposures=524288,
    means={
        a: float(np.mean([r["arms"][a]["final_test"]["exact_match"] for r in runs]))
        for a in arms
    },
    teacher_mean=float(np.mean([d["arms"]["teacher"]["native"] for d in diags])),
    heldout_auxiliary_kl_mean=float(
        np.mean([d["arms"]["transfer"]["heldout_auxiliary_kl"] for d in diags])
    ),
    per_seed=[
        dict(
            seed=r["seed"],
            **{a: r["arms"][a]["final_test"]["exact_match"] for a in arms},
        )
        for r in runs
    ],
)
(root / "summary.json").write_text(json.dumps(summary, indent=2))
with (root / "curves.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "seed",
            "arm",
            "unique_examples",
            "exposures",
            "updates",
            "distilled_positions",
            "input_tokens",
            "validation_exact_match",
            "loss",
        ]
    )
    for r in runs:
        for a in arms:
            for v in r["arms"][a]["curve"]:
                w.writerow(
                    [r["seed"], a]
                    + [
                        v[k]
                        for k in [
                            "unique_examples",
                            "exposures",
                            "updates",
                            "distilled_positions",
                            "input_tokens",
                            "validation_exact_match",
                            "loss",
                        ]
                    ]
                )
plt.rcParams.update(
    {"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
)
fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for a, col, name in zip(arms, colors, names):
    vals = np.array(
        [[v["validation_exact_match"] for v in r["arms"][a]["curve"]] for r in runs]
    )
    xs = [v["exposures"] for v in runs[0]["arms"][a]["curve"]]
    for row in vals:
        ax[0].plot(xs, row * 100, color=col, alpha=0.22, lw=1)
    ax[0].plot(xs, vals.mean(0) * 100, "o-", color=col, label=name)
ax[0].set(
    xscale="log",
    xlabel="Example presentations (16,384 unique examples)",
    ylabel="Whole-answer validation accuracy (%)",
    ylim=(-2, 103),
    title="Four-digit addition · three seed pairs",
)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.15)
for a, col, name in zip(arms, colors, names):
    vals = np.array([[v["loss"] for v in r["arms"][a]["curve"]] for r in runs])
    ax[1].plot(xs, vals.mean(0), "o-", color=col, label=name)
ax[1].set(
    xscale="log",
    yscale="log",
    xlabel="Example presentations",
    ylabel="Last-batch KL per distilled position (nats)",
    title="Fitting the carrier is not arithmetic transfer",
)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.15)
fig.savefig(root / "learning_curves.png", dpi=180)
fig.savefig(root / "learning_curves.pdf")
plt.close(fig)
rows = "\n".join(
    "| "
    + str(r["seed"])
    + " | "
    + " | ".join(
        f"{100 * r['arms'][a]['final_test']['exact_match']:.2f}%" for a in arms
    )
    + " |"
    for r in runs
)
drows = "\n".join(
    "| "
    + str(i)
    + " | "
    + " | ".join(
        f"{100 * d['arms'][a]['teacher_operators_and_head']:.2f}%"
        for a in ["base", "transfer", "neutral"]
    )
    + " |"
    for i, d in enumerate(diags)
)
aux_token = (
    np.mean(
        [
            d["arms"]["transfer"]["arithmetic_teacher_forced_token_accuracy"]
            for d in diags
        ]
    )
    * 100
)
neutral_token = (
    np.mean(
        [
            d["arms"]["neutral"]["arithmetic_teacher_forced_token_accuracy"]
            for d in diags
        ]
    )
    * 100
)
base_operator = np.mean([d["arms"]["teacher"]["base_operators"] for d in diags]) * 100
report = f"""# First addition transfer experiment: measured results

This pilot keeps the four-digit task fixed and tests ordinary token-output distillation on random digit strings. It uses three existing teacher/base initialization pairs. The settings are not a fresh confirmatory cohort selected independently of the baseline study.

## Primary result

| Seed | Auxiliary-only | Neutral teacher | Shuffled outputs | Direct target distillation |
|---|---|---|---|---|
{rows}

At the fixed final budget of **16,384 unique examples and 524,288 presentations (32 passes)**, mean exact-match accuracies are auxiliary-only **{100 * summary["means"]["transfer"]:.2f}%**, neutral **{100 * summary["means"]["neutral"]:.2f}%**, shuffled **{100 * summary["means"]["shuffled"]:.2f}%**, and direct **{100 * summary["means"]["direct"]:.2f}%**. Mean teacher accuracy on the same final evaluation set is **{100 * summary["teacher_mean"]:.2f}%**.

Final tests contain 256 held-out four-digit problems per seed, shared across seeds. Whole-answer accuracy includes EOS and uses greedy generation. The primary endpoint is fixed at the final exposure budget, not selected by the best validation checkpoint. All three paired auxiliary-minus-neutral effects are zero in this sample. This does not establish an exactly zero population effect or impossibility of transfer; the test is finite and there are only three training replicas. No Gaussian interval or sample-complexity ratio is claimed from these zero endpoints.

## What was trained

- Students begin at the exact random initializations of their teachers. The 817,679-parameter causal addition transformer and ordinary trainable token head are unchanged.
- Auxiliary inputs: BOS followed by 15 random digits. PLUS/EQUAL, arithmetic prompts, and oracle sums are excluded. Match the teacher's output distributions at the 15 digit positions. Outputs are soft logits over the original 15-token vocabulary, not hard sampled text.
- Neutral: match untrained-base outputs on the same inputs. Shuffled: permute teacher output sequences across inputs, preserving positional marginal distributions.
- Direct: match teacher outputs on actual addition examples (operand lengths up to four) with correct answer prefixes; loss only on answer digits/EOS. This is intentionally explicit task supervision. Unique-pair rejection reweights short versus long operand combinations compared with the original streaming teacher data.
- All arms: AdamW 0.0003, zero weight decay, batch 128, gradient clipping 1, no dropout. KL is averaged over supervised positions. The same cached input pool is revisited with reshuffled order each pass.
- Exposure budgets: 4,096; 16,384; 65,536; 262,144; 524,288. The unique-data budget stops growing after one pass. This is an exposure/optimization experiment, not a completed unique-data sweep.
- Four-digit validation (128 problems) supplies the plotted curves. The disjoint final test is evaluated after each arm reaches the fixed budget. Hash splits prevent swapped operands crossing partitions. Validation and test sampling use independent random streams.
- All forward batches have padded length 16. Distilled positions differ: auxiliary examples supply 15 distributions; direct examples supply fewer answer distributions. Raw per-position and unpadded-token counts are in curves.csv. Padded forward-token count at the endpoint is 8,388,608 per arm. Equal example counts are not equal supervision or transmitted information.
- All teacher parameter hashes remained unchanged. Source snapshots, configs, student checkpoints and raw metrics are retained in each pilot directory.

## Distillation fit versus target competence

Mean KL to the teacher on 1,024 fresh auxiliary sequences is **{summary["heldout_auxiliary_kl_mean"]:.4f} nats per position** for the auxiliary students. Training KL falls substantially, but matching this carrier does not produce complete correct sums. Inspect these domain-specific losses separately from arithmetic exact match; low auxiliary loss is not evidence that the addition algorithm transferred.

Neutral targets represent the base itself, but finite-precision cached/no-grad versus gradient-enabled forwards can cause small initial gradients and parameter drift under Adam. Parameter displacement and neutral KL are logged. Native arithmetic accuracy remains the relevant matched negative control; the neutral arm should not be described as parameter-identical to the base.

## Teacher-assisted interface diagnostic

The following table restores both teacher operator-token embeddings (BOS, PLUS, EQUAL) and teacher output-head weights. These are privileged teacher parameters and are not autonomous transfer results.

| Seed | Restored base | Restored auxiliary student | Restored neutral student |
|---|---|---|---|
{drows}

The complete diagnostic also tests operator-only and head-only restoration, teacher-forced arithmetic token accuracy/KL, and teacher evaluation with base operator embeddings. Teachers with base operator embeddings retain mean **{base_operator:.2f}%** exact match. Teacher-forced answer-token accuracy is {aux_token:.2f}% for auxiliary students versus {neutral_token:.2f}% for neutral students. This supplies correct answer prefixes and is a separate diagnostic; it is not free-running addition success. These interventions test limited interface explanations; they do not establish that all possible decoders would fail to recover information.

## Decision

Keep ordinary addition as the working task. This carrier has not demonstrated auxiliary-only addition transfer. Do not increase operand count or fit a scaling law to this result. First test a carrier with digit-position resets and numerical group structure while excluding addition operators and arithmetic labels, or a shared competent formatting base. Predeclare the next variant and retain these null results rather than replacing them. Use the existing MNIST auxiliary-logit result as a positive calibration alongside new addition variants.

Before declaring any carrier ineffective, require high teacher accuracy and a working direct-distillation control. If the direct arm is insufficient, extend or repair that control first. A broader unique-data × exposure grid and fresh-seed confirmation follow a promising carrier, not precede it.

## Artifacts

- `pilot_seed*/results.json`: per-seed curves, complete configuration, costs, and sample generations.
- `pilot_seed*/interface_diagnostic.json`: native and teacher-assisted results, plus held-out losses.
- `curves.csv`, `summary.json`, `learning_curves.png`, `learning_curves.pdf`.
- `experiments/addition/transfer.py`, `diagnose_transfer.py`, `check_transfer.py`, and `TRANSFER_PROTOCOL.md`.

All new artifacts remain local. Existing composition, arithmetic baselines, and flow results are preserved.
"""
(root / "REPORT.md").write_text(report)
print(json.dumps(summary, indent=2))
