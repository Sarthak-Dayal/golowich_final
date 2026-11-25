"""Generate auditable tables, plots, and a measured-results website from saved JSON."""

import csv, json, shutil
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/composition")
WEB = Path("research-plan/dist/measured")
WEB.mkdir(parents=True, exist_ok=True)
runs = [
    json.loads((ROOT / f"confirm_seed{s}_k1/results.json").read_text())
    for s in range(10, 15)
]
assert all(r["gate_passed"] and len(r["arms"]) == 4 for r in runs)
assert len({r["source_sha256"] for r in runs}) == 1
arms = ["transfer", "neutral", "shuffled", "direct"]
names = ["Auxiliary only", "Neutral teacher", "Shuffled outputs", "Direct target"]
colors = ["#18594f", "#85938a", "#b0914d", "#ba7953"]
N = [256, 1024, 4096, 16384]
values = {a: np.array([[row["h"] for row in r["arms"][a]] for r in runs]) for a in arms}
delta = values["transfer"][:, -1] - values["neutral"][:, -1]
ci = 2.776445105 * delta.std(ddof=1) / np.sqrt(5)
summary = {
    "status": "MEASURED; five fresh seeds, selected after discovery pilots",
    "seeds": list(range(10, 15)),
    "mean_teacher_accuracy": np.mean([r["teacher"]["h"] for r in runs]),
    "paired_transfer_minus_neutral_pp": 100 * delta.mean(),
    "paired_t95_interval_pp": [100 * (delta.mean() - ci), 100 * (delta.mean() + ci)],
    "final_accuracy": {a: float(v[:, -1].mean()) for a, v in values.items()},
    "per_seed": [
        {
            "seed": r["config"]["seed"],
            "base": r["initial"]["h"],
            "teacher": r["teacher"]["h"],
            **{a: r["arms"][a][-1]["h"] for a in arms},
        }
        for r in runs
    ],
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
with (ROOT / "curves.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "seed",
            "arm",
            "presentations",
            "unique_examples",
            "accuracy",
            "other_key_accuracy",
        ]
    )
    for r in runs:
        for a in arms:
            for row in r["arms"][a]:
                w.writerow(
                    [
                        r["config"]["seed"],
                        a,
                        row["exposures"],
                        row["unique_examples"],
                        row["h"],
                        row["other_key"],
                    ]
                )
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    }
)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
for a, name, col in zip(arms, names, colors):
    v = values[a]
    for row in v:
        axes[0].plot(N, row * 100, color=col, alpha=0.18, lw=1)
    axes[0].plot(N, v.mean(0) * 100, "o-", color=col, label=name, lw=2)
axes[0].set(
    xscale="log",
    xlabel="Examples presented (one pass in auxiliary arms)",
    ylabel="Target accuracy (%)",
    ylim=(0, 103),
    title="Measured k=1 results · five fresh seeds",
)
axes[0].set_xticks(N, labels=["256", "1,024", "4,096", "16,384"])
axes[0].grid(alpha=0.15)
axes[0].legend(fontsize=8)
for i, r in enumerate(runs):
    axes[1].plot(
        [0, 1],
        [values["neutral"][i, -1] * 100, values["transfer"][i, -1] * 100],
        "-o",
        color="#18594f",
        alpha=0.7,
    )
    axes[1].annotate(
        str(r["config"]["seed"]),
        (1, values["transfer"][i, -1] * 100),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8,
    )
axes[1].set(
    xticks=[0, 1],
    xticklabels=["Neutral teacher", "Auxiliary only"],
    ylabel="Target accuracy (%)",
    xlim=(-0.2, 1.25),
    ylim=(0, 65),
    title="Paired endpoints · 16,384 presentations",
)
axes[1].grid(axis="y", alpha=0.15)
fig.savefig(ROOT / "learning_curves.png", dpi=180)
fig.savefig(ROOT / "learning_curves.pdf")
plt.close(fig)
repeat = []
for seed in range(3):
    p = ROOT / f"gpu_seed{seed}_k1/repeat.json"
    if p.exists():
        repeat.append(json.loads(p.read_text()))
reptext = "Diagnostic still running."
if len(repeat) == 3 and all(len(r["arms"]) == 3 for r in repeat):
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for a, col, name in zip(arms[:3], colors[:3], names[:3]):
        vv = np.array([[q["accuracy"] for q in r["arms"][a]] for r in repeat])
        for row in vv:
            ax.plot([1, 2, 4, 8], row * 100, color=col, alpha=0.2)
        ax.plot([1, 2, 4, 8], vv.mean(0) * 100, "o-", color=col, label=name)
    ax.set(
        xlabel="Passes over the same 16,384 auxiliary examples",
        ylabel="Target accuracy (%)",
        title="Exploratory repetition diagnostic · three discovery seeds",
        ylim=(0, 100),
        xticks=[1, 2, 4, 8],
    )
    ax.grid(alpha=0.15)
    ax.legend()
    fig.savefig(ROOT / "repeated_data.png", dpi=180)
    plt.close(fig)
    reptext = (
        "At eight passes, mean accuracy is "
        + ", ".join(
            f"{a}: {100 * np.mean([r['arms'][a][-1]['accuracy'] for r in repeat]):.1f}%"
            for a in arms[:3]
        )
        + ". These are discovery seeds, not part of the five-seed estimate."
    )
for p in [
    "learning_curves.png",
    "learning_curves.pdf",
    "summary.json",
    "curves.csv",
    "repeated_data.png",
]:
    if (ROOT / p).exists():
        shutil.copy2(ROOT / p, WEB / p)
rows = "\n".join(
    "| "
    + " | ".join(
        [str(r["seed"])] + [f"{r[a] * 100:.2f}" for a in ["base", "teacher"] + arms]
    )
    + " |"
    for r in summary["per_seed"]
)
report = f"""# Measured composition pilot results

Five fresh seed pairs (10–14), after discovery-seed task calibration. Ran on Quadro RTX 6000 GPUs with PyTorch 2.5.1/CUDA 12.4. These are measured results, not the synthetic website figures.

## Main result

At 16,384 presentations, auxiliary-only accuracy is {100 * summary["final_accuracy"]["transfer"]:.2f}%, versus {100 * summary["final_accuracy"]["neutral"]:.2f}% for neutral-teacher distillation. Paired improvement: **{100 * delta.mean():.2f} percentage points**, approximate 95% Student-t interval **[{100 * (delta.mean() - ci):.2f}, {100 * (delta.mean() + ci):.2f}] pp** across five training seeds (df=4). The interval assumes independent seeds and approximately normal paired effects; five seeds give limited precision. No multiple-budget hypothesis tests were performed.

| Seed | Base | Teacher | Auxiliary only | Neutral | Shuffled | Direct |
|---|---|---|---|---|---|---|
{rows}

All entries are target accuracy percentages on 1,024 evaluation draws per seed. Thin figure lines show individual seeds; bold lines show means. Evaluation examples are not independent training replicas. Sampled test sets may contain duplicate examples; uncertainty is across training seeds. Test and validation inputs are excluded from training in the fresh runs.

## What was actually tested

- Eight symbols, k=1; four transformer layers, width 128, four heads. Not the proposed 32-symbol/depth sweep.
- Target y=h(p(x)); public permutation p changes every example; hidden h changes across seeds.
- Both output heads frozen. Shared base learns public composition for 2,400 steps; teacher then learns h. All five teachers passed the >=95% validation gate.
- Aux inputs use a disjoint random-symbol alphabet, with the same sequence length and source-position embedding structure. They are unrelated random sequences, but not intended to be distribution-matched natural text. The auxiliary channel contains 32 soft logits; direct supervision contains 8 target logits.
- Student arms start from exactly the same shared base. Teacher is frozen during distillation. Optimizer, batch size and exposure counts match across arms.
- Auxiliary inputs are unique and used once. Direct target inputs can repeat; their unique counts appear separately in curves.csv. Thus plots compare presentations, not strictly matched unique target counts or transmitted information.
- k=1 is table lookup plus a hidden output permutation. This supports partial transfer of that remapping, not acquisition of a multi-step algorithm.

The public base is not a random classifier: it predicts p(x). It already answers correctly wherever h fixes p(x). Its accuracy therefore varies with the hidden permutation’s fixed points (for example seed 12 starts at 36.6%). The matched neutral teacher, rather than the nominal 12.5% random-guess rate, is the correct transfer baseline.

## Difficulty manipulation failed its gate

The unstructured small CPU teachers stayed near 12.5% chance for k=1 and k=2. Adding explicit source keys and a public-algorithm warm-up solved k=1. The GPU k=2 discovery teacher still finished at 10.84% validation accuracy after 3,200 hidden-key steps, and the public warm-up loss remained near log(8). Its transfer sweep was correctly skipped. No efficiency-versus-depth claim can be made. k=4 was not run because k=2 was already unsolved.

## Optimization diagnostic

{reptext}

## Efficiency interpretation

None of the five auxiliary-only trajectories reaches 50% recovery of its teacher–base gap by 16,384 presentations. All direct trajectories exceed that recovery level by 1,024 presentations and remain above it at 4,096. This gives a >16-fold presentation-budget bound under the pilot's scheduled-checkpoint definition, not an estimate of a universal sample-complexity ratio. Teacher preparation, public pretraining, output-channel size, and GPU cost are not equated by this comparison.

## Next decisions

1. Keep this k=1 experiment as a working partial-transfer baseline. Reproduce with additional seeds and test teacher-key specificity by training matched alternative-key teachers, not only evaluating a second key.
2. Solve public k=2 composition before any larger depth sweep. Test staged intermediate-state supervision for base pretraining; account for its extra supervision. Do not select architectures based on transfer magnitude.
3. Run an explicit unique-data × exposure grid after fixing that base, since the exploratory repetition diagnostic can reveal optimization effects.
4. Only then increase symbols to 32 and compare k=1,2,4. Do not pool these discovery results with later confirmation.

## Artifacts and provenance

Raw results and source snapshots: `results/composition/confirm_seed*_k1/`. `summary.json`, `curves.csv`, and this report are generated by `experiments/composition/summarize.py`. Checkpoints are local under ignored `checkpoints/` directories. Early discovery results are retained separately and never pooled. Scripts borrow the fixed-readout/auxiliary-KL experimental design from Varun's MNIST branch, not its implementation code. Full experimental instructions and dependency lock are in `experiments/composition/`.
"""
repair_path = ROOT / "repair_seed20_k2/results.json"
repair_text = ""
if repair_path.exists():
    repair = json.loads(repair_path.read_text())
    repair_text = f"A second exploratory k=2 repair added explicit table-stage embeddings and a mixed k=1/k=2 curriculum: 6,400 public warm-up steps and {repair['teacher_log'][-1]['step']:,} teacher steps. Final target validation accuracy was {100 * repair['teacher_log'][-1]['validation']:.2f}%; gate passed: {repair['gate_passed']}. This is one discovery seed, not a replicated depth comparison. Scheduled training losses in this mixed run are from the easy k=1 batches; they must not be read as k=2 training success."
    report = report.replace(
        "## Next decisions",
        "## Bounded k=2 repair attempt\n\n" + repair_text + "\n\n## Next decisions",
    )
(ROOT / "REPORT.md").write_text(report)
tr = "".join(
    "<tr><td>"
    + str(r["seed"])
    + "</td>"
    + "".join(f"<td>{100 * r[a]:.1f}%</td>" for a in ["teacher"] + arms)
    + "</tr>"
    for r in summary["per_seed"]
)
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Measured composition results</title><link rel="stylesheet" href="../style.css"></head><body><aside><a class="brand" href="../index.html">∿ TRANSFER LAB</a><p>MEASURED RESULTS</p><a href="../index.html">← Research plan</a></aside><main><header><span>COMPOSITION PILOT</span><span>MEASURED · FIVE FRESH SEEDS</span></header><section><div class="eyebrow">ACTUAL GPU EXPERIMENTS · NOVEMBER 15, 2026</div><h1>Partial transfer.<br>A difficulty limit.</h1><p class="lead">Auxiliary-only distillation improves target accuracy, but is far slower than direct supervision. The depth-two teacher does not solve the task.</p><div class="metrics"><article><span>AUXILIARY ONLY</span><strong>{100 * summary["final_accuracy"]["transfer"]:.1f}%</strong><p>16,384 unique auxiliary prompts</p></article><article><span>PAIRED IMPROVEMENT</span><strong>+{100 * delta.mean():.1f}<small> pp</small></strong><p>95% t interval: {100 * (delta.mean() - ci):.1f} to {100 * (delta.mean() + ci):.1f} pp</p></article><article><span>DIRECT DISTILLATION</span><strong>{100 * summary["final_accuracy"]["direct"]:.1f}%</strong><p>Same number of presentations</p></article></div><img src="learning_curves.png" alt="Measured learning curves and paired seed endpoints" style="width:100%;height:auto"><p class="caption">Five fresh seeds. Thin lines: individual runs. Bold lines: means. Uncertainty is across training seeds, not test images. Auxiliary data is unique; target data may repeat.</p><div class="panel table-wrap"><table><thead><tr><th>Seed</th><th>Teacher</th><th>Auxiliary</th><th>Neutral</th><th>Shuffled</th><th>Direct</th></tr></thead><tbody>{tr}</tbody></table></div><div class="callout"><strong>What this establishes</strong><p>Partial transfer of a hidden output permutation at k=1 in an eight-symbol transformer task, using a shared public-composition base and fixed random readouts. This does not yet establish transfer of multi-step computation, ordinary text-only transfer, or a scaling law.</p></div></section><section><h2>The depth sweep is not ready.</h2><p>The GPU k=2 teacher ended at 10.84% validation accuracy; chance is 12.5%. Its transfer sweep was skipped. k=4 was not launched. Solve public composition and validate a curriculum before interpreting difficulty-dependent transfer.</p><p>{repair_text}</p><h3>Additional optimization on fixed data</h3><p>{reptext}</p>{'<img src="repeated_data.png" alt="Measured exploratory repeated-data diagnostic" style="width:100%;height:auto">' if (ROOT / "repeated_data.png").exists() else ""}<h3>Recommended next experiment</h3><p>Validate k=2 public composition using intermediate-state supervision during base training, then freeze that protocol before measuring hidden-key transfer. Separately cross unique auxiliary data with repeat exposures. Preserve this k=1 result as the calibration baseline.</p><p><a href="summary.json">Summary JSON</a> · <a href="curves.csv">Per-seed CSV</a> · <a href="learning_curves.pdf">Figure PDF</a> · <a href="REPORT.md">Full methods, limitations and results</a></p></section></main></body></html>"""
(WEB / "index.html").write_text(html)
shutil.copy2(ROOT / "REPORT.md", WEB / "REPORT.md")
print(json.dumps(summary, indent=2))
