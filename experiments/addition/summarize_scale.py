"""Report the single exploratory eight-digit training pilot separately."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("results/addition")
r = json.loads((root / "scale_seed0_digits8/results.json").read_text())
rows = "\n".join(
    f"| {n} | {'Trained length' if int(n) <= 8 else 'Untrained length'} | {100 * v['exact_match']:.2f}% |"
    for n, v in r["test"].items()
)
text = f"""# Eight-digit training: exploratory size-scaling pilot

This run keeps the same architecture and optimizer as the four-digit baseline, but trains on independently sampled operand lengths from 1–8. It starts from scratch, not from the four-digit checkpoint. This is one seed, not a replicated comparison.

| Test digits per operand | Status | Whole-answer exact match |
|---|---|---|
{rows}

Validation stopping gate passed: **{r["validation_gate_passed"]}**. Training updates: **{r["log"][-1]["step"]:,}**. Example presentations: **{r["log"][-1]["presentations"]:,}**. Distinct unordered training problems: **{r["log"][-1]["unique_problems"]:,}**. Wall time including evaluation: **{r["seconds"]:.1f}s**.

The two- and eight-digit tests contain 256 problems each, held out by the same swap-safe hash rule as before. Sixteen-digit performance is an untrained-length extrapolation test, not an in-distribution result. Test outcomes did not select the stopping checkpoint.

## Interpretation

Successful eight-digit learning, if the held-out score is high, is evidence that this task family can grow beyond the four-digit baseline in the same model. One seed does not establish a scaling law, reliability across seeds, or an equal-compute comparison: longer examples process more tokens per update. Auxiliary-only transfer is still untested on addition.

Before adding more operands or multiplication, replicate this pilot and run the first direct-versus-auxiliary transfer comparison at fixed digit length. Preserve the ordinary-addition baseline and neutral/shuffled controls.

Raw metrics and code snapshot are in `scale_seed0_digits8/`; `scale_learning_curve.png` shows validation, not test-set tuning.
"""
(root / "SCALE_REPORT.md").write_text(text)
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
for n, col in [("2", "#18594f"), ("8", "#ba7953")]:
    ax.plot(
        [v["presentations"] for v in r["log"]],
        [v["validation"][n] * 100 for v in r["log"]],
        "o-",
        color=col,
        label=n + " digits",
    )
ax.set(
    xlabel="Training example presentations",
    ylabel="Whole-answer validation accuracy (%)",
    title="Eight-digit training · single exploratory seed",
    ylim=(-2, 103),
)
ax.legend()
ax.grid(alpha=0.15)
fig.savefig(root / "scale_learning_curve.png", dpi=180)
print(json.dumps({n: v["exact_match"] for n, v in r["test"].items()}))
