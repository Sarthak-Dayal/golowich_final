"""Summarize the two bounded parallel pilots without selecting checkpoints."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/parallel_diagnostics"


def main():
    carry_path = ROOT / "results/addition_shared_base/pilot_seed0/results.json"
    trait_path = ROOT / "results/addition_trait/pilot_seed0/results.json"
    carry = json.loads(carry_path.read_text())
    trait = json.loads(trait_path.read_text())
    assert carry["status"] == trait["status"] == "completed", "Wait for complete fixed-budget pilots"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in ["base", "teacher", "transfer", "neutral", "shuffled", "direct"]:
        scores = carry[arm + "_test"] if arm in ("base", "teacher") else carry["arms"][arm]["final_test"]
        for subset, value in scores.items():
            rows.append(dict(experiment="incremental_carry", arm=arm, metric=subset + "_exact", accuracy=value))
    for name, arm in trait["arms"].items():
        for shift, value in arm["final_test"]["exact"].items():
            rows.append(dict(experiment="output_convention", arm=name, metric="shift_" + shift + "_exact", accuracy=value))
    with (OUT / "endpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["experiment", "arm", "metric", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    arms = ["base", "teacher", "transfer", "neutral", "shuffled", "direct"]
    labels = ["Base", "Teacher", "Auxiliary", "Neutral", "Shuffled", "Direct"]
    values = [100 * next(r["accuracy"] for r in rows if r["experiment"] == "incremental_carry" and r["arm"] == a and r["metric"] == "carry_exact") for a in arms]
    colors = ["#8a99ac", "#183b56", "#138a7a", "#a6b1bf", "#c5cdd6", "#e6a23c"]
    bars = axes[0].bar(labels, values, color=colors)
    axes[0].bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    axes[0].set_title("Incremental carry skill (512 carry problems)")

    kinds = ["transfer", "shuffled", "direct"]
    x = np.arange(len(kinds))
    for shift, label, color, offset in [(1, "Digits +1", "#138a7a", -.19), (9, "Digits −1", "#4079b4", .19)]:
        y = [100 * trait["arms"][f"{kind}_{shift}"]["final_test"]["exact"][str(shift)] for kind in kinds]
        bars = axes[1].bar(x + offset, y, .36, label=label, color=color)
        axes[1].bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    axes[1].set_xticks(x, ["Auxiliary", "Shuffled", "Direct"])
    axes[1].set_title("New output convention (256 problems)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_ylim(0, 111)
        axis.set_ylabel("Whole-answer exact accuracy (%)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelrotation=20)
    fig.suptitle("Exploratory seed-0 pilots · fixed final budgets · no independent-seed uncertainty", fontsize=12)
    fig.savefig(OUT / "endpoints.png", dpi=180)
    fig.savefig(OUT / "endpoints.pdf")
    plt.close(fig)

    lines = ["# Parallel addition diagnostics", "", "Measured exploratory seed-0 pilots. These endpoints were not selected by best checkpoint. No independent-seed replication or statistical significance claim is made.", "", "| Experiment | Student arm | Exact accuracy |", "|---|---|---:|"]
    for row in rows:
        if row["experiment"] == "incremental_carry" and row["metric"] == "carry_exact":
            lines.append(f"| Carry problems | {row['arm']} | {100*row['accuracy']:.2f}% |")
        elif row["experiment"] == "output_convention":
            arm_shift = row["arm"].rsplit("_", 1)[-1]
            if row["metric"] == "shift_" + arm_shift + "_exact" and arm_shift != "0":
                lines.append(f"| Output convention | {row['arm']} | {100*row['accuracy']:.2f}% |")
    lines += ["", "Both students inherit the competence of their respective saved bases before distillation. Carry students start from no-carry arithmetic; convention students start from full arithmetic. Output-convention accuracy is not evidence of newly acquiring addition.", "", "Each arm repeats a fixed cache, so these are exposure curves rather than completed unique-data efficiency sweeps. Direct and auxiliary arms supply different numbers of supervised positions per example. All cached teacher queries precede student training; early checkpoints do not imply a matching total teacher-query budget.", "", f"Measured process time: carry {carry['seconds']:.1f}s; output convention {trait['seconds']:.1f}s. Runs used separate GPUs and overlapped.", "", "See [carry report](../addition_shared_base/pilot_seed0/REPORT.md), [convention report](../addition_trait/pilot_seed0/REPORT.md), and [forward-only carrier audit](../addition_diagnostics/carrier_seed0/REPORT.md) for data access, provenance, controls and interpretation. Additional base-seed replications, when available, are summarized in the [convention aggregate report](../addition_trait/REPORT.md).", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print(OUT / "REPORT.md")


if __name__ == "__main__":
    main()
