"""Read-only notebook views of measured convention-transfer checkpoints."""

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"+1": "#bb5915", "−1": "#176ca4"}
BUDGETS = (512, 1024, 2048, 4096, 8192)


def load_data(root):
    root = Path(root)
    runs, rows, pairs = [], [], {}
    for seed in range(3):
        folder = root / f"seed{seed}"
        run = json.loads((folder / "results.json").read_text())
        runs.append(run)
        pairs[seed] = json.loads((folder / "evaluation_pairs.json").read_text())["test"]
        for arm_name, arm in run["arms"].items():
            kind, s = arm_name.split("_")
            for point in arm["curve"]:
                for shift in ((1, 9) if kind == "neutral" else (int(s),)):
                    d = point["test"]["diagnostics"][str(shift)]
                    rows.append({"seed": seed, "kind": kind, "shift": shift,
                                 "convention": "+1" if shift == 1 else "−1",
                                 "updates": point["updates"], "multiple": point["updates"] / 512,
                                 "presentations": point["presentations"], "point": point,
                                 "exact": d["exact"], "digits": d["digit_accuracy"],
                                 "format": d["valid_format_rate"],
                                 "stop_only": d["right_digits_wrong_termination_rate"],
                                 "correct_prefix_digits": point["teacher_forced_test"]["answer_digit_accuracy"],
                                 "aux_kl": point["heldout_auxiliary_kl"]})
    return runs, rows, pairs


def selected(rows, seed):
    return rows if seed == "All three" else [r for r in rows if r["seed"] == int(seed[-1])]


def series(rows, kind, shift, metric="exact"):
    # Aggregate only checkpoints present for every selected seed; missing is never zero.
    seeds = {r["seed"] for r in rows}
    points = []
    for u in BUDGETS:
        use = [r for r in rows if r["kind"] == kind and r["shift"] == shift and r["updates"] == u]
        if {r["seed"] for r in use} == seeds and use:
            values = [r[metric] for r in use]
            points.append({"updates": u, "x": u / 512, "mean": mean(values),
                           "min": min(values), "max": max(values)})
    return points


def axes_budget(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], ["1×", "2×", "4×", "8×", "16×"])
    ax.set_xlabel("Student training budget (1× = 512 updates)")
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def overview(rows, seed="All three", show_seeds=True):
    rows = selected(rows, seed)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for shift, label in ((1, "+1"), (9, "−1")):
        points = series(rows, "transfer", shift)
        color = COLORS[label]
        ax.plot([p["x"] for p in points], [100 * p["mean"] for p in points], "o-", color=color, linewidth=2.5, label=label + " auxiliary")
        if seed == "All three":
            ax.fill_between([p["x"] for p in points], [100 * p["min"] for p in points], [100 * p["max"] for p in points], color=color, alpha=.1)
            if show_seeds:
                for sid in range(3):
                    rr = sorted([r for r in rows if r["seed"] == sid and r["shift"] == shift and r["kind"] == "transfer"], key=lambda r: r["updates"])
                    ax.plot([r["multiple"] for r in rr], [100 * r["exact"] for r in rr], color=color, linewidth=.9, alpha=.35)
    axes_budget(ax)
    ax.set(ylabel="Complete shifted-answer accuracy (%)", ylim=(0, 100),
           title="More training helps initially; the last checkpoint declines")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def controls(rows, seed="All three"):
    rows = selected(rows, seed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
    for ax, shift, label in zip(axes, (1, 9), ("+1", "−1")):
        for kind, color, style, name in (("transfer", COLORS[label], "-", "Auxiliary"),
                                       ("direct", "#3a764b", "--", "Direct"),
                                       ("shuffled", "#b74646", ":", "Shuffled"),
                                       ("neutral", "#666666", "-.", "Neutral")):
            points = series(rows, kind, shift)
            if points:
                ax.plot([p["x"] for p in points], [100 * p["mean"] for p in points], style, marker="o", color=color, markersize=4, label=name)
        axes_budget(ax)
        ax.set(title=label + " convention", ylim=(-3, 103))
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("Complete shifted-answer accuracy (%)")
    fig.tight_layout()
    return fig


def diagnostic_curves(rows, seed="All three"):
    rows = selected(rows, seed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for shift, label in ((1, "+1"), (9, "−1")):
        for key, style, suffix in (("digits", "-", "generated"), ("correct_prefix_digits", "--", "correct prefixes")):
            points = series(rows, "transfer", shift, key)
            axes[0].plot([p["x"] for p in points], [100 * p["mean"] for p in points], style, color=COLORS[label], marker="o", markersize=3, label=f"{label}: {suffix}")
        points = series(rows, "transfer", shift, "aux_kl")
        axes[1].plot([p["x"] for p in points], [p["mean"] for p in points], "o-", color=COLORS[label], label=label)
    for ax in axes:
        axes_budget(ax)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set(title="Digit errors persist with correct answer prefixes", ylabel="Shifted digit accuracy (%)", ylim=(0, 100))
    axes[1].set(title="Imitation on fresh random strings", ylabel="Teacher-to-student KL (nats / position)", yscale="log")
    fig.tight_layout()
    return fig


def position_errors(rows, updates, seed="All three"):
    rows = selected(rows, seed)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for shift, label in ((1, "+1"), (9, "−1")):
        use = [r for r in rows if r["kind"] == "transfer" and r["shift"] == shift and r["updates"] == updates]
        counts = np.zeros((5, 2))
        for r in use:
            for j, p in enumerate(r["point"]["test"]["diagnostics"][str(shift)]["per_digit_position"]):
                counts[j] += [p["correct"], p["n"]]
        values = 100 * counts[:, 0] / np.maximum(1, counts[:, 1])
        ax.bar(np.arange(5) + (-.18 if shift == 1 else .18), values, .35, color=COLORS[label], label=label)
    ax.set_xticks(range(5), ["Ones", "Tens", "Hundreds", "Thousands", "Ten-thousands\n(5-digit sums only)"])
    ax.set(ylabel="Generated shifted digit correct (%)", ylim=(0, 102), title=f"Where answers fail at {updates:,} updates ({updates / 512:g}×)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def summary_table(rows, seed="All three"):
    rows = selected(rows, seed)
    out = []
    plus = {p["updates"]: p for p in series(rows, "transfer", 1)}
    minus = {p["updates"]: p for p in series(rows, "transfer", 9)}
    for u in BUDGETS:
        p, m = plus.get(u), minus.get(u)
        out.append({"Updates": u, "Presentations": u * 128, "Relative budget": f"{u / 512:g}×",
                    "+1 exact (%)": None if not p else round(p["mean"] * 100, 2),
                    "−1 exact (%)": None if not m else round(m["mean"] * 100, 2),
                    "−1 minus +1 (pp)": None if not p or not m else round((m["mean"] - p["mean"]) * 100, 2)})
    return out


def neutral_retention(runs):
    table = []
    for updates in BUDGETS:
        row = {"Updates": updates, "Relative budget": f"{updates / 512:g}×"}
        values = []
        for run in runs:
            point = next((p for p in run["arms"].get("neutral_0", {}).get("curve", []) if p["updates"] == updates), None)
            value = None if point is None else 100 * point["test"]["exact"]["0"]
            row[f"Seed {run['seed']}: ordinary addition (%)"] = None if value is None else round(value, 2)
            if value is not None:
                values.append(value)
        row["Mean ordinary addition (%)"] = round(mean(values), 2) if len(values) == 3 else None
        table.append(row)
    return table


def decode(tokens):
    names = {10: "+", 11: "=", 12: "EOS", 13: "PAD", 14: "BOS"}
    digits = tokens[:-1] if tokens and tokens[-1] == 12 else tokens
    result = "".join(str(t) if t < 10 else f"<{names.get(t, t)}>" for t in digits[::-1])
    return result + ("" if tokens and tokens[-1] == 12 else " [missing EOS]")


def examples_table(rows, pairs, updates, seed, convention, outcome):
    sid = 0 if seed == "All three" else int(seed[-1])
    shift = 1 if convention == "+1" else 9
    found = [r for r in rows if r["kind"] == "transfer" and r["shift"] == shift and r["seed"] == sid and r["updates"] == updates]
    if not found:
        return []
    out = []
    for (a, b), pred in zip(pairs[sid], found[0]["point"]["test"]["predictions"]):
        target = [(int(d) + shift) % 10 for d in str(a + b)[::-1]] + [12]
        correct = pred == target
        if outcome == "Correct only" and not correct or outcome == "Incorrect only" and correct:
            continue
        out.append({"Input": f"{a} + {b}", "Ordinary sum": str(a + b),
                    "Shifted target": decode(target), "Student output": decode(pred), "Exact": correct})
    return out
