"""Report every fixed duration endpoint, without selecting a best checkpoint."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


def pct(x):
    return f"{100 * x:.2f}%"


def summarize(root, partial=False):
    root = Path(root)
    runs = [json.loads((root / f"seed{s}/results.json").read_text()) for s in range(3)]
    if not partial:
        assert all(r["status"] == "completed" for r in runs), "Wait for all fixed endpoints"
        assert all(len(r["arms"]) == 7 and all(a["complete"] for a in r["arms"].values()) for r in runs)
        assert all(r["teachers_unchanged"] and r["base_unchanged"] for r in runs)
    rows = []
    for r in runs:
        seed = r["seed"]
        p = root / f"seed{seed}"
        assert hashlib.sha256((p / "executed_source.txt").read_bytes()).hexdigest() == r["source_sha256"]
        for name, arm in r["arms"].items():
            if "reconstruction" in arm:
                assert arm["reconstruction"]["tensors_equal"]
                assert arm["reconstruction"]["max_abs_difference"] == 0
            kind, source_shift = name.split("_")
            source_shift = int(source_shift)
            for point in arm["curve"]:
                for shift in ([1, 9] if kind == "neutral" else [source_shift]):
                    s = str(shift)
                    d = point["test"]["diagnostics"][s]
                    rows.append({
                        "seed": seed, "kind": kind, "convention": "+1" if shift == 1 else "-1",
                        "updates": point["updates"], "presentations": point["presentations"],
                        "budget_multiple": point["updates"] / 512, "unique_strings_or_problems": point["unique_examples"],
                        "passes": point["passes"], "test_exact": point["test"]["exact"][s],
                        "validation_exact": point["validation"][s],
                        "free_generation_digit_accuracy": d["digit_accuracy"],
                        "first_digit_accuracy": d["first_digit_accuracy"],
                        "valid_format_rate": d["valid_format_rate"],
                        "right_digits_wrong_termination_rate": d["right_digits_wrong_termination_rate"],
                        "heldout_auxiliary_kl": point["heldout_auxiliary_kl"],
                        # Neutral's teacher-forced diagnostic uses ordinary answer prefixes.
                        "teacher_forced_own_rule_digit_accuracy": point["teacher_forced_test"]["answer_digit_accuracy"],
                        "teacher_forced_own_rule_token_ce": point["teacher_forced_test"]["answer_token_ce"],
                        "seconds_from_arm_start": point["seconds"],
                    })
    assert rows
    with (root / "curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregated = []
    keys = sorted({(x["kind"], x["convention"], x["updates"]) for x in rows})
    metrics = ["test_exact", "validation_exact", "free_generation_digit_accuracy", "first_digit_accuracy",
               "valid_format_rate", "right_digits_wrong_termination_rate", "heldout_auxiliary_kl",
               "teacher_forced_own_rule_digit_accuracy", "teacher_forced_own_rule_token_ce"]
    for kind, convention, updates in keys:
        values = [x for x in rows if (x["kind"], x["convention"], x["updates"]) == (kind, convention, updates)]
        stats = {k: {"mean": mean(v[k] for v in values), "min": min(v[k] for v in values),
                     "max": max(v[k] for v in values)} for k in metrics}
        aggregated.append({"kind": kind, "convention": convention, "updates": updates,
                           "presentations": updates * 128, "budget_multiple": updates / 512,
                           "n_seed_groups": len(values), "metrics": stats})
    summary = {"status": "partial" if partial else "completed", "n_seed_groups": 3,
               "study": "Longer auxiliary-only convention transfer from an addition-competent base",
               "fixed_endpoint_updates": 8192, "fixed_endpoint_presentations": 1048576,
               "same_test_inputs_across_seeds": True, "test_pairs_per_model": 256,
               "unique_auxiliary_inputs": 4096, "all_reconstructed_student_states_exact": all(
                   a.get("reconstruction", {}).get("tensors_equal", False) for r in runs for a in r["arms"].values()),
               "runs": [{"seed": r["seed"], "status": r["status"], "seconds": r["seconds"],
                         "teachers_unchanged": r.get("teachers_unchanged"), "base_unchanged": r.get("base_unchanged"),
                         "initial_sha256": r["initial_sha256"]} for r in runs],
               "aggregated": aggregated, "rows": rows}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def get(kind, convention, updates):
        found = [a for a in aggregated if (a["kind"], a["convention"], a["updates"]) == (kind, convention, updates) and a["n_seed_groups"] == 3]
        return found[0] if found else None

    def value(kind, convention, updates, metric="test_exact"):
        found = get(kind, convention, updates)
        return "pending" if not found else pct(found["metrics"][metric]["mean"])

    lines = ["# Longer student training: +1 and −1 answer conventions", "",
             "Status: **" + summary["status"] + "**. Three existing base/teacher/student seed groups. Students already know addition before distillation; these results concern the teacher's new answer-digit convention.", "",
             "## Fixed duration comparison", "",
             "The original students trained for 512 updates. The fixed longer endpoint is 8,192 updates, sixteen times longer, at the original constant learning rate. Teachers, initialization, input pools, shuffle assignments, minibatches, optimizer and loss are held fixed. Original AdamW states were reconstructed by replay; all 54 saved student tensors must match exactly at update 512 before an arm continues. No teacher retraining, best-checkpoint selection or gradient-alignment analysis is used.", "",
             "The table reports mean whole-answer exact test accuracy including EOS across three seed groups, on the same 256 four-digit held-out problems. Every listed budget is retained.", "",
             "| Updates | Presentations | Relative budget | +1 auxiliary | −1 auxiliary | −1 minus +1 |",
             "|---:|---:|---:|---:|---:|---:|"]
    for u in (512, 1024, 2048, 4096, 8192):
        plus, minus = get("transfer", "+1", u), get("transfer", "-1", u)
        gap = "pending" if not plus or not minus else f"{100 * (minus['metrics']['test_exact']['mean'] - plus['metrics']['test_exact']['mean']):+.2f} pp"
        lines.append(f"| {u:,} | {u * 128:,} | {u / 512:g}× | {value('transfer', '+1', u)} | {value('transfer', '-1', u)} | {gap} |")
    if all(get("transfer", conv, u) for conv in ("+1", "-1") for u in (512, 4096, 8192)):
        initial_gap = 100 * (get("transfer", "-1", 512)["metrics"]["test_exact"]["mean"] - get("transfer", "+1", 512)["metrics"]["test_exact"]["mean"])
        final_gap = 100 * (get("transfer", "-1", 8192)["metrics"]["test_exact"]["mean"] - get("transfer", "+1", 8192)["metrics"]["test_exact"]["mean"])
        lines += ["", "## Interpretation of the duration curves", "",
                  f"Additional training improves both conventions relative to the original 512-update endpoint. At the intermediate 8× budget, the auxiliary means are {value('transfer', '+1', 4096)} (+1) and {value('transfer', '-1', 4096)} (−1). At the fixed 16× endpoint they fall to {value('transfer', '+1', 8192)} and {value('transfer', '-1', 8192)}. More training at this constant learning rate is not uniformly better.", "",
                  f"The mean −1 advantage changes from {initial_gap:.2f} to {final_gap:.2f} percentage points. The smaller final gap accompanies a decline in both curves, especially −1, so it does not establish convergence to equal transfer. Seed 2 reverses the ordering at the final checkpoint while the other two groups still favor −1. The measured gap depends on training duration and seed; these observations do not establish an intrinsic asymmetry.", "",
                  "The initial difference is visible in generated-digit accuracy and persists when correct answer prefixes are supplied. Termination errors alone do not account for it. The fifth digit of five-digit sums is particularly weak initially for both conventions and improves with training. These are descriptive error locations; the mechanism behind the asymmetry and later decline remains untested. A learning-rate decay comparison on validation, followed by new-seed confirmation, is a reasonable next intervention; it has not been run here."]
    lines += ["", "## Controls and individual seed endpoints", "",
              "| Seed | Convention | Teacher | Auxiliary at 512 | Auxiliary at 8,192 | Shuffled at 8,192 | Neutral at 8,192 | Direct at 8,192 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        for shift, convention in ((1, "+1"), (9, "-1")):
            def endpoint(kind, updates):
                hit = [x for x in rows if x["seed"] == r["seed"] and x["convention"] == convention and x["kind"] == kind and x["updates"] == updates]
                return "pending" if not hit else pct(hit[0]["test_exact"])
            lines.append(f"| {r['seed']} | {convention} | {pct(r['teacher_test'][str(shift)]['exact'][str(shift)])} | {endpoint('transfer', 512)} | {endpoint('transfer', 8192)} | {endpoint('shuffled', 8192)} | {endpoint('neutral', 8192)} | {endpoint('direct', 8192)} |")
    lines += ["", "## Ordinary addition retention in neutral controls", "",
              "This is a separate metric from shifted-answer accuracy. Neutral targets come from the frozen competent base. Nevertheless, extended neutral training degrades ordinary addition in two seed groups while its unrelated-input KL becomes nonzero. Do not describe the neutral students as unchanged checkpoints. This instability motivates an optimizer/numerics follow-up; its cause is not isolated here.", "",
              "| Updates | Seed 0 original sums | Seed 1 original sums | Seed 2 original sums |",
              "|---:|---:|---:|---:|"]
    for u in (512, 1024, 2048, 4096, 8192):
        values = []
        for r in runs:
            point = next((p for p in r["arms"].get("neutral_0", {}).get("curve", []) if p["updates"] == u), None)
            values.append("pending" if not point else pct(point["test"]["exact"]["0"]))
        lines.append(f"| {u:,} | " + " | ".join(values) + " |")
    lines += ["", "## What accounts for failed complete answers?", "",
              "Free-generation digit accuracy counts each expected digit position, with early EOS or missing digits counted wrong. Valid format means the correct answer length, all digits, and a final EOS; it does not require correct digits. The termination column requires all answer digits to be correct while the complete sequence is wrong. Teacher-forced accuracy supplies the correct coded answer prefix, so it measures a different, easier condition than autonomous generation.", "",
              "| Convention | Budget | Exact answer | Digit accuracy | First digit | Valid format | Right digits, wrong termination | Teacher-forced digit accuracy | Fresh random-string KL |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for convention in ("+1", "-1"):
        for u in (512, 8192):
            a = get("transfer", convention, u)
            if a:
                m = a["metrics"]
                vals = [pct(m[k]["mean"]) for k in ("test_exact", "free_generation_digit_accuracy", "first_digit_accuracy", "valid_format_rate", "right_digits_wrong_termination_rate", "teacher_forced_own_rule_digit_accuracy")]
                lines.append(f"| {convention} | {u / 512:g}× | " + " | ".join(vals) + f" | {m['heldout_auxiliary_kl']['mean']:.5f} |")
    lines += ["", "These diagnostics describe where errors occur. They do not establish a causal reason for the +1/−1 asymmetry. Both teachers have almost perfect task accuracy. Training for longer can test whether the gap is transient under this optimizer and fixed input pool; it cannot isolate a representational cause.", "",
              "## Budget, provenance and interpretation", "",
              "Each arm receives 4,096 unique inputs reused for 256 passes at the endpoint: 1,048,576 presentations and 8,192 optimizer updates. Auxiliary and shuffled/neutral arms supervise 15,728,640 token distributions and process 16,777,216 padded forward tokens. Direct arms have different numbers of supervised answer positions, listed below. Equal example/update counts are not equal task information or supervised-token counts.", "",
              "| Seed | Direct supervised positions per arm | Auxiliary supervised positions per arm | All seven reconstructed checkpoints exact? | Run seconds |",
              "|---|---:|---:|---|---:|"]
    for r in runs:
        direct = r["data"]["direct_positions_per_pool"] * 256
        exact = len(r["arms"]) == 7 and all(a.get("reconstruction", {}).get("tensors_equal", False) for a in r["arms"].values())
        lines.append(f"| {r['seed']} | {direct:,} | {15 * 1048576:,} | {exact} | {r['seconds']:.1f} |")
    lines += ["", "Timers include trajectory replay, training, checkpoint evaluation and checkpoint I/O; they exclude initial preparation of the frozen teacher-query caches. Replay is included in the 8,192-update budget, not added on top. Teacher/base preparation costs belong to the original experiments and are not rerun here. These measurements are duration comparisons, not an end-to-end compute-efficiency claim.", "",
              "The effective replication count is three training seed groups; +1 and −1 share a base and test problems within each group. Observed seed ranges are descriptive, not confidence intervals. All groups and budgets have already been examined, so this remains exploratory. Ordinary addition is present in the starting base; shifted-answer performance is not arithmetic acquired from scratch. The original null experiments are preserved.", "",
              "Artifacts: `summary.json`, `curves.csv`, `duration_curves.png/.pdf`, `error_diagnostics.png/.pdf`, and each `seed*/results.json`, full test predictions, evaluation pairs, exact source archives and checkpoints with optimizer/data-generator states. No site upload or publication was performed."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    plot(root, rows, aggregated, runs)
    print(json.dumps({"status": summary["status"], "report": str(root / "REPORT.md"),
                      "auxiliary_endpoints": {c: value("transfer", c, 8192) for c in ("+1", "-1")}}))


def plot(root, rows, aggregated, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    colors = {"+1": "#c45d16", "-1": "#176ca4"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    for ax, conv in zip(axes, ("+1", "-1")):
        for kind, color, style, label in [("transfer", colors[conv], "-", "Auxiliary"),
                                           ("direct", "#44744b", "--", "Direct"),
                                           ("shuffled", "#ad4444", ":", "Shuffled"),
                                           ("neutral", "#666666", "-.", "Neutral")]:
            use = [a for a in aggregated if a["kind"] == kind and a["convention"] == conv and a["n_seed_groups"] == 3]
            use.sort(key=lambda a: a["updates"])
            if not use:
                continue
            x = [a["budget_multiple"] for a in use]
            y = [a["metrics"]["test_exact"]["mean"] * 100 for a in use]
            ax.plot(x, y, style, color=color, marker="o", linewidth=2, markersize=4, label=label)
            if kind == "transfer":
                low = [a["metrics"]["test_exact"]["min"] * 100 for a in use]
                high = [a["metrics"]["test_exact"]["max"] * 100 for a in use]
                ax.fill_between(x, low, high, color=color, alpha=.13, label="Observed seed range")
                for seed in range(3):
                    rr = sorted([r for r in rows if r["seed"] == seed and r["kind"] == kind and r["convention"] == conv], key=lambda r: r["updates"])
                    ax.plot([r["budget_multiple"] for r in rr], [100 * r["test_exact"] for r in rr], color=color, alpha=.22, linewidth=.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1×", "2×", "4×", "8×", "16×"])
        ax.set(xlabel="Training budget (1× = 512 updates)", title=f"{conv} answer-digit convention", ylim=(-3, 103))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Complete shifted-answer accuracy (%)")
    if axes[1].has_data():
        axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Fixed teachers and 4,096 strings; three training seed groups", fontsize=11)
    fig.tight_layout()
    fig.savefig(root / "duration_curves.png", dpi=190)
    fig.savefig(root / "duration_curves.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for conv in ("+1", "-1"):
        use = sorted([a for a in aggregated if a["kind"] == "transfer" and a["convention"] == conv and a["n_seed_groups"] == 3], key=lambda a: a["updates"])
        x = [a["budget_multiple"] for a in use]
        axes[0].plot(x, [100 * a["metrics"]["free_generation_digit_accuracy"]["mean"] for a in use], "o-", color=colors[conv], label=conv + " generated digits")
        axes[0].plot(x, [100 * a["metrics"]["teacher_forced_own_rule_digit_accuracy"]["mean"] for a in use], "--", color=colors[conv], alpha=.7, label=conv + " correct prefixes")
        axes[1].plot(x, [a["metrics"]["heldout_auxiliary_kl"]["mean"] for a in use], "o-", color=colors[conv], label=conv)
        shift = 1 if conv == "+1" else 9
        selected = [r["arms"].get(f"transfer_{shift}", {}).get("curve", []) for r in runs]
        if all(any(p["updates"] == 8192 for p in curve) for curve in selected):
            confusion = np.zeros((10, 16))
            for curve in selected:
                point = next(p for p in curve if p["updates"] == 8192)
                confusion += np.asarray(point["test"]["diagnostics"][str(shift)]["digit_confusion"])
            by_plain_digit = [100 * confusion[(d + shift) % 10, (d + shift) % 10] / max(1, confusion[(d + shift) % 10].sum()) for d in range(10)]
            offset = -.18 if shift == 1 else .18
            axes[2].bar(np.arange(10) + offset, by_plain_digit, width=.35, color=colors[conv], label=conv)
    for ax in axes[:2]:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1×", "2×", "4×", "8×", "16×"])
        ax.set_xlabel("Training budget")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    axes[0].set(title="Digits: generation vs. correct prefixes", ylabel="Digit accuracy (%)", ylim=(0, 102))
    axes[1].set(title="Imitation on fresh random strings", ylabel="Teacher-to-student KL (nats/position)", yscale="log")
    axes[2].set(title="Final errors by ordinary sum digit", xlabel="Original answer digit before shifting", ylabel="Generated shifted digit correct (%)", ylim=(0, 102))
    axes[2].set_xticks(range(10))
    if axes[2].has_data():
        axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(root / "error_diagnostics.png", dpi=190)
    fig.savefig(root / "error_diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="results/addition_trait_long")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    summarize(args.root, args.partial)
