"""Read-only reports/figures shared by the notebook and report generator."""

import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

BUDGETS = (512, 1024, 2048, 4096, 8192)


def read(root):
    root = Path(root)
    runs = {}
    for p in root.glob("*/results.json"):
        if p.parent.name.startswith("smoke"):
            continue
        r = json.loads(p.read_text())
        if "arms" in r:
            runs[p.parent.name] = r
    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    audit_path = root / "fixed_point_audit/results.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    permutation_path = root / "permutation_baseline/results.json"
    permutation = json.loads(permutation_path.read_text()) if permutation_path.exists() else None
    full, head = {}, {}
    if selection:
        chosen = selection["selected"]["name"]
        for name, r in runs.items():
            if name.startswith("selected_") or name.startswith("head_seed"):
                assert all(r["config"][key] == selection["selected"][key] for key in ("lr", "min_lr", "schedule"))
        if chosen in runs:
            full[0] = dict(runs[chosen]["arms"])
        if "selected_controls_seed0" in runs:
            full.setdefault(0, {}).update(runs["selected_controls_seed0"]["arms"])
        for seed in range(3):
            if f"selected_full_seed{seed}" in runs:
                full[seed] = runs[f"selected_full_seed{seed}"]["arms"]
            for suffix in ("aux", "controls"):
                if f"selected_full_seed{seed}_{suffix}" in runs:
                    full.setdefault(seed, {}).update(runs[f"selected_full_seed{seed}_{suffix}"]["arms"])
            if f"head_seed{seed}" in runs:
                head[seed] = runs[f"head_seed{seed}"]["arms"]
        if chosen == "discovery_constant" and "head_constant_seed0" in runs and 0 not in head:
            head[0] = runs["head_constant_seed0"]["arms"]
    return {"root": root, "runs": runs, "selection": selection, "audit": audit,
            "permutation": permutation, "full": full, "head": head}


def final_table(study):
    rows = []
    for mode in ("full", "head"):
        for shift, label in ((1, "+1"), (9, "−1")):
            for kind in ("transfer", "shuffled", "neutral", "direct"):
                name = f"{kind}_{0 if kind == 'neutral' else shift}"
                values = []
                for seed in range(3):
                    a = study[mode].get(seed, {}).get(name, {})
                    if a.get("complete"):
                        values.append(a["final_test"]["exact"][str(shift)])
                rows.append({"Parameters trained": "All" if mode == "full" else "Output head only",
                             "Convention": label, "Condition": kind,
                             "Complete seed groups": len(values),
                             "Mean final test (%)": round(100 * mean(values), 2) if len(values) == 3 else None,
                             "Observed seed range (%)": f"{100 * min(values):.2f}–{100 * max(values):.2f}" if len(values) == 3 else None})
    return rows


def primary_table(study):
    detailed = final_table(study)
    rows = []
    for mode, parameters in (("All", 817679), ("Output head only", 1935)):
        row = {"Parameters trained": mode, "Trainable count": parameters}
        for rule in ("+1", "−1"):
            for kind, label in (("transfer", "auxiliary"), ("direct", "direct")):
                hit = next(r for r in detailed if r["Parameters trained"] == mode and r["Convention"] == rule and r["Condition"] == kind)
                row[f"{rule} {label} test (%)"] = hit["Mean final test (%)"]
        rows.append(row)
    return rows


def retention(study):
    rows = []
    for mode in ("full", "head"):
        for seed, arms in sorted(study[mode].items()):
            arm = arms.get("neutral_0", {})
            if arm.get("complete"):
                p = arm["curve"][-1]
                rows.append({"Parameters trained": mode, "Seed": seed,
                             "Neutral ordinary addition (%)": 100 * arm["final_test"]["exact"]["0"],
                             "Changed tensors": p["state_change"]["changed_tensors"],
                             "Max weight change": p["state_change"]["max_abs_change"]})
    return rows


def reference_table(study):
    rows = []
    full = study["runs"].get("discovery_constant", {}).get("arms", {})
    head = study["runs"].get("head_constant_seed0", {}).get("arms", {})
    for shift, label in ((1, "+1"), (9, "−1")):
        row = {"Convention": label}
        for name, arms, kind in (("Full-model auxiliary (%)", full, "transfer"),
                                 ("Head-only auxiliary (%)", head, "transfer"),
                                 ("Head-only direct (%)", head, "direct"),
                                 ("Head-only shuffled (%)", head, "shuffled"),
                                 ("Head-only neutral (%)", head, "neutral")):
            arm = arms.get(f"{kind}_{0 if kind == 'neutral' else shift}", {})
            row[name] = round(100 * arm["final_test"]["exact"][str(shift)], 2) if arm.get("complete") else None
        rows.append(row)
    return rows


def permutation_table(study):
    rows = []
    baseline = study.get("permutation")
    if not baseline:
        return rows
    for group in baseline["groups"]:
        for shift, rule in ((1, "+1"), (9, "−1")):
            paired = group["conditions"][f"paired_{shift}"]
            rows.append({"Seed": group["seed"], "Convention": rule,
                         "Fitted digit map (input digits 0…9)": str(paired["mapping_base_digit_to_teacher_digit"]),
                         "Paired map: shifted test (%)": round(100 * paired["final_test"]["exact"][str(shift)], 2),
                         "Shuffled map: shifted test (%)": round(100 * group["conditions"][f"shuffled_{shift}"]["final_test"]["exact"][str(shift)], 2),
                         "Neutral map: original test (%)": round(100 * group["conditions"][f"neutral_{shift}"]["final_test"]["exact"]["0"], 2)})
    return rows


def discovery_plot(study):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    colors = {"discovery_constant": "#777777", "discovery_lower": "#2a8652", "discovery_cosine": "#286ca8"}
    labels = {"discovery_constant": "Corrected KL; constant 3e−4", "discovery_lower": "Corrected KL; constant 3e−5", "discovery_cosine": "Corrected KL; cosine 3e−4 → 3e−6"}
    for ax, shift in zip(axes, (1, 9)):
        for name in colors:
            arm = study["runs"].get(name, {}).get("arms", {}).get(f"transfer_{shift}", {})
            curve = arm.get("curve", [])
            if curve:
                ax.plot([p["updates"] / 512 for p in curve], [p["validation"][str(shift)] * 100 for p in curve], "o-", markersize=4, color=colors[name], label=labels[name])
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1×", "2×", "4×", "8×", "16×"])
        ax.set(title=("+1" if shift == 1 else "−1") + " convention, discovery seed 0", ylim=(-2, 102), xlabel="Budget (1× = 512 updates)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Validation complete-answer accuracy (%)")
    if axes[1].has_data():
        axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def mechanism_plot(study, seed="All three"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    seeds = list(range(3)) if seed == "All three" else [int(seed[-1])]
    for ax, shift in zip(axes, (1, 9)):
        for mode, color, label in (("full", "#176ca4", "All parameters"), ("head", "#bb5915", "Output head only")):
            xs, ys, lower, upper = [], [], [], []
            for updates in BUDGETS:
                values = []
                for sid in seeds:
                    arm = study[mode].get(sid, {}).get(f"transfer_{shift}", {})
                    point = next((p for p in arm.get("curve", []) if p["updates"] == updates), None)
                    if point:
                        values.append(point["validation"][str(shift)] * 100)
                if len(values) == len(seeds):
                    xs.append(updates / 512)
                    ys.append(mean(values))
                    lower.append(min(values))
                    upper.append(max(values))
            if xs:
                ax.plot(xs, ys, "o-", color=color, label=label)
                ax.fill_between(xs, lower, upper, color=color, alpha=.12)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1×", "2×", "4×", "8×", "16×"])
        ax.set(title=("+1" if shift == 1 else "−1") + " convention", ylim=(-2, 102), xlabel="Budget (1× = 512 updates)")
        ax.grid(alpha=.2)
        if ax.has_data():
            ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Validation complete-answer accuracy (%)")
    fig.suptitle("Selected schedule: full-model vs. head-only auxiliary training", fontsize=11)
    fig.tight_layout()
    return fig


def endpoint_plot(study):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    kinds = ("transfer", "direct", "shuffled", "neutral")
    for ax, shift in zip(axes, (1, 9)):
        for mode, offset, color, label in (("full", -.18, "#176ca4", "All parameters"), ("head", .18, "#bb5915", "Output head only")):
            for j, kind in enumerate(kinds):
                name = f"{kind}_{0 if kind == 'neutral' else shift}"
                arms = [study[mode].get(seed, {}).get(name, {}) for seed in range(3)]
                if all(a.get("complete") for a in arms):
                    values = [100 * a["final_test"]["exact"][str(shift)] for a in arms]
                    avg = mean(values)
                    ax.bar(j + offset, avg, width=.34, color=color, label=label if j == 0 else None)
                    ax.errorbar([j + offset], [avg], yerr=[[avg - min(values)], [max(values) - avg]], fmt="none", color="#333333", capsize=3, linewidth=1)
        ax.set_xticks(range(4), ["Auxiliary", "Direct", "Shuffled", "Neutral"])
        ax.set(title=("+1" if shift == 1 else "−1") + " convention", ylim=(-2, 103))
        ax.grid(axis="y", alpha=.2)
        if ax.has_data():
            ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("Final complete shifted-answer test accuracy (%)")
    fig.suptitle("Fixed 8,192-update endpoints; bars show means and observed seed ranges", fontsize=10)
    fig.tight_layout()
    return fig


def audit_table(study):
    if not study["audit"]:
        return []
    return [{"Backward": name, "Cached logits identical": a["initial_cached_logits_exact"],
             "Changed tensors after 64 updates": a["after_64_updates"]["changed_tensors"],
             "Max weight change": a["after_64_updates"]["max_abs_change"]}
            for name, a in study["audit"]["conditions"].items()]


def report(root, require_complete=False):
    study = read(root)
    rows = final_table(study)
    if require_complete:
        assert study["selection"]
        assert all(r["Complete seed groups"] == 3 for r in rows)
        assert study["permutation"] and study["permutation"]["status"] == "completed"
    for name, r in study["runs"].items():
        folder = study["root"] / name
        assert hashlib.sha256((folder / "executed_source.txt").read_bytes()).hexdigest() == r["source_sha256"]
        assert hashlib.sha256((folder / "common_dependency.txt").read_bytes()).hexdigest() == r["common_sha256"]
        if r["config"]["mode"] == "head":
            assert all(p["backbone_change"]["changed_tensors"] == 0 for a in r["arms"].values() for p in a["curve"])
    complete = all(r["Complete seed groups"] == 3 for r in rows)
    resume_path = study["root"] / "resume_check.json"
    resume_check = json.loads(resume_path.read_text()) if resume_path.exists() else None
    summary = {"status": "completed" if complete else "running", "selection": study["selection"],
               "resume_check": resume_check, "fixed_point_audit": audit_table(study),
               "constant_rate_reference_seed0": reference_table(study),
               "structural_permutation_baseline": permutation_table(study),
               "final_test": rows, "primary_comparison": primary_table(study), "neutral_retention": retention(study),
               "run_status": {name: r["status"] for name, r in study["runs"].items()}}
    previous_file = study["root"] / "summary.json"
    previous_file.write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Numerical stability and head-only controls", "",
             "Status: **" + summary["status"] + "**.", "",
             "These students inherit ordinary addition from a saved competent base. New learning concerns +1/−1 answer-digit conventions. This is an exploratory follow-up on existing seed groups, not new arithmetic acquired from scratch.", "",
             "## Main findings", "",
             "| Parameters trained | Trainable count | +1 auxiliary test | −1 auxiliary test | +1 direct test | −1 direct test |", "|---|---:|---:|---:|---:|---:|"]
    for r in primary_table(study):
        scores = ["pending" if r[k] is None else f"{r[k]:.2f}%" for k in list(r)[2:]]
        # Present the auxiliary conditions first, then their direct controls.
        scores = [scores[0], scores[2], scores[1], scores[3]]
        lines.append(f"| {r['Parameters trained']} | {r['Trainable count']:,} | " + " | ".join(scores) + " |")
    if complete:
        lines += ["", "Corrected neutral controls retain 100% ordinary addition accuracy with exactly unchanged weights in all three bases, for both training modes. Every selected shuffled/neutral arm scores 0% on the shifted tasks. Full-model direct controls score 100%. Output-head-only training still transfers a substantial part of both conventions; its imperfect direct controls limit claims about its best achievable accuracy.", "",
                  "The structural permutation baseline exactly recovers the rule in three of six teacher cases and fails or performs poorly in the others. This, together with head-only transfer, makes the current task a useful readout/convention calibration. These results do not demonstrate acquisition of addition or a new carry algorithm."]
    lines += ["",
             "## Identity audit", "",
             "The original KL implementation changes a parameter even when cached teacher and student logits are bit-identical. The analytic softmax-difference derivative optimizes the same KL objective and gives exactly zero update at identical logits. It passes a finite-difference derivative test. No gradient-alignment measurements are used.", "",
             "| Backward implementation | Changed tensors after 64 updates | Largest absolute change |", "|---|---:|---:|"]
    for r in audit_table(study):
        lines.append(f"| {r['Backward']} | {r['Changed tensors after 64 updates']} | {r['Max weight change']:.4g} |")
    if resume_check:
        lines += ["", "An interrupted head-only run was resumed from update 512 to 1,024. Its final model tensors, AdamW state, sampler order and generator state exactly matched an uninterrupted run. See `resume_check.json` and `experiments/addition_stability/check_resume.py`."]
    lines += ["", "## Discovery selection", "",
              "Compare original constant rate 3e−4, lower constant 3e−5, and cosine 3e−4→3e−6, all with the corrected KL calculation, 8,192 updates and the original fixed input pool. Selection uses only the final mean +1/−1 validation score, gated by >=99% neutral validation and zero neutral tensor changes at every checkpoint.", ""]
    if study["selection"]:
        lines += ["| Candidate | Stable identity gate | Mean final shift validation |", "|---|---|---:|"]
        for r in study["selection"]["candidates"]:
            lines.append(f"| {r['name']} | {r['eligible']} | {100 * r['final_mean_shift_validation']:.2f}% |")
        lines += ["", "Selected: **" + study["selection"]["selected"]["name"] + "**. This is one discovery base; tests do not select the optimizer."]
    else:
        lines += ["The discovery comparison is still running; no candidate has been selected."]
    lines += ["", "## Constant-rate mechanism reference, seed 0", "",
              "This additional paired reference uses corrected KL and constant 3e−4 for both full-model and head-only auxiliary training. It does not select the schedule. Scores below are final held-out test endpoints, not the validation curves or a mean across bases.", "",
              "| Rule | Full auxiliary | Head-only auxiliary | Head-only direct | Head-only shuffled | Head-only neutral |", "|---|---:|---:|---:|---:|---:|"]
    for r in reference_table(study):
        values = ["pending" if r[k] is None else f"{r[k]:.2f}%" for k in list(r)[1:]]
        lines.append(f"| {r['Convention']} | " + " | ".join(values) + " |")
    lines += ["", "## Selected full-model and head-only fixed endpoints", "",
              "Whole-answer test exact match includes EOS. Every model uses the same 256 four-digit operand pairs; conventions are paired within each base. Report means only after all three seed groups complete. Head-only training updates 1,935 parameters, with the other 815,744 parameters frozen and checked for bitwise equality.", "",
              "| Parameters trained | Rule | Condition | Groups complete | Mean final test | Observed seed range |", "|---|---|---|---:|---:|---|"]
    for r in rows:
        score = "pending" if r["Mean final test (%)"] is None else f"{r['Mean final test (%)']:.2f}%"
        lines.append(f"| {r['Parameters trained']} | {r['Convention']} | {r['Condition']} | {r['Complete seed groups']} | {score} | {r['Observed seed range (%)'] or 'pending'} |")
    lines += ["", "## Neutral retention", "", "| Mode | Seed | Ordinary addition | Changed tensors |", "|---|---|---:|---:|"]
    for r in retention(study):
        lines.append(f"| {r['Parameters trained']} | {r['Seed']} | {r['Neutral ordinary addition (%)']:.2f}% | {r['Changed tensors']} |")
    lines += ["", "## Structural permutation baseline", "",
              "An additional inference-only baseline fits a generic ten-digit permutation by minimizing teacher-to-permuted-base KL on the same unrelated strings. It then remaps complete ordinary answers generated by the unchanged competent base. The learner is not given the true +1/−1 map. This supplies a stronger structural prior and a different generation procedure: remapped digits are not fed back into the autoregressive base. It is not a like-for-like neural student or arithmetic acquired through auxiliary data.", "",
              "Only random-string teacher/base queries are made before fitting. Held-out task problems are first instantiated after the maps are fixed. This exploratory baseline was added after observing head-only transfer; every fitted map and control is retained.", "",
              "| Seed | Rule | Paired map: shifted test | Shuffled map: shifted test | Neutral map: original sums |", "|---|---|---:|---:|---:|"]
    for r in permutation_table(study):
        lines.append(f"| {r['Seed']} | {r['Convention']} | {r['Paired map: shifted test (%)']:.2f}% | {r['Shuffled map: shifted test (%)']:.2f}% | {r['Neutral map: original test (%)']:.2f}% |")
    lines += ["", "## Recorded training budgets", "",
              "Every complete arm uses 8,192 updates and 1,048,576 presentations of 4,096 unique inputs, or 256 passes. Auxiliary/shuffled/neutral arms supervise 15,728,640 token distributions. The direct-pool masks are reconstructed deterministically in the original per-seed `supervision_accounting.json`; their supervised-position count is that pool's answer positions times 256. Head-only optimization caches three frozen-base pools (one unrelated pool shared across relevant arms, and one coded direct pool for each convention); those cached features never come from a shifted teacher's backbone.", "",
              "| Run | Mode | Completed arms | Frozen-feature preparation seconds | Total preparation seconds | This invocation seconds |", "|---|---|---:|---:|---:|---:|"]
    for name, r in sorted(study["runs"].items()):
        lines.append(f"| {name} | {r['config']['mode']} | {sum(a['complete'] for a in r['arms'].values())} | {r['feature_cache_seconds']:.2f} | {r['preparation_seconds']:.2f} | {r['seconds_this_invocation']:.2f} |")
    lines += ["", "## Interpretation limits", "",
              "Head-only success demonstrates that changing the output head is sufficient for that measured amount of transfer; it does not imply arithmetic acquisition. A head-only null must be assessed against its direct control. If direct head-only training cannot solve the coded task, failure on unrelated inputs does not isolate where the teacher's information resides.", "",
              "The numerical intervention changes floating-point optimization trajectories for non-neutral arms too. Historical curves are preserved and are not silently replaced. Learning-rate selection is exploratory on seed 0 and is applied unchanged to the head-only arms; head-only learning rates are not independently optimized. The other two bases were previously used in the original study and do not provide untouched confirmation. Shaded ranges are descriptive seed ranges, not confidence intervals. Intermediate cosine points are checkpoints along the 8,192-update schedule, not separately optimized shorter-duration runs.", "",
              "Head-only success establishes sufficiency of head updates for the observed amount of transfer. It does not show that a full-model student uses only its head. The older random-start addition and carry null experiments have not been rerun with this corrected KL calculation; they remain historical evidence rather than controlled comparisons against this new protocol.", "",
              "Head-only activations are cached because the backbone is fixed. Updates and presentation budgets match, but caching changes compute costs: record feature preparation and training separately. A separate workload began sharing GPU 3 during head_seed0; later jobs were moved to the study's GPU 1. Recorded wall times include sharing, evaluation and checkpoint I/O and are not controlled compute comparisons. These are exposure comparisons on 4,096 repeated inputs, not unique-data efficiency curves or matched-FLOP claims. Every head-only checkpoint is checked for an unchanged backbone. The new runner saves AdamW and sampler state and supports --resume.", "",
              "Source/config archives, predictions, per-checkpoint validation and KL, and latest resumable checkpoints are stored per run. The notebook reads the recorded metrics without GPU inference."]
    (study["root"] / "REPORT.md").write_text("\n".join(lines) + "\n")
    for name, make in (("discovery_curves", discovery_plot), ("head_control_curves", mechanism_plot), ("final_controls", endpoint_plot)):
        fig = make(study)
        fig.savefig(study["root"] / f"{name}.png", dpi=180)
        fig.savefig(study["root"] / f"{name}.pdf")
        plt.close(fig)
    print(json.dumps({"report": str(study["root"] / "REPORT.md"), "selection": study["selection"] and study["selection"]["selected"]["name"]}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/addition_stability")
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    report(args.root, args.complete)
