"""Render the fixed exploratory calibration without selecting or extending runs."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from statistics import mean


def pct(v):
    return f"{100 * v:.2f}%"


def supervision_counts(p, r):
    """Reconstruct deterministic input masks only, using the saved model source."""
    path = p / "supervision_accounting.json"
    if path.exists():
        return json.loads(path.read_text())
    c = r["config"]
    source = p / "addition_source.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == r["model_source_sha256"]
    spec = importlib.util.spec_from_file_location("trait_saved_addition_for_accounting", source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rng, pairs, seen = random.Random(13000 + c["seed"]), [], set()
    while len(pairs) < c["n"]:
        for pair in m.sample(rng, min(256, c["n"] - len(pairs)), 4, "train"):
            key = tuple(sorted(pair))
            if key not in seen:
                seen.add(key)
                pairs.append(pair)
    positions = sum(len(m.answer(*pair)) for pair in pairs)
    a = {"method": "Reconstructed fixed direct input pool from saved source/config and deterministic data seed; no model evaluation or optimization.",
         "direct_pool_unique_pairs": len(pairs), "direct_pool_answer_positions": positions,
         "direct_positions_per_input_mean": positions / len(pairs), "full_cache_passes": c["exposures"] // c["n"],
         "auxiliary_distilled_positions_per_arm": 15 * c["exposures"],
         "direct_distilled_positions_per_arm": positions * (c["exposures"] // c["n"]),
         "padded_forward_tokens_per_arm": 16 * c["exposures"], "padded_input_length": 16}
    path.write_text(json.dumps(a, indent=2) + "\n")
    return a


def main(root):
    p = Path(root)
    r = json.loads((p / "results.json").read_text())
    c = r["config"]
    supervision_counts(p, r)
    lines = [
        "# Output-convention transfer calibration",
        "",
        f"Status: **{r['status']}**. One exploratory model seed, shared competent addition base; no inference across training seeds.",
        "",
        "This experiment tests transfer of a newly learned digit-output convention. The student already knows ordinary addition before distillation. It is not evidence of learning addition from scratch or transferring a new carry algorithm.",
        "",
        f"Fixed protocol: {c['teacher_steps']:,} updates per teacher, learning rate {c['teacher_lr']}; {c['n']:,} cached unrelated strings; {c['exposures']:,} student presentations, batch {c['batch']}, learning rate {c['student_lr']}. Normal prompts and teacher-forced coded answer prefixes are used only for teacher training and the explicit direct controls. Auxiliary students see BOS plus 15 random digits and ordinary soft token distributions, with no addition operators or arithmetic answers.",
        "",
        "The +1 teacher shifts each answer digit up one modulo 10; the −1 teacher shifts each down one. EOS remains unchanged. All seven student arms reset to the exact same competent base snapshot. No auxiliary output head is added.",
        "",
        "## Teacher validation gates",
        "",
        "| Teacher | Own coded validation accuracy | Gate passed |",
        "|---|---:|---|",
    ]
    for shift, t in r["teachers"].items():
        lines.append(f"| {'+1' if shift == '1' else '−1'} | {pct(t['validation']['exact'][shift])} | {t['gate_passed']} |")
    if not all(t["gate_passed"] for t in r["teachers"].values()) or len(r["teachers"]) != 2:
        lines += ["", "Teacher solvability did not pass for both conventions. Student transfer is therefore not interpreted. The bounded failed preparation is retained."]
    if "base_test" in r:
        lines += ["", "## Fixed-endpoint final test", "",
                  f"Each accuracy uses {r['base_test']['n']} hash-held-out four-digit operand pairs and greedy free generation of the complete answer including EOS. No test-based checkpoint or condition selection was performed.", "",
                  "| Model | Original sums | +1 convention | −1 convention |",
                  "|---|---:|---:|---:|"]
        rows = [("Competent base", r["base_test"])]
        rows += [("Teacher " + ("+1" if s == "1" else "−1"), t["final_test"]) for s, t in r["teachers"].items() if "final_test" in t]
        rows += [(name, a["final_test"]) for name, a in r["arms"].items() if "final_test" in a]
        for name, score in rows:
            lines.append(f"| {name} | " + " | ".join(pct(score["exact"][s]) for s in ("0", "1", "9")) + " |")
        complete = all(a.get("complete", False) for a in r["arms"].values()) and len(r["arms"]) == 7
        lines += ["", f"Complete student arms: {sum(a.get('complete', False) for a in r['arms'].values())}/7."]
        if complete:
            lines += ["", "## Interpretation", ""]
            for s in ("1", "9"):
                aux = r["arms"][f"transfer_{s}"]["final_test"]["exact"][s]
                shuffled = r["arms"][f"shuffled_{s}"]["final_test"]["exact"][s]
                direct = r["arms"][f"direct_{s}"]["final_test"]["exact"][s]
                neutral = r["arms"]["neutral_0"]["final_test"]["exact"][s]
                lines.append(f"For the {'+1' if s == '1' else '−1'} convention: unrelated-input student {pct(aux)}, neutral {pct(neutral)}, shuffled {pct(shuffled)}, direct {pct(direct)}. Auxiliary-minus-neutral difference {(aux - neutral) * 100:.2f} percentage points; auxiliary-minus-shuffled {(aux - shuffled) * 100:.2f} points.")
                lines.append("")
            lines += ["These are one-seed descriptive differences. Teacher-specific transfer requires the appropriate convention to follow its teacher; a shuffled result can reflect output marginals. The direct controls determine whether the convention can be transmitted within this student budget. Any positive result is an easier convention calibration on a pretrained arithmetic model, not a resolution of the failed random-start addition experiment."]
    lines += ["", "## Provenance and limits", "",
              f"Process time recorded by the run: {r['seconds']:.1f} seconds (timer starts after initial base validation). Teacher unchanged assertion: {r.get('teachers_unchanged', 'not reached')}. Shared base state SHA-256: `{r['initial_sha256']}`.", "",
              "`results.json` retains configuration, curves, predictions, hashes and all gates; `source.py` and `addition_source.py` retain the executed code, and `checkpoints/` retains the base, teachers and students. Data partitions use sorted operand pairs so swapped operands cannot cross the train/validation/test partition. Distinct arithmetic information and supervised position counts differ from the random-input arm, so presentation equality is not information equality."]
    if (p / "supervision_accounting.json").exists():
        a = json.loads((p / "supervision_accounting.json").read_text())
        lines += ["", f"Supervision accounting: each unrelated-input arm matches {a['auxiliary_distilled_positions_per_arm']:,} token distributions; each direct arm matches {a['direct_distilled_positions_per_arm']:,}. Every arm processes {a['padded_forward_tokens_per_arm']:,} padded forward tokens, with {a['full_cache_passes']} passes over its fixed pool. The direct pool contains {a['direct_pool_unique_pairs']:,} operand pairs and {a['direct_pool_answer_positions']:,} answer positions. Counts were reconstructed from the executed source/config and deterministic input generator without changing model outputs; details are in `supervision_accounting.json`."]
    (p / "REPORT.md").write_text("\n".join(lines) + "\n")
    if r["arms"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        colors = {"transfer": "#2171b5", "shuffled": "#969696", "direct": "#238b45"}
        for ax, shift in zip(axes, ("1", "9")):
            for kind in ("transfer", "shuffled", "direct"):
                a = r["arms"].get(f"{kind}_{shift}")
                if a and a["curve"]:
                    ax.plot([x["presentations"] for x in a["curve"]], [100 * x["validation"][shift] for x in a["curve"]], "o-", label=kind, color=colors[kind])
            a = r["arms"].get("neutral_0")
            if a and a["curve"]:
                ax.plot([x["presentations"] for x in a["curve"]], [100 * x["validation"][shift] for x in a["curve"]], "--", label="neutral", color="#cb181d")
            ax.set(title=("+1" if shift == "1" else "−1") + " answer convention", xlabel="Student presentations", xscale="log", ylim=(-2, 102))
            ax.grid(alpha=.2)
        axes[0].set_ylabel("Validation exact accuracy (%)")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"Exploratory seed {c['seed']}; addition already learned in shared base", fontsize=10)
        fig.tight_layout()
        fig.savefig(p / "learning_curves.png", dpi=170)
        fig.savefig(p / "learning_curves.pdf")
        plt.close(fig)
    print(p / "REPORT.md")


def aggregate(root):
    """Summarize all expected seed groups, retaining failed gates explicitly."""
    p = Path(root)
    runs = []
    for seed in (0, 1, 2):
        path = p / f"pilot_seed{seed}" / "results.json"
        if not path.exists():
            raise RuntimeError(f"Expected seed {seed} has no results yet; aggregation waits for all three groups")
        r = json.loads(path.read_text())
        if r["status"] == "in_progress":
            raise RuntimeError(f"Seed {seed} is still running; do not summarize a partial final endpoint")
        assert r["config"]["seed"] == seed
        complete = r["status"] == "completed" and len(r["arms"]) == 7 and all(a.get("complete", False) for a in r["arms"].values())
        base_gate = r["base_validation"]["exact"]["0"] >= 0.95
        teacher_gate = len(r["teachers"]) == 2 and all(t.get("gate_passed", False) for t in r["teachers"].values())
        runs.append({"seed": seed, "path": str(path), "r": r, "complete": complete, "base_gate": base_gate,
                     "teacher_gate": teacher_gate, "eligible": complete and base_gate and teacher_gate})
    # Config and executed-source checks prevent pooling a tuned follow-up silently.
    ignored_config = {"base", "out", "seed", "device"}
    ref = {k: v for k, v in runs[0]["r"]["config"].items() if k not in ignored_config}
    config_equal = all({k: v for k, v in x["r"]["config"].items() if k not in ignored_config} == ref for x in runs)
    source_equal = len({x["r"]["source_sha256"] for x in runs}) == 1
    model_source_equal = len({x["r"]["model_source_sha256"] for x in runs}) == 1
    if not (config_equal and source_equal and model_source_equal):
        raise RuntimeError("Protocols or executed sources differ; inspect before pooling results")

    def exact(score, shift):
        return None if not score else score["exact"][shift]

    rows = []
    for x in runs:
        r = x["r"]
        for shift in ("1", "9"):
            teacher = r["teachers"].get(shift, {})
            row = {"seed": x["seed"], "convention": "+1" if shift == "1" else "-1", "status": r["status"],
                   "eligible": x["eligible"], "base_validation_original": r["base_validation"]["exact"]["0"],
                   "teacher_validation": exact(teacher.get("validation"), shift),
                   "teacher_test": exact(teacher.get("final_test"), shift),
                   "base_test_original": exact(r.get("base_test"), "0")}
            for kind in ("transfer", "shuffled", "direct"):
                a = r["arms"].get(f"{kind}_{shift}", {})
                row[kind] = exact(a.get("final_test"), shift)
                row[kind + "_original"] = exact(a.get("final_test"), "0")
                row[kind + "_opposite"] = exact(a.get("final_test"), "9" if shift == "1" else "1")
            neutral = r["arms"].get("neutral_0", {})
            row["neutral"] = exact(neutral.get("final_test"), shift)
            row["neutral_original"] = exact(neutral.get("final_test"), "0")
            row["transfer_minus_shuffled_pp"] = None if row["transfer"] is None or row["shuffled"] is None else 100 * (row["transfer"] - row["shuffled"])
            row["transfer_minus_neutral_pp"] = None if row["transfer"] is None or row["neutral"] is None else 100 * (row["transfer"] - row["neutral"])
            rows.append(row)
    stats = {}
    for convention in ("+1", "-1"):
        use = [row for row in rows if row["convention"] == convention and row["eligible"]]
        stats[convention] = {}
        for key in ("teacher_test", "transfer", "shuffled", "neutral", "direct", "transfer_minus_shuffled_pp", "transfer_minus_neutral_pp"):
            vals = [row[key] for row in use if row[key] is not None]
            stats[convention][key] = {"n_training_seed_groups": len(vals), "mean": mean(vals) if vals else None,
                                       "min": min(vals) if vals else None, "max": max(vals) if vals else None}
    summary = {"study": "Exploratory output-convention transfer from already competent arithmetic bases",
               "n_expected_training_seed_groups": 3, "n_eligible_training_seed_groups": sum(x["eligible"] for x in runs),
               "test_inputs_per_model": 256, "same_test_inputs_across_seed_groups": True,
               "design": "Seed 0 prototype followed by two unchanged independent-base/data-seed replications; not an untouched confirmatory cohort.",
               "protocol_config_equal": config_equal, "executed_source_equal": source_equal, "model_source_equal": model_source_equal,
               "seed_groups": [{"seed": x["seed"], "path": x["path"], "status": x["r"]["status"],
                                 "base_gate": x["base_gate"], "teacher_gate": x["teacher_gate"], "complete": x["complete"],
                                 "eligible": x["eligible"], "seconds": x["r"]["seconds"],
                                 "initial_sha256": x["r"]["initial_sha256"], "teacher_unchanged": x["r"].get("teachers_unchanged", False)} for x in runs],
               "rows": rows, "descriptive_stats": stats}
    (p / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (p / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def display(v):
        return "unavailable" if v is None else pct(v)

    def effect(v):
        return "unavailable" if v is None else f"{v:+.2f} pp"

    c = runs[0]["r"]["config"]
    lines = ["# Output-convention transfer across three exploratory seed groups", "",
             "The base models already solve addition. This experiment asks whether an unrelated-input student acquires a teacher's new digit-output convention: +1 or −1 modulo 10 on each answer digit, with EOS unchanged. It does not test acquiring the arithmetic algorithm from random initialization.", "",
             "Seed 0 was an exploratory prototype. Its positive result motivated two unchanged replications using separately trained arithmetic bases and fresh teacher/student data seeds 1 and 2. These three groups remain an exploratory study, not an untouched confirmation cohort. The two conventions within a group share a base and are not independent training seeds.", "",
             f"Fixed protocol: two teachers per base, {c['teacher_steps']:,} updates each; {c['n']:,} cached unrelated inputs; {c['exposures']:,} student presentations ({c['exposures'] // c['n']} passes), {c['exposures'] // c['batch']} updates per arm. All student starts are exact saved competent-base clones. Ordinary token heads remain trainable; no auxiliary head or teacher parameters are supplied to students. Auxiliary inputs contain BOS plus 15 random digits and no addition operators or answer labels.", "",
             "## Gates and completion", "", "| Training seed group | Base original validation | +1 teacher validation | −1 teacher validation | Teacher gates | Completed student arms | Status |", "|---|---:|---:|---:|---|---:|---|"]
    for x in runs:
        r = x["r"]
        vals = [exact(r["teachers"].get(s, {}).get("validation"), s) for s in ("1", "9")]
        lines.append(f"| {x['seed']} | {display(r['base_validation']['exact']['0'])} | {display(vals[0])} | {display(vals[1])} | {x['teacher_gate']} | {sum(a.get('complete', False) for a in r['arms'].values())}/7 | {r['status']} |")
    if not all(x["eligible"] for x in runs):
        lines += ["", "Some seed groups failed a gate or stopped before all endpoints. They remain in the table; descriptive means below use only complete gate-passing groups and must not be read as an unconditional success rate."]
    lines += ["", "## Fixed-endpoint final test", "",
              "Every model is tested by greedy generation of the complete answer including EOS on the same 256 hash-held-out four-digit operand pairs. Test examples are not training replicas. The effective replication count is three independently trained base/teacher/student seed groups, not 768 test problems or six independent conventions.", "",
              "| Seed | Convention | Teacher | Auxiliary transfer | Shuffled | Neutral | Direct | Transfer − shuffled |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['convention']} | {display(row['teacher_test'])} | {display(row['transfer'])} | {display(row['shuffled'])} | {display(row['neutral'])} | {display(row['direct'])} | {effect(row['transfer_minus_shuffled_pp'])} |")
    lines += ["", "## Descriptive mean and range across training seed groups", "",
              "| Convention | Condition | Mean | Observed seed range | Groups |", "|---|---|---:|---:|---:|"]
    for convention, values in stats.items():
        for key in ("transfer", "shuffled", "neutral", "direct"):
            s = values[key]
            lines.append(f"| {convention} | {key} | {display(s['mean'])} | {display(s['min'])}–{display(s['max'])} | {s['n_training_seed_groups']} |")
        s = values["transfer_minus_shuffled_pp"]
        lines.append(f"| {convention} | Paired transfer − shuffled | {effect(s['mean'])} | {effect(s['min'])}–{effect(s['max'])} | {s['n_training_seed_groups']} |")
    for row in rows:
        if row["eligible"] and row["direct"] is not None and row["direct"] < .95:
            lines += ["", f"The seed-{row['seed']} {row['convention']} direct control ends at {display(row['direct'])} on the fixed final test. Earlier validation peaks do not replace that endpoint. There was no predeclared 95% direct-student gate, but this lower endpoint should remain visible when assessing this seed's calibration."]
    lines += ["", "No significance test or confidence interval is claimed from three groups. The observed seed range describes sensitivity; it is not a population confidence interval.", "",
              "## Original arithmetic and convention specificity", "",
              "| Seed | Base original | Neutral original | +1 auxiliary original | −1 auxiliary original | +1 student under −1 convention | −1 student under +1 convention |", "|---|---:|---:|---:|---:|---:|---:|"]
    for seed in (0, 1, 2):
        plus = next(row for row in rows if row["seed"] == seed and row["convention"] == "+1")
        minus = next(row for row in rows if row["seed"] == seed and row["convention"] == "-1")
        lines.append(f"| {seed} | {display(plus['base_test_original'])} | {display(plus['neutral_original'])} | {display(plus['transfer_original'])} | {display(minus['transfer_original'])} | {display(plus['transfer_opposite'])} | {display(minus['transfer_opposite'])} |")
    lines += ["", "The appropriate convention following its teacher is evidence of teacher specificity. Check its magnitude beside the direct controls and loss of original arithmetic behavior. A shuffled effect can reflect marginal output statistics; that possibility is measured rather than assumed absent.", "",
              "## Costs, provenance, and interpretation", "",
              f"Recorded process time totals {sum(x['r']['seconds'] for x in runs):.1f} seconds; these timers start after initial base validation. Model/optimizer preparation and cached teacher queries are included after that point. Exact run configurations, source snapshots, model/checkpoint/data hashes, predictions and gates are retained in each `pilot_seed*/` directory. Executed source and protocol settings match across groups apart from base/seed/device/output-path fields.", "",
              f"Each unrelated-input arm matches {15 * c['exposures']:,} token distributions and processes {16 * c['exposures']:,} padded forward tokens. Direct arms supervise only answer positions and contain different task information, so equal presentations are not equal supervision or information. The finite fixed input pool is reused; this is an exposure study, not a completed unique-data sweep.", "",
              "These observations characterize an easier output-convention change from an already competent base. They do not establish transfer of a newly learned carry algorithm, make the random-start addition experiment successful, or isolate pretraining as the sole cause: teacher task, starting competence and parameter displacement differ. Fresh-seed confirmation and a direct comparison holding the teacher task fixed would be needed for stronger claims.", "",
              "Artifacts: `summary.json`, `summary.csv`, `per_seed_results.png`, `per_seed_results.pdf`, and per-seed reports and curves."]
    lines += ["", "| Seed | Auxiliary positions per arm | Direct positions per arm | Padded forward tokens per arm |", "|---|---:|---:|---:|"]
    for x in runs:
        a = supervision_counts(p / f"pilot_seed{x['seed']}", x["r"])
        lines.append(f"| {x['seed']} | {a['auxiliary_distilled_positions_per_arm']:,} | {a['direct_distilled_positions_per_arm']:,} | {a['padded_forward_tokens_per_arm']:,} |")
    (p / "REPORT.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    kinds = ("transfer", "shuffled", "neutral", "direct")
    colors = ("#2171b5", "#238b45", "#cb181d")
    for ax, convention in zip(axes, ("+1", "-1")):
        for seed, color in zip((0, 1, 2), colors):
            row = next(row for row in rows if row["seed"] == seed and row["convention"] == convention)
            xs = [i + (seed - 1) * .09 for i, k in enumerate(kinds) if row[k] is not None]
            ys = [row[k] * 100 for k in kinds if row[k] is not None]
            ax.plot(xs, ys, "o", color=color, label=f"Training seed {seed}", markersize=5, alpha=.9)
        for i, kind in enumerate(kinds):
            s = stats[convention][kind]
            if s["mean"] is not None:
                ax.plot(i, s["mean"] * 100, "_", color="black", markersize=16, label="Mean" if i == 0 else None)
        ax.set_xticks(range(len(kinds)), ["Auxiliary", "Shuffled", "Neutral", "Direct"])
        ax.set_title(convention + " output convention")
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Final full-answer exact accuracy (%)")
    axes[1].legend(fontsize=8, loc="center right")
    fig.suptitle("Three exploratory training seed groups; 256 test inputs per model", fontsize=10)
    fig.tight_layout()
    fig.savefig(p / "per_seed_results.png", dpi=180)
    fig.savefig(p / "per_seed_results.pdf")
    plt.close(fig)
    print(p / "REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--all", action="store_true", help="Aggregate the three completed or gate-stopped seed groups")
    args = parser.parse_args()
    if args.all:
        aggregate(args.root)
    else:
        main(args.root)
