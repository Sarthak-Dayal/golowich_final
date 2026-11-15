"""Summarize one exploratory shared-base run without selecting a new checkpoint."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(root):
    root = Path(root)
    d = json.loads((root / "results.json").read_text())
    text = ["# Shared-base carry-transfer discovery pilot", "", f"Status: **{d['status']}**. One exploratory training seed; no population-level transfer claim.", "", "The common base was trained on no-carry addition before the teacher learned ordinary addition. Student transfer arms receive ordinary token distributions on random digit strings only. Any improvement therefore concerns incremental carry learning from a competent shared base, not addition acquired from scratch.", "", f"Process time: {d.get('seconds', 0):.1f} seconds. Model: {d['parameters']:,} parameters; width {d['config']['width']}, {d['config']['layers']} layers.", "", "## Solvability gates", "", f"Base gate passed: {d.get('base_gate_passed', False)}; stop: {d.get('base_stop')}. Teacher gate passed: {d.get('teacher_gate_passed', False)}; stop: {d.get('teacher_stop', 'not reached')}.", ""]
    for stage in ["base", "teacher"]:
        curve = d.get(stage+"_curve", [])
        if curve:
            last = curve[-1]
            text.append(f"Last {stage} validation: {100*last['validation']['no_carry']:.2f}% no-carry, {100*last['validation']['carry']:.2f}% carry at {last['step']:,} updates.")
            text.append("")
    text += ["## Fixed-checkpoint final tests", "", "| Model | No-carry exact match | Carry exact match | Student presentations |", "|---|---:|---:|---:|"]
    for stage in ["base", "teacher"]:
        if stage+"_test" in d:
            q = d[stage+"_test"]
            text.append(f"| {stage} | {100*q['no_carry']:.2f}% | {100*q['carry']:.2f}% | — |")
    for arm, row in d.get("arms", {}).items():
        q = row["final_test"]
        text.append(f"| {arm} | {100*q['no_carry']:.2f}% | {100*q['carry']:.2f}% | {row['actual_exposures']:,} |")
    text += ["", "Each final-test subset contains 512 exact four-digit problems from the original symmetric hash-test partition. Whole-answer accuracy includes EOS; all checkpoints were selected using validation only.", ""]
    arms = d.get("arms", {})
    if "transfer" in arms and "neutral" in arms:
        transfer, neutral = arms["transfer"], arms["neutral"]
        gap = 100*(transfer["final_test"]["carry"]-neutral["final_test"]["carry"])
        if transfer["actual_exposures"] == neutral["actual_exposures"]:
            text += [f"Transfer-minus-neutral carry difference: **{gap:+.2f} percentage points**, at matched presentation budgets. This is one discovery seed, not a confidence interval or a confirmed effect.", ""]
            retained = 100*transfer["final_test"]["no_carry"]
            text += [f"The transfer student retains {retained:.2f}% no-carry accuracy. Assess any carry gain alongside loss of the base's existing skill; this pilot does not establish reliable or broadly useful carry transfer.", ""]
        else:
            text += ["Student budgets are incomplete and unequal; no matched-budget transfer-minus-neutral endpoint is reported.", ""]
    if d["status"] in ["stopped_base_gate", "stopped_teacher_gate"]:
        text += ["The predeclared solvability gate failed. Student transfer was not attempted; this result is inconclusive about incremental carry transfer.", ""]
    elif d["status"] == "stopped_time_limit":
        text += ["The process-time limit stopped the pilot. Missing or incomplete arms must not be treated as zero outcomes.", ""]
    if (root / "auxiliary_fit.json").exists():
        fit = json.loads((root / "auxiliary_fit.json").read_text())
        text += ["## Held-out unrelated-input fit", "", f"A forward-only diagnostic uses {fit['n']:,} fresh random strings ({fit['positions']:,} distilled positions), independently generated after the fixed training endpoint. No parameters or settings are changed.", "", "| Model | KL to teacher, nats per position |", "|---|---:|"]
        text.extend(f"| {name} | {row['kl_to_teacher_nats_per_position']:.5f} |" for name, row in fit["models"].items())
        text += ["", "Low held-out AUX KL shows successful imitation of the unrelated-input outputs. It does not establish autonomous addition or carry-skill acquisition. Raw diagnostic: `auxiliary_fit.json`.", ""]
    text += ["## Interpretation and provenance", "", "- Base training conditions the original variable-length generator on no-carry examples, which changes the operand-length distribution. Evaluation uses exact four-digit subsets.", "- The shared base must reach >=95% no-carry validation with <=75% carry validation at two checks. The teacher must reach >=95% on both subsets at two checks.", "- Students begin at the actual pre-specialization base snapshot; no trained teacher tensors are copied into transfer students. Teacher, base and evaluation streams remain frozen during distillation.", "- AUX training excludes PLUS/EQUAL operators, arithmetic prompts and answer labels. The direct arm intentionally includes target-task examples and correct answer prefixes.", "- A fixed 16,384-example cache is repeated. Unique observations, presentations and distilled positions are separate; this is not a completed unique-data sweep or an information-matched sample-efficiency comparison.", "- Targets for the full cache are generated upfront. Shuffled early checkpoints may access teacher targets from outside their early input prefix; these are student-exposure checkpoints, not unique-teacher-query budgets.", f"- Shared-base state hash: `{d.get('shared_base_sha256', 'unavailable')}`.", f"- Frozen teacher unchanged after distillation: `{d.get('teacher_unchanged', 'not reached')}`.", "- Raw measurements: `results.json`; exact evaluation pairs: `evaluation_pairs.json`; source snapshots and checkpoints are retained alongside them.", "- Settings and continuation decisions must be frozen before fresh-seed confirmation; do not select a follow-up from this final-test score.", ""]
    (root / "REPORT.md").write_text("\n".join(text))
    if arms:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        colors = {"transfer":"#1764ab", "neutral":"#666666", "shuffled":"#bd7f1f", "direct":"#228257"}
        for ax, mode in zip(axes, ["no_carry", "carry"]):
            for arm, row in arms.items():
                curve = row["curve"]
                ax.plot([p["exposures"] for p in curve], [100*p["validation"][mode] for p in curve], marker="o", label=arm, color=colors[arm])
            for stage, style in [("base", ":"), ("teacher", "--")]:
                curve = d.get(stage+"_curve", [])
                if curve:
                    ax.axhline(100*curve[-1]["validation"][mode], color="#222222", linestyle=style, alpha=.6, label=stage)
            ax.set(xscale="log", ylim=(-2, 102), xlabel="Training presentations", ylabel="Validation exact match (%)", title=mode.replace("_", " "))
            ax.grid(alpha=.15)
        axes[-1].legend(frameon=False, fontsize=8)
        fig.suptitle("Shared-base addition: one exploratory seed")
        fig.savefig(root / "learning_curves.png", dpi=180)
        fig.savefig(root / "learning_curves.pdf")
        plt.close(fig)
    print(root / "REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    summarize(parser.parse_args().root)
