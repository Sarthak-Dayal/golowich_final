"""Report bounded duration/diversity/capacity follow-ups without pooling their seeds."""

import hashlib, json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("results/addition_followup")
base = json.loads(
    Path("results/addition_transfer/pilot_seed0/results.json").read_text()
)
long = json.loads((root / "long_fixed_seed0/results.json").read_text())
fresh = json.loads((root / "fresh_seed0/results.json").read_text())
group = json.loads((root / "grouped_seed0/results.json").read_text())
audit = json.loads((root / "initialization_audit.json").read_text())
parts = [
    json.loads((root / p / "results.json").read_text())
    for p in ["small_carrier", "small_neutral", "small_direct"]
]
assert all(p["teacher_unchanged"] and p["exact_initialization_asserted"] for p in parts)
assert len({p["initial_state_sha256"] for p in parts}) == 1
small = {a: v for p in parts for a, v in p["arms"].items()}
assert set(small) == {"transfer", "neutral", "shuffled", "direct"}
teacher = json.loads(
    Path("results/addition/small_refined_seed0_digits4/results.json").read_text()
)
conditions = [
    dict(
        name="Original fixed strings",
        unique=16384,
        exposures=524288,
        updates=4096,
        test=base["arms"]["transfer"]["final_test"]["exact_match"],
        seeds=1,
    ),
    dict(
        name="Same strings, 4x training",
        unique=long["curve"][-1]["unique_inputs"],
        exposures=long["curve"][-1]["exposures"],
        updates=long["curve"][-1]["updates"],
        test=long["final_test"]["exact_match"],
        seeds=1,
    ),
    dict(
        name="Fresh plain strings",
        unique=fresh["curve"][-1]["unique_inputs"],
        exposures=fresh["curve"][-1]["exposures"],
        updates=fresh["curve"][-1]["updates"],
        test=fresh["final_test"]["exact_match"],
        seeds=1,
    ),
    dict(
        name="Fresh grouped strings",
        unique=group["curve"][-1]["unique_inputs"],
        exposures=group["curve"][-1]["exposures"],
        updates=group["curve"][-1]["updates"],
        test=group["final_test"]["exact_match"],
        seeds=1,
    ),
    dict(
        name="Narrower transformer, fixed strings",
        unique=16384,
        exposures=524288,
        updates=4096,
        test=small["transfer"]["final_test"]["exact_match"],
        seeds=1,
    ),
]
summary = dict(
    status="MEASURED exploratory single-seed sensitivities; do not pool with original three-seed study",
    initialization_audit=audit,
    conditions=conditions,
    small_model_parameters=teacher["parameters"],
    small_teacher_test=teacher["test"]["4"]["exact_match"],
    small_conditions={a: v["final_test"]["exact_match"] for a, v in small.items()},
    small_initial_state_sha256=parts[0]["initial_state_sha256"],
)
(root / "summary.json").write_text(json.dumps(summary, indent=2))
rows = "\n".join(
    f"| {r['name']} | {r['unique']:,} | {r['exposures']:,} | {r['updates']:,} | {100 * r['test']:.2f}% |"
    for r in conditions
)
srows = "\n".join(
    f"| {a} | {100 * v['final_test']['exact_match']:.2f}% |" for a, v in small.items()
)
plt.rcParams.update(
    {"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
)
fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
ax[0].plot(
    [v["exposures"] for v in long["curve"]],
    [v["last_loss"] for v in long["curve"]],
    "o-",
    color="#18594f",
)
ax[0].set(
    xlabel="Presentations of the same 16,384 strings",
    ylabel="Last-batch auxiliary KL per position",
    title="Longer training improves carrier fit",
    xscale="log",
)
ax[0].grid(alpha=0.15)
for a, color in [
    ("transfer", "#18594f"),
    ("neutral", "#85938a"),
    ("shuffled", "#b0914d"),
    ("direct", "#ba7953"),
]:
    vals = small[a]["curve"]
    ax[1].plot(
        [v["exposures"] for v in vals],
        [100 * v["validation_exact_match"] for v in vals],
        "o-",
        label=a,
        color=color,
    )
ax[1].set(
    xlabel="Example presentations",
    ylabel="Whole-answer validation accuracy (%)",
    title="212k-parameter model · one seed",
    xscale="log",
    ylim=(-2, 103),
)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.15)
fig.savefig(root / "followup_curves.png", dpi=180)
fig.savefig(root / "followup_curves.pdf")
plt.close(fig)
report = f"""# Addition transfer follow-ups: duration, exact initialization, and capacity

These are measured, bounded, exploratory tests on model seed 0. They respond to three hypotheses: more training is needed; initialization was mismatched; or excess capacity allows fitting the unrelated data without learning addition. They are not a new three-seed confirmatory study.

## How long the original students trained

The original auxiliary students saw 16,384 distinct strings for 32 passes: 524,288 presentations, 4,096 optimizer updates, and 7,864,320 supervised token positions. Each auxiliary arm took approximately 76 seconds on a Quadro RTX 6000. That was not a long wall-clock run, and the result applied only to that budget and carrier.

## Separate sensitivity tests

| Condition | Distinct inputs | Presentations | Updates | Four-digit exact test accuracy |
|---|---|---|---|---|
{rows}

All test endpoints use the same 256 held-out four-digit addition pairs, greedy generation, and complete-answer exact match including EOS. They are single-seed sensitivity results; do not infer a population improvement from the grouped-input result. One correct answer out of 256 is not a reliable transfer demonstration. The original fixed-string result was zero across all three prior seeds; this table shows its seed-0 reference for the paired sensitivities.

### More time

The 128-pass run reruns the same initial state, optimizer, fixed dataset and order from scratch rather than resuming without optimizer state. Its first 32-pass loss matches the original seed-0 run. It ends after 2,097,152 presentations and 16,384 updates, four times the original budget. Last-batch auxiliary KL drops from {long["curve"][0]["last_loss"]:.5f} at 32 passes to {long["curve"][-1]["last_loss"]:.5f} at 128 passes. Whole-sum accuracy does not improve in this run. This does not rule out other schedules, learning rates, or much longer budgets.

### More data versus a broader format

Fresh plain strings provide 524,288 distinct inputs at the original 4,096 updates, rather than reusing 16,384. The input family remains iid digits. This tests data support rather than multiple semantic tasks.

Grouped inputs additionally insert PAD separators between random digit groups, changing the digit-position resets. They exclude PLUS and EQUAL and supply no arithmetic labels. PAD is a token here, not an attention mask. This is still unrelated-output distillation, but it changes the teacher's input distribution. It is not the original paper's sampled number-completion setup.

### Exact initial weights

Every one of the 54 state tensors in each old reference checkpoint matches an independent reconstruction from that run's exact stored source and initialization seed. Students were deep copies of those loaded references. No mismatch was found.

The old initial checkpoints were reconstructed rather than captured during teacher training. For the narrower run, the initial weights were physically saved before any teacher update. Every transfer/control student is checked against that snapshot before training, tensor-for-tensor and by a state hash. All assertions passed. The shared initial-state hash is `{parts[0]["initial_state_sha256"]}`. This compares the student's start to the teacher's original start, not to the teacher's trained final weights.

### Smaller capacity

Reduce width 128 to 64 while keeping four layers and four heads. Parameters fall from 817,679 to {teacher["parameters"]:,}. The narrower teacher first trained for 8,000 updates; it only crossed the strict validation threshold at the last checkpoint. A separate 1,000-update refinement with a restarted AdamW optimizer at 0.0001 confirmed >=99% validation at both scheduled checks. Its total preparation therefore differs from the larger teacher's preparation and must not be treated as an equal-compute teacher comparison.

Narrow teacher test accuracy: **{100 * teacher["test"]["4"]["exact_match"]:.2f}%**.

| Narrow-model student condition | Exact test accuracy |
|---|---|
{srows}

Transfer conditions retain the same 16,384 inputs, 524,288 presentations, optimizer and fixed four-digit evaluation. The direct control determines whether the smaller student can learn addition under the distillation protocol. A single width comparison cannot identify an overparameterization mechanism: width also changes representation geometry and optimization.

## Interpreting the user's pretraining hypothesis

A model pretrained on many tasks differs from a smaller random model in representation, parameter displacement during teacher training, tokenizer, output behavior, and training history. Reducing width alone does not test that hypothesis. A shared multi-task base or pretrained SLM remains an untested, separate intervention. Its neutral teacher and base performance would need to be measured before calling any gain new capability transfer.

## Relation to the original paper

We retained ordinary-logit distillation for this transformer experiment, consistent with the user's clarification. The original paper's LLM setup uses sampled, filtered teacher completions from a common pretrained model. Its MLP example uses separate auxiliary outputs on noise. Those are two distinct experiments; copying the MLP head arrangement is not necessary to make our transformer test legitimate. Our current random-input soft-logit experiment is still an adaptation, not an exact reproduction of the LLM setting.

Original references inspected: [Cloud et al.](https://arxiv.org/html/2507.14805v1), Sections 3 and 6.2; [authors' MNIST code](https://github.com/MinhxLe/subliminal-learning/blob/main/scripts/run_mnist_experiment.py). The paper's main number-sequence setting uses 10,000 retained examples over 10 epochs; epoch counts alone do not match input distribution, supervision, or compute across models.

## What the results support

The tested extra training, fresh data, and reduced width have not established useful autonomous addition transfer. Initialization mismatch is not supported by the audit. The results do not exclude a different carrier, a competent shared multi-task base, or a pretrained SLM. Keep the tested null conditions in the experiment log; choose the next base/carrier before collecting a new confirmation cohort.

## Artifacts and reproduction

- `initialization_audit.json`, `summary.json`, `followup_curves.png`, `.pdf`.
- `long_fixed_seed0/`, `fresh_seed0/`, `grouped_seed0/`: source snapshots, raw metrics, ignored checkpoints.
- `small_carrier/`, `small_neutral/`, `small_direct/`: independently executed matched conditions, all sharing the verified initial-state hash. They are separate processes, not separate model seeds.
- Narrow teacher preparation: `results/addition/small_seed0_digits4/` and `small_refined_seed0_digits4/`.
- Protocol and scripts in `experiments/addition/FOLLOWUP_PROTOCOL.md`, `followup_carriers.py`, `audit_initialization.py`, `refine_teacher.py`, and `transfer.py`.

All artifacts remain local. Existing composition, addition, and flow results were preserved.
"""
(root / "REPORT.md").write_text(report)
print(json.dumps(summary, indent=2))
