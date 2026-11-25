# Addition transfer follow-ups: duration, exact initialization, and capacity

These are measured, bounded, exploratory tests on model seed 0. They respond to three hypotheses: more training is needed; initialization was mismatched; or excess capacity allows fitting the unrelated data without learning addition. They are not a new three-seed confirmatory study.

## How long the original students trained

The original auxiliary students saw 16,384 distinct strings for 32 passes: 524,288 presentations, 4,096 optimizer updates, and 7,864,320 supervised token positions. Each auxiliary arm took approximately 76 seconds on a Quadro RTX 6000. That was not a long wall-clock run, and the result applied only to that budget and carrier.

## Separate sensitivity tests

| Condition | Distinct inputs | Presentations | Updates | Four-digit exact test accuracy |
|---|---|---|---|---|
| Original fixed strings | 16,384 | 524,288 | 4,096 | 0.00% |
| Same strings, 4x training | 16,384 | 2,097,152 | 16,384 | 0.00% |
| Fresh plain strings | 524,288 | 524,288 | 4,096 | 0.00% |
| Fresh grouped strings | 524,288 | 524,288 | 4,096 | 0.39% |
| Narrower transformer, fixed strings | 16,384 | 524,288 | 4,096 | 0.00% |

All test endpoints use the same 256 held-out four-digit addition pairs, greedy generation, and complete-answer exact match including EOS. They are single-seed sensitivity results; do not infer a population improvement from the grouped-input result. One correct answer out of 256 is not a reliable transfer demonstration. The original fixed-string result was zero across all three prior seeds; this table shows its seed-0 reference for the paired sensitivities.

### More time

The 128-pass run reruns the same initial state, optimizer, fixed dataset and order from scratch rather than resuming without optimizer state. Its first 32-pass loss matches the original seed-0 run. It ends after 2,097,152 presentations and 16,384 updates, four times the original budget. Last-batch auxiliary KL drops from 0.01483 at 32 passes to 0.00618 at 128 passes. Whole-sum accuracy does not improve in this run. This does not rule out other schedules, learning rates, or much longer budgets.

### More data versus a broader format

Fresh plain strings provide 524,288 distinct inputs at the original 4,096 updates, rather than reusing 16,384. The input family remains iid digits. This tests data support rather than multiple semantic tasks.

Grouped inputs additionally insert PAD separators between random digit groups, changing the digit-position resets. They exclude PLUS and EQUAL and supply no arithmetic labels. PAD is a token here, not an attention mask. This is still unrelated-output distillation, but it changes the teacher's input distribution. It is not the original paper's sampled number-completion setup.

### Exact initial weights

Every one of the 54 state tensors in each old reference checkpoint matches an independent reconstruction from that run's exact stored source and initialization seed. Students were deep copies of those loaded references. No mismatch was found.

The old initial checkpoints were reconstructed rather than captured during teacher training. For the narrower run, the initial weights were physically saved before any teacher update. Every transfer/control student is checked against that snapshot before training, tensor-for-tensor and by a state hash. All assertions passed. The shared initial-state hash is `7a04003efb8b4b09f47f3e8ad0481610614a0882c01c46038652808735442ae1`. This compares the student's start to the teacher's original start, not to the teacher's trained final weights.

### Smaller capacity

Reduce width 128 to 64 while keeping four layers and four heads. Parameters fall from 817,679 to 212,239. The narrower teacher first trained for 8,000 updates; it only crossed the strict validation threshold at the last checkpoint. A separate 1,000-update refinement with a restarted AdamW optimizer at 0.0001 confirmed >=99% validation at both scheduled checks. Its total preparation therefore differs from the larger teacher's preparation and must not be treated as an equal-compute teacher comparison.

Narrow teacher test accuracy: **100.00%**.

| Narrow-model student condition | Exact test accuracy |
|---|---|
| transfer | 0.00% |
| shuffled | 0.00% |
| neutral | 0.00% |
| direct | 98.05% |

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

## Source formatting provenance

Python snapshots were reformatted after execution. Each run now includes `executed_source.txt`, which preserves the exact launched text and matches the original recorded source hash. AST equality against the formatted `source.py` was verified; details are in `source_provenance.json`. Metrics and original hash fields were preserved.
