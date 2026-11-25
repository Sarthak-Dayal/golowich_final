# Output-convention calibration from a competent addition base

This is an exploratory seed-0 calibration, not a new arithmetic-learning claim. The shared base is the existing trained four-digit addition model (`results/addition/seed0_digits4/checkpoints/model.pt`). Its learned arithmetic is present before student distillation.

Train two teacher copies for a fixed 1,000 updates each, AdamW 1e-4, batch 128, no weight decay, on ordinary addition prompts with a changed answer convention. Teacher +1 emits each reversed answer digit increased by one modulo 10; teacher −1 emits each decreased by one. EOS is unchanged. For 37+58=95, the base emits `5 9 EOS`, teacher +1 should emit `6 0 EOS`, and teacher −1 `4 8 EOS`. Autoregressive answer prefixes during teacher/direct training use the coded digits. Hash-based original partitions prevent operand-pair leakage; swapped operands share a partition.

Require both fixed-endpoint teachers to reach ≥95% free-generation exact-match accuracy on their own coded four-digit validation task. Stop if either fails. Do not tune using the final test.

Every student starts from an exact, physically saved competent base snapshot, verified by a state hash. Use the ordinary trainable token head throughout. Cache 4,096 unrelated inputs, BOS plus 15 independent random digits, excluding +/= and arithmetic answers. Match full teacher distributions at the 15 digit positions with mean per-position KL, AdamW 3e-4, batch 128, zero weight decay, clipping 1. Run 65,536 presentations (16 passes/512 updates), evaluating at 4,096/16,384/65,536 presentations. This is an exposure curve; unique data stop increasing after 4,096.

Seven student conditions: one base-neutral; teacher+1 and teacher−1 transfer; separately shuffled teacher output sequences for each teacher; direct distillation from each teacher on task-related prompts and coded answer prefixes. All see matched presentation/update budgets; supervised position counts differ between direct and unrelated inputs. Teacher outputs are frozen; trained teacher parameters are never supplied to students. The fixed cache means the shuffle controls preserve teacher positional marginals.

Free-generate full answers including EOS. Evaluate each model against the original convention and both coded conventions. Opposing teacher/student preferences supply specificity; shuffled controls test whether output marginals explain a result. A positive finding would demonstrate output-convention transfer from a competent base, not learning the addition algorithm from random initialization. It does not resolve carry-skill transfer, which is a separate parallel intervention.

Resource limit: one seed, one assigned idle GPU, a 300-second process timer. Preserve partial results if the limit is reached. The pilot is local only. Teacher/student snapshots, dataset/source/checkpoint hashes, metrics, and failed gates are retained.

Commands:

```bash
.venv/bin/python experiments/addition_trait/run.py --check
.venv/bin/python experiments/addition_trait/run.py --device cuda:DEVICE --out results/addition_trait/pilot_seed0
```

No GPU training is authorized by this protocol text alone; launch follows the parent's device assignment and normal sandbox permissions.
