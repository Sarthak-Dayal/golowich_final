# First addition transfer experiment: measured results

This pilot keeps the four-digit task fixed and tests ordinary token-output distillation on random digit strings. It uses three existing teacher/base initialization pairs. The settings are not a fresh confirmatory cohort selected independently of the baseline study.

## Primary result

| Seed | Auxiliary-only | Neutral teacher | Shuffled outputs | Direct target distillation |
|---|---|---|---|---|
| 0 | 0.00% | 0.00% | 0.00% | 100.00% |
| 1 | 0.00% | 0.00% | 0.00% | 99.61% |
| 2 | 0.00% | 0.00% | 0.00% | 100.00% |

At the fixed final budget of **16,384 unique examples and 524,288 presentations (32 passes)**, mean exact-match accuracies are auxiliary-only **0.00%**, neutral **0.00%**, shuffled **0.00%**, and direct **99.87%**. Mean teacher accuracy on the same final evaluation set is **99.87%**.

Final tests contain 256 held-out four-digit problems per seed, shared across seeds. Whole-answer accuracy includes EOS and uses greedy generation. The primary endpoint is fixed at the final exposure budget, not selected by the best validation checkpoint. All three paired auxiliary-minus-neutral effects are zero in this sample. This does not establish an exactly zero population effect or impossibility of transfer; the test is finite and there are only three training replicas. No Gaussian interval or sample-complexity ratio is claimed from these zero endpoints.

## What was trained

- Students begin at the exact random initializations of their teachers. The 817,679-parameter causal addition transformer and ordinary trainable token head are unchanged.
- Auxiliary inputs: BOS followed by 15 random digits. PLUS/EQUAL, arithmetic prompts, and oracle sums are excluded. Match the teacher's output distributions at the 15 digit positions. Outputs are soft logits over the original 15-token vocabulary, not hard sampled text.
- Neutral: match untrained-base outputs on the same inputs. Shuffled: permute teacher output sequences across inputs, preserving positional marginal distributions.
- Direct: match teacher outputs on actual addition examples (operand lengths up to four) with correct answer prefixes; loss only on answer digits/EOS. This is intentionally explicit task supervision. Unique-pair rejection reweights short versus long operand combinations compared with the original streaming teacher data.
- All arms: AdamW 0.0003, zero weight decay, batch 128, gradient clipping 1, no dropout. KL is averaged over supervised positions. The same cached input pool is revisited with reshuffled order each pass.
- Exposure budgets: 4,096; 16,384; 65,536; 262,144; 524,288. The unique-data budget stops growing after one pass. This is an exposure/optimization experiment, not a completed unique-data sweep.
- Four-digit validation (128 problems) supplies the plotted curves. The disjoint final test is evaluated after each arm reaches the fixed budget. Hash splits prevent swapped operands crossing partitions. Validation and test sampling use independent random streams.
- All forward batches have padded length 16. Distilled positions differ: auxiliary examples supply 15 distributions; direct examples supply fewer answer distributions. Raw per-position and unpadded-token counts are in curves.csv. Padded forward-token count at the endpoint is 8,388,608 per arm. Equal example counts are not equal supervision or transmitted information.
- All teacher parameter hashes remained unchanged. Source snapshots, configs, student checkpoints and raw metrics are retained in each pilot directory.

## Distillation fit versus target competence

Mean KL to the teacher on 1,024 fresh auxiliary sequences is **0.0246 nats per position** for the auxiliary students. Training KL falls substantially, but matching this carrier does not produce complete correct sums. Inspect these domain-specific losses separately from arithmetic exact match; low auxiliary loss is not evidence that the addition algorithm transferred.

Neutral targets represent the base itself, but finite-precision cached/no-grad versus gradient-enabled forwards can cause small initial gradients and parameter drift under Adam. Parameter displacement and neutral KL are logged. Native arithmetic accuracy remains the relevant matched negative control; the neutral arm should not be described as parameter-identical to the base.

## Teacher-assisted interface diagnostic

The following table restores both teacher operator-token embeddings (BOS, PLUS, EQUAL) and teacher output-head weights. These are privileged teacher parameters and are not autonomous transfer results.

| Seed | Restored base | Restored auxiliary student | Restored neutral student |
|---|---|---|---|
| 0 | 0.00% | 0.00% | 0.00% |
| 1 | 0.00% | 0.00% | 0.00% |
| 2 | 0.00% | 0.00% | 0.00% |

The complete diagnostic also tests operator-only and head-only restoration, teacher-forced arithmetic token accuracy/KL, and teacher evaluation with base operator embeddings. Teachers with base operator embeddings retain mean **99.87%** exact match. Teacher-forced answer-token accuracy is 15.68% for auxiliary students versus 6.65% for neutral students. This supplies correct answer prefixes and is a separate diagnostic; it is not free-running addition success. These interventions test limited interface explanations; they do not establish that all possible decoders would fail to recover information.

## Decision

Keep ordinary addition as the working task. This carrier has not demonstrated auxiliary-only addition transfer. Do not increase operand count or fit a scaling law to this result. First test a carrier with digit-position resets and numerical group structure while excluding addition operators and arithmetic labels, or a shared competent formatting base. Predeclare the next variant and retain these null results rather than replacing them. Use the existing MNIST auxiliary-logit result as a positive calibration alongside new addition variants.

Before declaring any carrier ineffective, require high teacher accuracy and a working direct-distillation control. If the direct arm is insufficient, extend or repair that control first. A broader unique-data × exposure grid and fresh-seed confirmation follow a promising carrier, not precede it.

## Artifacts

- `pilot_seed*/results.json`: per-seed curves, complete configuration, costs, and sample generations.
- `pilot_seed*/interface_diagnostic.json`: native and teacher-assisted results, plus held-out losses.
- `curves.csv`, `summary.json`, `learning_curves.png`, `learning_curves.pdf`.
- `experiments/addition/transfer.py`, `diagnose_transfer.py`, `check_transfer.py`, and `TRANSFER_PROTOCOL.md`.

All new artifacts remain local. Existing composition, arithmetic baselines, and flow results are preserved.
