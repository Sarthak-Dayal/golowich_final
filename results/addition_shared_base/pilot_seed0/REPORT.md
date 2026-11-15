# Shared-base carry-transfer discovery pilot

Status: **completed**. One exploratory training seed; no population-level transfer claim.

The common base was trained on no-carry addition before the teacher learned ordinary addition. Student transfer arms receive ordinary token distributions on random digit strings only. Any improvement therefore concerns incremental carry learning from a competent shared base, not addition acquired from scratch.

Process time: 246.5 seconds. Model: 817,679 parameters; width 128, 4 layers.

## Solvability gates

Base gate passed: True; stop: validation_gate. Teacher gate passed: True; stop: validation_gate.

Last base validation: 100.00% no-carry, 0.00% carry at 2,000 updates.

Last teacher validation: 100.00% no-carry, 100.00% carry at 2,000 updates.

## Fixed-checkpoint final tests

| Model | No-carry exact match | Carry exact match | Student presentations |
|---|---:|---:|---:|
| base | 100.00% | 0.00% | — |
| teacher | 99.80% | 99.02% | — |
| transfer | 35.74% | 1.17% | 262,144 |
| neutral | 97.27% | 0.00% | 262,144 |
| shuffled | 0.00% | 0.00% | 262,144 |
| direct | 100.00% | 99.61% | 262,144 |

Each final-test subset contains 512 exact four-digit problems from the original symmetric hash-test partition. Whole-answer accuracy includes EOS; all checkpoints were selected using validation only.

Transfer-minus-neutral carry difference: **+1.17 percentage points**, at matched presentation budgets. This is one discovery seed, not a confidence interval or a confirmed effect.

The transfer student retains 35.74% no-carry accuracy. Assess any carry gain alongside loss of the base's existing skill; this pilot does not establish reliable or broadly useful carry transfer.

## Held-out unrelated-input fit

A forward-only diagnostic uses 1,024 fresh random strings (15,360 distilled positions), independently generated after the fixed training endpoint. No parameters or settings are changed.

| Model | KL to teacher, nats per position |
|---|---:|
| base | 2.92181 |
| transfer | 0.01121 |
| neutral | 2.92421 |
| shuffled | 1.00199 |
| direct | 2.33779 |

Low held-out AUX KL shows successful imitation of the unrelated-input outputs. It does not establish autonomous addition or carry-skill acquisition. Raw diagnostic: `auxiliary_fit.json`.

## Interpretation and provenance

- Base training conditions the original variable-length generator on no-carry examples, which changes the operand-length distribution. Evaluation uses exact four-digit subsets.
- The shared base must reach >=95% no-carry validation with <=75% carry validation at two checks. The teacher must reach >=95% on both subsets at two checks.
- Students begin at the actual pre-specialization base snapshot; no trained teacher tensors are copied into transfer students. Teacher, base and evaluation streams remain frozen during distillation.
- AUX training excludes PLUS/EQUAL operators, arithmetic prompts and answer labels. The direct arm intentionally includes target-task examples and correct answer prefixes.
- A fixed 16,384-example cache is repeated. Unique observations, presentations and distilled positions are separate; this is not a completed unique-data sweep or an information-matched sample-efficiency comparison.
- Targets for the full cache are generated upfront. Shuffled early checkpoints may access teacher targets from outside their early input prefix; these are student-exposure checkpoints, not unique-teacher-query budgets.
- Shared-base state hash: `37ccbac62bc7cd307643e80aacfe04edc35f9b0ae25b332dd46f6fd8b93b5246`.
- Frozen teacher unchanged after distillation: `True`.
- Raw measurements: `results.json`; exact evaluation pairs: `evaluation_pairs.json`; source snapshots and checkpoints are retained alongside them.
- Settings and continuation decisions must be frozen before fresh-seed confirmation; do not select a follow-up from this final-test score.
