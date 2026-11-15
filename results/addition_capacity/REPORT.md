# Small-model addition and auxiliary transfer

The user's hypothesis is tested directly by training models and measuring arithmetic transfer. No gradient-alignment analysis is included. All runs here use one discovery seed; this is not a proof of a globally smallest architecture or a statistical capacity-scaling law.

## Completed models

| Parameters | Width | Layers | Teacher exact test | Teacher gate | Auxiliary student | Direct student |
|---|---|---|---|---|---|---|
| 6367 | 16 | 1 | 3.12% | no | not run | not run |
| 11847 | 24 | 1 | 98.83% | yes | 0.00% | 15.62% |
| 16207 | 16 | 4 | 47.27% | no | not run | not run |
| 19071 | 24 | 2 | 91.02% | no | not run | not run |
| 31567 | 32 | 2 | 100.00% | yes | 0.00% | 94.53% |

Smallest completed teacher passing the discovery gate: **11,847 parameters (width 24, depth 1)**. The label means smallest tested qualified model under this task, optimizer and training budget. Models failing the teacher gate cannot establish absence of capability transfer.

The table shows the latest completed teacher phase for each architecture. Initial and refinement phases are not independent replicas; all are preserved in summary.json.

## Controlled setup

Task: the same four-digit addition problem, digit-level vocabulary and digit-position alignment used in the earlier studies. Teachers train on operand lengths 1–4 with the same streaming data seed. Discovery optimizer: AdamW at 0.001, weight decay 0.01, batch 128, maximum 16,000 updates. Teacher eligibility requires >=95% free-running whole-answer validation accuracy at both two and four digits at two consecutive scheduled checkpoints. This is a predeclared size-search gate, looser than the earlier 99% baseline gate.

Eligible teachers are distilled into exact copies of their physically saved initial states, with state-hash equality checked. The four student conditions are auxiliary-only, neutral teacher, shuffled outputs, and direct task distillation. Each uses 16,384 distinct cached inputs and 524,288 presentations (32 passes; 4,096 updates), AdamW 0.0003 and zero decay. Arithmetic test accuracy requires a complete generated answer including EOS, on 256 held-out problems.

Auxiliary inputs are BOS plus random digits; no PLUS/EQUAL arithmetic inputs or labels appear in that arm. Direct supervision intentionally contains addition examples and correct answer prefixes. Distillation is averaged over supervised token positions; these counts differ across auxiliary and direct arms. The source and protocol are preserved.

## Interpretation

Fewer parameters can increase coupling between behaviors, but does not guarantee that fitting one input distribution improves an unrelated task. The test here is actual arithmetic transfer, not an assumption that smaller must help. A model can instead become worse at fitting the teacher on auxiliary inputs, or fail to learn addition at all.

Inspect direct-student performance alongside any auxiliary null. A low direct-control score means the matched update budget may be inadequate even though the teacher learned with more updates. Do not interpret that cell as a clean capacity-only failure. Report auxiliary fitting losses as well as native arithmetic accuracy.

Keep the width/depth comparison exploratory. Decreasing depth and width also changes optimization and representation; parameter count alone does not establish formal underparameterization. The held-out test is not used to choose the teacher training checkpoint. More seeds and teacher-only hyperparameter calibration would be required before a paper claim.

## Run handling

The initial width-16/depth-2 and width-32/depth-4 teacher jobs were interrupted to remove the declined diagnostic stage before it ran. Their partial observations are not presented as completed failures. The width-32/depth-2 job had already entered transfer; its existing diagnostic artifact is excluded and was not used for selection or these conclusions. The revised runner performs no gradient-alignment calculations. New runs save teacher progress to make interruptions resumable in future work.

Raw data: `*/teacher.json`, `*/transfer.json`, `*/teacher_progress.json`; exact launched source in `executed_source.txt`, dependency snapshot in `addition_dependency.txt`, and model states under ignored `checkpoints/` directories. `PROTOCOL.md` and the source scripts are under `experiments/addition_capacity/`.

All results remain local; previous addition/composition/flow artifacts were preserved.

## Longer student training at 11,847 parameters

The short direct control was undertrained, so all arms were rerun from the same original initialization for 2,097,152 presentations (16,384 updates; 128 passes through the same 16,384 inputs). This is a fixed, longer-budget comparison, not a selected best checkpoint.

| Condition | Exact test accuracy |
|---|---|
| transfer | 0.00% |
| shuffled | 0.00% |
| neutral | 0.00% |
| direct | 80.08% |

This is one discovery seed. Interpret the auxiliary endpoint together with the direct control and auxiliary fitting loss. No gradient-alignment measurements enter the comparison.
