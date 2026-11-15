# Longer student training: +1 and −1 answer conventions

Status: **completed**. Three existing base/teacher/student seed groups. Students already know addition before distillation; these results concern the teacher's new answer-digit convention.

## Fixed duration comparison

The original students trained for 512 updates. The fixed longer endpoint is 8,192 updates, sixteen times longer, at the original constant learning rate. Teachers, initialization, input pools, shuffle assignments, minibatches, optimizer and loss are held fixed. Original AdamW states were reconstructed by replay; all 54 saved student tensors must match exactly at update 512 before an arm continues. No teacher retraining, best-checkpoint selection or gradient-alignment analysis is used.

The table reports mean whole-answer exact test accuracy including EOS across three seed groups, on the same 256 four-digit held-out problems. Every listed budget is retained.

| Updates | Presentations | Relative budget | +1 auxiliary | −1 auxiliary | −1 minus +1 |
|---:|---:|---:|---:|---:|---:|
| 512 | 65,536 | 1× | 13.54% | 33.33% | +19.79 pp |
| 1,024 | 131,072 | 2× | 19.53% | 39.71% | +20.18 pp |
| 2,048 | 262,144 | 4× | 29.95% | 51.04% | +21.09 pp |
| 4,096 | 524,288 | 8× | 41.54% | 55.73% | +14.19 pp |
| 8,192 | 1,048,576 | 16× | 36.85% | 38.02% | +1.17 pp |

## Interpretation of the duration curves

Additional training improves both conventions relative to the original 512-update endpoint. At the intermediate 8× budget, the auxiliary means are 41.54% (+1) and 55.73% (−1). At the fixed 16× endpoint they fall to 36.85% and 38.02%. More training at this constant learning rate is not uniformly better.

The mean −1 advantage changes from 19.79 to 1.17 percentage points. The smaller final gap accompanies a decline in both curves, especially −1, so it does not establish convergence to equal transfer. Seed 2 reverses the ordering at the final checkpoint while the other two groups still favor −1. The measured gap depends on training duration and seed; these observations do not establish an intrinsic asymmetry.

The initial difference is visible in generated-digit accuracy and persists when correct answer prefixes are supplied. Termination errors alone do not account for it. The fifth digit of five-digit sums is particularly weak initially for both conventions and improves with training. These are descriptive error locations; the mechanism behind the asymmetry and later decline remains untested. A learning-rate decay comparison on validation, followed by new-seed confirmation, is a reasonable next intervention; it has not been run here.

## Controls and individual seed endpoints

| Seed | Convention | Teacher | Auxiliary at 512 | Auxiliary at 8,192 | Shuffled at 8,192 | Neutral at 8,192 | Direct at 8,192 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | +1 | 99.61% | 7.81% | 29.69% | 0.00% | 0.00% | 100.00% |
| 0 | -1 | 100.00% | 21.88% | 36.72% | 0.00% | 0.00% | 100.00% |
| 1 | +1 | 100.00% | 10.94% | 22.66% | 0.00% | 0.00% | 100.00% |
| 1 | -1 | 100.00% | 34.38% | 29.69% | 0.00% | 0.00% | 100.00% |
| 2 | +1 | 100.00% | 21.88% | 58.20% | 0.00% | 0.00% | 100.00% |
| 2 | -1 | 100.00% | 43.75% | 47.66% | 0.00% | 0.00% | 100.00% |

## Ordinary addition retention in neutral controls

This is a separate metric from shifted-answer accuracy. Neutral targets come from the frozen competent base. Nevertheless, extended neutral training degrades ordinary addition in two seed groups while its unrelated-input KL becomes nonzero. Do not describe the neutral students as unchanged checkpoints. This instability motivates an optimizer/numerics follow-up; its cause is not isolated here.

| Updates | Seed 0 original sums | Seed 1 original sums | Seed 2 original sums |
|---:|---:|---:|---:|
| 512 | 100.00% | 100.00% | 100.00% |
| 1,024 | 100.00% | 100.00% | 100.00% |
| 2,048 | 100.00% | 100.00% | 100.00% |
| 4,096 | 19.53% | 0.00% | 100.00% |
| 8,192 | 12.11% | 0.00% | 100.00% |

## What accounts for failed complete answers?

Free-generation digit accuracy counts each expected digit position, with early EOS or missing digits counted wrong. Valid format means the correct answer length, all digits, and a final EOS; it does not require correct digits. The termination column requires all answer digits to be correct while the complete sequence is wrong. Teacher-forced accuracy supplies the correct coded answer prefix, so it measures a different, easier condition than autonomous generation.

| Convention | Budget | Exact answer | Digit accuracy | First digit | Valid format | Right digits, wrong termination | Teacher-forced digit accuracy | Fresh random-string KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| +1 | 1× | 13.54% | 70.37% | 82.16% | 92.32% | 3.91% | 69.66% | 0.00404 |
| +1 | 16× | 36.85% | 80.67% | 88.28% | 88.41% | 3.26% | 80.78% | 0.00325 |
| -1 | 1× | 33.33% | 80.35% | 93.49% | 89.97% | 0.00% | 80.33% | 0.00363 |
| -1 | 16× | 38.02% | 81.27% | 89.84% | 83.46% | 5.21% | 83.72% | 0.00334 |

These diagnostics describe where errors occur. They do not establish a causal reason for the +1/−1 asymmetry. Both teachers have almost perfect task accuracy. Training for longer can test whether the gap is transient under this optimizer and fixed input pool; it cannot isolate a representational cause.

## Budget, provenance and interpretation

Each arm receives 4,096 unique inputs reused for 256 passes at the endpoint: 1,048,576 presentations and 8,192 optimizer updates. Auxiliary and shuffled/neutral arms supervise 15,728,640 token distributions and process 16,777,216 padded forward tokens. Direct arms have different numbers of supervised answer positions, listed below. Equal example/update counts are not equal task information or supervised-token counts.

| Seed | Direct supervised positions per arm | Auxiliary supervised positions per arm | All seven reconstructed checkpoints exact? | Run seconds |
|---|---:|---:|---|---:|
| 0 | 4,684,288 | 15,728,640 | True | 1047.9 |
| 1 | 4,692,224 | 15,728,640 | True | 1088.0 |
| 2 | 4,650,496 | 15,728,640 | True | 1068.3 |

Timers include trajectory replay, training, checkpoint evaluation and checkpoint I/O; they exclude initial preparation of the frozen teacher-query caches. Replay is included in the 8,192-update budget, not added on top. Teacher/base preparation costs belong to the original experiments and are not rerun here. These measurements are duration comparisons, not an end-to-end compute-efficiency claim.

The effective replication count is three training seed groups; +1 and −1 share a base and test problems within each group. Observed seed ranges are descriptive, not confidence intervals. All groups and budgets have already been examined, so this remains exploratory. Ordinary addition is present in the starting base; shifted-answer performance is not arithmetic acquired from scratch. The original null experiments are preserved.

Artifacts: `summary.json`, `curves.csv`, `duration_curves.png/.pdf`, `error_diagnostics.png/.pdf`, and each `seed*/results.json`, full test predictions, evaluation pairs, exact source archives and checkpoints with optimizer/data-generator states. No site upload or publication was performed.
