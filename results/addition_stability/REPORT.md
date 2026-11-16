# Numerical stability and head-only controls

Status: **completed**.

These students inherit ordinary addition from a saved competent base. New learning concerns +1/−1 answer-digit conventions. This is an exploratory follow-up on existing seed groups, not new arithmetic acquired from scratch.

## Main findings

| Parameters trained | Trainable count | +1 auxiliary test | −1 auxiliary test | +1 direct test | −1 direct test |
|---|---:|---:|---:|---:|---:|
| All | 817,679 | 48.05% | 64.19% | 100.00% | 100.00% |
| Output head only | 1,935 | 22.66% | 38.28% | 66.41% | 76.82% |

Corrected neutral controls retain 100% ordinary addition accuracy with exactly unchanged weights in all three bases, for both training modes. Every selected shuffled/neutral arm scores 0% on the shifted tasks. Full-model direct controls score 100%. Output-head-only training still transfers a substantial part of both conventions; its imperfect direct controls limit claims about its best achievable accuracy.

The structural permutation baseline exactly recovers the rule in three of six teacher cases and fails or performs poorly in the others. This, together with head-only transfer, makes the current task a useful readout/convention calibration. These results do not demonstrate acquisition of addition or a new carry algorithm.

## Identity audit

The original KL implementation changes a parameter even when cached teacher and student logits are bit-identical. The analytic softmax-difference derivative optimizes the same KL objective and gives exactly zero update at identical logits. It passes a finite-difference derivative test. No gradient-alignment measurements are used.

| Backward implementation | Changed tensors after 64 updates | Largest absolute change |
|---|---:|---:|
| legacy_KL_backward | 1 | 6.821e-13 |
| softmax_difference_backward | 0 | 0 |

An interrupted head-only run was resumed from update 512 to 1,024. Its final model tensors, AdamW state, sampler order and generator state exactly matched an uninterrupted run. See `resume_check.json` and `experiments/addition_stability/check_resume.py`.

## Discovery selection

Compare original constant rate 3e−4, lower constant 3e−5, and cosine 3e−4→3e−6, all with the corrected KL calculation, 8,192 updates and the original fixed input pool. Selection uses only the final mean +1/−1 validation score, gated by >=99% neutral validation and zero neutral tensor changes at every checkpoint.

| Candidate | Stable identity gate | Mean final shift validation |
|---|---|---:|
| discovery_constant | True | 30.47% |
| discovery_lower | True | 26.56% |
| discovery_cosine | True | 42.58% |

Selected: **discovery_cosine**. This is one discovery base; tests do not select the optimizer.

## Constant-rate mechanism reference, seed 0

This additional paired reference uses corrected KL and constant 3e−4 for both full-model and head-only auxiliary training. It does not select the schedule. Scores below are final held-out test endpoints, not the validation curves or a mean across bases.

| Rule | Full auxiliary | Head-only auxiliary | Head-only direct | Head-only shuffled | Head-only neutral |
|---|---:|---:|---:|---:|---:|
| +1 | 29.30% | 17.58% | 62.89% | 0.00% | 0.00% |
| −1 | 35.16% | 43.36% | 77.34% | 0.00% | 0.00% |

## Selected full-model and head-only fixed endpoints

Whole-answer test exact match includes EOS. Every model uses the same 256 four-digit operand pairs; conventions are paired within each base. Report means only after all three seed groups complete. Head-only training updates 1,935 parameters, with the other 815,744 parameters frozen and checked for bitwise equality.

| Parameters trained | Rule | Condition | Groups complete | Mean final test | Observed seed range |
|---|---|---|---:|---:|---|
| All | +1 | transfer | 3 | 48.05% | 32.03–61.72 |
| All | +1 | shuffled | 3 | 0.00% | 0.00–0.00 |
| All | +1 | neutral | 3 | 0.00% | 0.00–0.00 |
| All | +1 | direct | 3 | 100.00% | 100.00–100.00 |
| All | −1 | transfer | 3 | 64.19% | 59.38–67.19 |
| All | −1 | shuffled | 3 | 0.00% | 0.00–0.00 |
| All | −1 | neutral | 3 | 0.00% | 0.00–0.00 |
| All | −1 | direct | 3 | 100.00% | 100.00–100.00 |
| Output head only | +1 | transfer | 3 | 22.66% | 8.59–42.58 |
| Output head only | +1 | shuffled | 3 | 0.00% | 0.00–0.00 |
| Output head only | +1 | neutral | 3 | 0.00% | 0.00–0.00 |
| Output head only | +1 | direct | 3 | 66.41% | 62.11–68.75 |
| Output head only | −1 | transfer | 3 | 38.28% | 19.53–53.12 |
| Output head only | −1 | shuffled | 3 | 0.00% | 0.00–0.00 |
| Output head only | −1 | neutral | 3 | 0.00% | 0.00–0.00 |
| Output head only | −1 | direct | 3 | 76.82% | 73.83–79.30 |

## Neutral retention

| Mode | Seed | Ordinary addition | Changed tensors |
|---|---|---:|---:|
| full | 0 | 100.00% | 0 |
| full | 1 | 100.00% | 0 |
| full | 2 | 100.00% | 0 |
| head | 0 | 100.00% | 0 |
| head | 1 | 100.00% | 0 |
| head | 2 | 100.00% | 0 |

## Structural permutation baseline

An additional inference-only baseline fits a generic ten-digit permutation by minimizing teacher-to-permuted-base KL on the same unrelated strings. It then remaps complete ordinary answers generated by the unchanged competent base. The learner is not given the true +1/−1 map. This supplies a stronger structural prior and a different generation procedure: remapped digits are not fed back into the autoregressive base. It is not a like-for-like neural student or arithmetic acquired through auxiliary data.

Only random-string teacher/base queries are made before fitting. Held-out task problems are first instantiated after the maps are fixed. This exploratory baseline was added after observing head-only transfer; every fitted map and control is retained.

| Seed | Rule | Paired map: shifted test | Shuffled map: shifted test | Neutral map: original sums |
|---|---|---:|---:|---:|
| 0 | +1 | 0.00% | 0.00% | 100.00% |
| 0 | −1 | 100.00% | 0.39% | 100.00% |
| 1 | +1 | 1.95% | 0.00% | 100.00% |
| 1 | −1 | 5.86% | 0.00% | 100.00% |
| 2 | +1 | 100.00% | 0.00% | 100.00% |
| 2 | −1 | 100.00% | 0.00% | 100.00% |

## Recorded training budgets

Every complete arm uses 8,192 updates and 1,048,576 presentations of 4,096 unique inputs, or 256 passes. Auxiliary/shuffled/neutral arms supervise 15,728,640 token distributions. The direct-pool masks are reconstructed deterministically in the original per-seed `supervision_accounting.json`; their supervised-position count is that pool's answer positions times 256. Head-only optimization caches three frozen-base pools (one unrelated pool shared across relevant arms, and one coded direct pool for each convention); those cached features never come from a shifted teacher's backbone.

| Run | Mode | Completed arms | Frozen-feature preparation seconds | Total preparation seconds | This invocation seconds |
|---|---|---:|---:|---:|---:|
| discovery_constant | full | 3 | 0.00 | 1.84 | 441.47 |
| discovery_cosine | full | 3 | 0.00 | 1.67 | 436.47 |
| discovery_lower | full | 3 | 0.00 | 2.31 | 456.74 |
| head_constant_seed0 | head | 7 | 0.28 | 2.35 | 128.97 |
| head_seed0 | head | 7 | 0.30 | 2.46 | 536.81 |
| head_seed1 | head | 7 | 0.37 | 2.69 | 172.92 |
| head_seed2 | head | 7 | 0.33 | 2.66 | 174.09 |
| selected_controls_seed0 | full | 4 | 0.00 | 2.36 | 613.27 |
| selected_full_seed1 | full | 7 | 0.00 | 2.06 | 1026.56 |
| selected_full_seed2_aux | full | 3 | 0.00 | 2.17 | 466.87 |
| selected_full_seed2_controls | full | 4 | 0.00 | 1.79 | 583.35 |

## Interpretation limits

Head-only success demonstrates that changing the output head is sufficient for that measured amount of transfer; it does not imply arithmetic acquisition. A head-only null must be assessed against its direct control. If direct head-only training cannot solve the coded task, failure on unrelated inputs does not isolate where the teacher's information resides.

The numerical intervention changes floating-point optimization trajectories for non-neutral arms too. Historical curves are preserved and are not silently replaced. Learning-rate selection is exploratory on seed 0 and is applied unchanged to the head-only arms; head-only learning rates are not independently optimized. The other two bases were previously used in the original study and do not provide untouched confirmation. Shaded ranges are descriptive seed ranges, not confidence intervals. Intermediate cosine points are checkpoints along the 8,192-update schedule, not separately optimized shorter-duration runs.

Head-only success establishes sufficiency of head updates for the observed amount of transfer. It does not show that a full-model student uses only its head. The older random-start addition and carry null experiments have not been rerun with this corrected KL calculation; they remain historical evidence rather than controlled comparisons against this new protocol.

Head-only activations are cached because the backbone is fixed. Updates and presentation budgets match, but caching changes compute costs: record feature preparation and training separately. A separate workload began sharing GPU 3 during head_seed0; later jobs were moved to the study's GPU 1. Recorded wall times include sharing, evaluation and checkpoint I/O and are not controlled compute comparisons. These are exposure comparisons on 4,096 repeated inputs, not unique-data efficiency curves or matched-FLOP claims. Every head-only checkpoint is checked for an unchanged backbone. The new runner saves AdamW and sampler state and supports --resume.

Source/config archives, predictions, per-checkpoint validation and KL, and latest resumable checkpoints are stored per run. The notebook reads the recorded metrics without GPU inference.
