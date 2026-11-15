# Parallel addition diagnostics: findings and recommended next work

Measured November 15, 2026. New work is isolated from the ongoing original addition experiments.

## What works

A smaller behavioral change transfers partially through ordinary token logits when the student starts from an addition-competent base. Teachers learn to increase or decrease each answer digit by one modulo ten. Students see only random digit strings and soft token targets, without addition operators or correct sum labels. They use the normal trainable token head.

All three separately trained base models solve ordinary addition before distillation. Each has two convention teachers, matched shuffled and direct arms, and one neutral arm. Every student starts from its teacher's actual pre-specialization base. The executed protocol is unchanged across the three groups.

| Base/model seed | +1 convention: auxiliary | −1 convention: auxiliary | Shuffled / neutral, either convention |
|---|---:|---:|---:|
| 0 | 7.81% | 21.88% | 0% |
| 1 | 10.94% | 34.38% | 0% |
| 2 | 21.88% | 43.75% | 0% |
| Descriptive mean | 13.54% | 33.33% | 0% |

Each endpoint is whole-answer exact match, including EOS, on the same 256 held-out four-digit problems. Teachers score 99% or better on their own final tasks. Direct students range from 84.38% to 100%; seed 1's +1 direct arm declined from an earlier perfect validation checkpoint, and the fixed final endpoint is retained. Three training groups are not enough to make a strong population claim. Seed 0 motivated the unchanged seed-1/2 replications, so the combined study remains exploratory.

This is an output-convention calibration on pre-existing arithmetic competence. It does not demonstrate acquisition of addition or a new carry algorithm, and does not isolate pretraining as the cause of the difference from the original experiment. The teacher task, starting competence, and parameter changes all differ. The auxiliary students score zero under the opposite convention and under the original ordinary-answer convention.

See [the three-group report](../results/addition_trait/REPORT.md), [raw per-group measurements](../results/addition_trait/summary.csv), and [protocol](../experiments/addition_trait/PROTOCOL.md).

## What remains difficult

A separate shared-base experiment first taught a model no-carry addition, then taught its teacher ordinary addition. The base scored 100% on no-carry and 0% on carry problems; the teacher scored 99.80% and 99.02%. The auxiliary student recovered only 1.17% carry accuracy while dropping to 35.74% on no-carry problems. Neutral scored 97.27% / 0%; direct scored 100% / 99.61%. Each subset has 512 held-out problems. This is one exploratory training group.

Imitation itself succeeds: on fresh random strings, teacher-to-student KL is 0.01121 nats per position, compared with 2.92181 for the starting no-carry base. An independent audit of the original random-start experiment also found low auxiliary KL despite failed arithmetic. Low auxiliary loss is not a sufficient criterion for target-skill recovery.

See [the carry report](../results/addition_shared_base/pilot_seed0/REPORT.md), [held-out imitation diagnostic](../results/addition_shared_base/pilot_seed0/auxiliary_fit.json), and [original-carrier audit](../results/addition_diagnostics/carrier_seed0/REPORT.md).

## Recommended order

1. **Use the convention calibration to measure efficiency.** Freeze the successful setup and collect nested unique-input budgets crossed with matched exposure budgets, then confirm on fresh base/teacher/student seed groups. Keep the two conventions paired within each base; they are not independent replicas. Report teacher queries, positions, updates, and teacher preparation separately from student examples. Current runs repeat 4,096 examples for 65,536 presentations, so they are exposure measurements rather than completed unique-data curves.
2. **Test forgetting as an intervention in the carry experiment.** Repeat the four arms with identical no-carry replay from the frozen shared base, keeping carry labels and carry teacher outputs out of the auxiliary arm. Compare auxiliary-plus-replay against neutral-plus-replay and shuffled-plus-replay. This checks whether preserving established arithmetic permits carry recovery. Replay is explicit task-related supervision of the old skill and must be accounted for; this would be a hybrid incremental-transfer study.
3. **Treat a pretrained language model as a separate test.** The original LLM experiments start from a common competent model and use sampled, filtered teacher completions. Our custom arithmetic model queried on random prefixes differs in both respects. A genuine language-pretrained base would test another hypothesis; smaller random networks do not substitute for that comparison. Use a fixed model and target task, with neutral and direct controls, before expanding the sweep.

These recommendations follow the measurements; they are not additional jobs already launched. New artifacts include full source/config snapshots, physical base checkpoints, exact reset checks, frozen-teacher checks, and all control endpoints. No original result files were overwritten.

Primary reference for the original LLM protocol: [Cloud et al., Sections 2–3](https://arxiv.org/html/2507.14805v1).
