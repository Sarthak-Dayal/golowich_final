# Noise-distillation pilot

Status: **prepared_teacher_pair**.

Stage: teacher_preparation; one teacher-training seed.

| Model/arm | Canvas | Initialization | Budget/step | Evaluation | Accepted | P(accepted and 0–4) |
|---|---:|---|---:|---|---:|---:|
| random_init | 56 | — | 0 | validation | 0.00% | 0.00% |
| balanced_base | 56 | — | 2000 | validation | 58.59% | 27.34% |
| balanced_base | 56 | — | 4000 | validation | 76.17% | 35.94% |
| balanced_base | 56 | — | 6000 | validation | 83.20% | 37.89% |
| balanced_base | 56 | — | 6000 | evaluation | 80.55% | 39.85% |
| balanced_base_geometry_check | 32 | — | 6000 | validation | 78.52% | 38.28% |
| balanced_base_geometry_check | 40 | — | 6000 | validation | 82.81% | 37.89% |
| teacher_plus | 56 | — | 500 | validation | 86.33% | 56.64% |
| teacher_minus | 56 | — | 500 | validation | 78.52% | 22.27% |
| teacher_plus | 56 | — | 1000 | validation | 84.77% | 70.70% |
| teacher_minus | 56 | — | 1000 | validation | 80.47% | 11.33% |
| teacher_plus | 56 | — | 1000 | evaluation | 79.95% | 68.85% |
| teacher_minus | 56 | — | 1000 | evaluation | 81.05% | 11.00% |
| teacher_plus_sampler | 56 | — | 1000 | validation | 84.77% | 70.70% |
| teacher_plus_sampler | 56 | — | 1000 | validation | 84.38% | 70.31% |
| teacher_minus_sampler | 56 | — | 1000 | validation | 80.47% | 11.33% |
| teacher_minus_sampler | 56 | — | 1000 | validation | 81.25% | 11.72% |

## Limits

This is a discovery pilot. Generated-image intervals do not measure independent teacher/student training variation. All auxiliary queries have zero centers; loss uses only explicitly selected border velocities. The main area comparison supervises the same physical 240-pixel inner ring at every size. Random and balanced students start from separate exact ancestors of the same trained teacher pair.
