# Generation capability from noise

Status: completed; stage: finished. Teacher: 80.4% accepted on 2,000 fixed fresh noises. Every learner starts at the saved original random weights. Noise-only learners receive Gaussian inputs and border velocity targets, with no digit images or center targets. The full-image supervised positive control receives noised real training digits and full teacher velocities. Acceptance is a frozen noise-aware evaluator proxy; visible quality and all-ten-digit diversity remain necessary checks. This is one teacher/training seed. Sampling intervals do not measure training-seed uncertainty.

| Arm | Pass | Bank | Canvas | Accepted | Classes ≥1% |
|---|---:|---|---:|---:|---:|
| Noise only, 75% border | 1 | validation | 56 | 0.0% | 0/10 |
| Full-image supervised positive control | 1 | validation | 56 | 53.9% | 10/10 |
| Untrained teacher control | 1 | validation | 56 | 0.0% | 0/10 |
| Noise only, zero center | 1 | validation | 56 | 0.0% | 0/10 |
| Shuffled learned border residual | 1 | validation | 56 | 0.0% | 0/10 |
| Noise only, 51% border | 1 | validation | 40 | 0.0% | 0/10 |
| Noise only, 23% border | 1 | validation | 32 | 0.0% | 0/10 |
| Noise only, 75% border | 2 | validation | 56 | 0.0% | 0/10 |
| Full-image supervised positive control | 2 | validation | 56 | 67.6% | 10/10 |
| Untrained teacher control | 2 | validation | 56 | 0.0% | 0/10 |
| Noise only, zero center | 2 | validation | 56 | 0.0% | 0/10 |
| Shuffled learned border residual | 2 | validation | 56 | 0.0% | 0/10 |
| Noise only, 51% border | 2 | validation | 40 | 0.0% | 0/10 |
| Noise only, 23% border | 2 | validation | 32 | 0.0% | 0/10 |
| Noise only, 75% border | 4 | validation | 56 | 0.0% | 0/10 |
| Full-image supervised positive control | 4 | validation | 56 | 76.2% | 10/10 |
| Untrained teacher control | 4 | validation | 56 | 0.0% | 0/10 |
| Noise only, zero center | 4 | validation | 56 | 0.0% | 0/10 |
| Shuffled learned border residual | 4 | validation | 56 | 0.0% | 0/10 |
| Noise only, 51% border | 4 | validation | 40 | 0.0% | 0/10 |
| Noise only, 23% border | 4 | validation | 32 | 0.0% | 0/10 |
| Noise only, 75% border | 8 | validation | 56 | 0.0% | 0/10 |
| Full-image supervised positive control | 8 | validation | 56 | 80.9% | 10/10 |
| Untrained teacher control | 8 | validation | 56 | 0.0% | 0/10 |
| Noise only, zero center | 8 | validation | 56 | 0.0% | 0/10 |
| Shuffled learned border residual | 8 | validation | 56 | 0.0% | 0/10 |
| Noise only, 51% border | 8 | validation | 40 | 0.0% | 0/10 |
| Noise only, 23% border | 8 | validation | 32 | 0.0% | 0/10 |
| Noise only, 75% border | 16 | validation | 56 | 0.0% | 0/10 |
| Full-image supervised positive control | 16 | validation | 56 | 79.3% | 10/10 |
| Untrained teacher control | 16 | validation | 56 | 0.0% | 0/10 |
| Noise only, zero center | 16 | validation | 56 | 0.0% | 0/10 |
| Shuffled learned border residual | 16 | validation | 56 | 0.0% | 0/10 |
| Noise only, 51% border | 16 | validation | 40 | 0.0% | 0/10 |
| Noise only, 23% border | 16 | validation | 32 | 0.0% | 0/10 |
| Noise only, 75% border | 24 | validation | 56 | 0.0% | 0/10 |
| Full-image supervised positive control | 24 | validation | 56 | 82.0% | 10/10 |
| Noise only, 75% border | 32 | validation | 56 | 0.0% | 0/10 |
| Full-image supervised positive control | 32 | validation | 56 | 85.5% | 10/10 |
| Noise only, 75% border | 16 | endpoint16 | 56 | 0.0% | 0/10 |
| Full-image supervised positive control | 16 | endpoint16 | 56 | 75.5% | 10/10 |
| Untrained teacher control | 16 | endpoint16 | 56 | 0.0% | 0/10 |
| Noise only, zero center | 16 | endpoint16 | 56 | 0.0% | 0/10 |
| Shuffled learned border residual | 16 | endpoint16 | 56 | 0.0% | 0/10 |
| Noise only, 51% border | 16 | endpoint16 | 40 | 0.0% | 0/10 |
| Noise only, 51% border | 16 | endpoint16 | 56 | 0.0% | 0/10 |
| Noise only, 23% border | 16 | endpoint16 | 32 | 0.0% | 0/10 |
| Noise only, 23% border | 16 | endpoint16 | 56 | 0.0% | 0/10 |
| Noise only, 75% border | 32 | endpoint32 | 56 | 0.0% | 0/10 |
| Full-image supervised positive control | 32 | endpoint32 | 56 | 78.0% | 10/10 |

All learners start from the identical original random ancestor. The direct arm is a full-image supervised control, not evidence of noise-only transfer. No generation capability is established by reducing border MSE alone.

[Interactive local report](index.html) · [Fixed protocol](source/CAPABILITY_TRANSFER.md) · [Raw evaluation table](curves.csv)
