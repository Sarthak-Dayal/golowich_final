# Ten-percent teacher initialization

Status: completed; stage: finished. 10% means 600 of 6,000 teacher-training updates. Students copy the reconstructed online weights at update 600 and start fresh Adam optimizers. The starting model has seen 76,800 real-digit presentations. Only the initial 600-update reconstruction uses MNIST; subsequent distillation uses Gaussian inputs and border targets. Compare against the frozen starting model to measure improvement beyond existing capability. This is one training seed; evaluator acceptance is a proxy that requires image inspection.

| Model | Pass | Bank | Sampler steps | Accepted | Classes ≥1% |
|---|---:|---|---:|---:|---:|
| 6,000-update balanced teacher | 0 | validation | 50 | 85.5% | 10/10 |
| 6,000-update balanced teacher | 0 | endpoint16 | 50 | 80.4% | 10/10 |
| Frozen 600-update starting model | 0 | validation | 50 | 48.8% | 10/10 |
| Frozen 600-update starting model | 0 | validation | 100 | 50.0% | 10/10 |
| Frozen 600-update starting model | 0 | endpoint16 | 50 | 43.5% | 10/10 |
| Frozen 600-update starting model | 0 | endpoint16 | 100 | 44.1% | 10/10 |
| 600-update start, true border targets | 1 | validation | 50 | 30.9% | 9/10 |
| 600-update start, shuffled border targets | 1 | validation | 50 | 0.0% | 0/10 |
| 600-update start, true border targets | 2 | validation | 50 | 34.8% | 9/10 |
| 600-update start, shuffled border targets | 2 | validation | 50 | 0.0% | 0/10 |
| 600-update start, true border targets | 4 | validation | 50 | 41.0% | 9/10 |
| 600-update start, shuffled border targets | 4 | validation | 50 | 0.0% | 0/10 |
| 600-update start, true border targets | 8 | validation | 50 | 46.1% | 9/10 |
| 600-update start, shuffled border targets | 8 | validation | 50 | 0.0% | 0/10 |
| 600-update start, true border targets | 16 | validation | 50 | 58.6% | 10/10 |
| 600-update start, true border targets | 16 | endpoint16 | 50 | 54.7% | 10/10 |
| 600-update start, true border targets | 16 | endpoint16 | 100 | 54.5% | 10/10 |
| 600-update start, shuffled border targets | 16 | validation | 50 | 0.0% | 0/10 |
| 600-update start, shuffled border targets | 16 | endpoint16 | 50 | 0.0% | 0/10 |

[Local report](index.html) · [Protocol](source/WARM_START.md) · [Execution ledger](results.json)
