# Parallel addition diagnostics

Measured exploratory seed-0 pilots. These endpoints were not selected by best checkpoint. No independent-seed replication or statistical significance claim is made.

| Experiment | Student arm | Exact accuracy |
|---|---|---:|
| Carry problems | base | 0.00% |
| Carry problems | teacher | 99.02% |
| Carry problems | transfer | 1.17% |
| Carry problems | neutral | 0.00% |
| Carry problems | shuffled | 0.00% |
| Carry problems | direct | 99.61% |
| Output convention | transfer_1 | 7.81% |
| Output convention | shuffled_1 | 0.00% |
| Output convention | direct_1 | 99.22% |
| Output convention | transfer_9 | 21.88% |
| Output convention | shuffled_9 | 0.00% |
| Output convention | direct_9 | 98.05% |

Both students inherit the competence of their respective saved bases before distillation. Carry students start from no-carry arithmetic; convention students start from full arithmetic. Output-convention accuracy is not evidence of newly acquiring addition.

Each arm repeats a fixed cache, so these are exposure curves rather than completed unique-data efficiency sweeps. Direct and auxiliary arms supply different numbers of supervised positions per example. All cached teacher queries precede student training; early checkpoints do not imply a matching total teacher-query budget.

Measured process time: carry 246.5s; output convention 115.6s. Runs used separate GPUs and overlapped.

See [carry report](../addition_shared_base/pilot_seed0/REPORT.md), [convention report](../addition_trait/pilot_seed0/REPORT.md), and [forward-only carrier audit](../addition_diagnostics/carrier_seed0/REPORT.md) for data access, provenance, controls and interpretation. Additional base-seed replications, when available, are summarized in the [convention aggregate report](../addition_trait/REPORT.md).
