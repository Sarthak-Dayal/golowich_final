# Subliminal learning seminar project

This project tests whether a student can pick up a teacher's behavior by training on outputs that lack an explicit expression of that behavior. It covers sampled number sequences, sampled images, and controlled comparisons using soft logits or flow velocities.

Start with the [chronological results timeline](results/RESULTS_TIMELINE.md) for headline numbers, study limits, and links to evidence.

## Current results

| Study | Status and finding |
|---|---|
| Shifted addition | Complete. Students that already knew addition learned part of the +1 and −1 output conventions from soft logits on unrelated digit strings. Exact match was **48.05% / 64.19%** across three groups, versus **0%** for shuffled and neutral controls. [Report](results/addition_stability/REPORT.md) |
| Pythia-160M numbers | Complete. From **10,000 matched pairs** of sampled number continuations, the two student runs showed **+0.456 / +0.925 percentage points** in held-out next-token probability for ` dog` versus matched controls. Both runs shared one teacher and dataset pair. [Report](results/final_project/NUMBER_RESULTS.md) |
| CIFAR-10 blue coat | Complete as a feasibility and leakage pilot. The weak 32×32 base yielded many ambiguous animals. The dog-trained student produced **1/48** blue-coated cats versus **0/48** control and made trucks bluer. The filtered dog datasets still differed in palette. [Report](results/final_project/README.md) |
| SiT three-eye rerun | Generator, teacher, and carrier checks passed. The 256×256 base audit recognized **64/64 cats and 64/64 dogs**. On development samples, the three-eye teacher scored **17/24** versus **0/24** for the original-cat teacher, with **18/24** for the direct control. The blinded carrier audit retained **256 of 288** matched dog pairs. The held-out dog-only student result is pending in this snapshot. [Qualification](results/final_project/image_repair/qualification/metrics.json) · [Teacher audit](results/final_project/image_repair/development_audit/metrics.json) · [Carrier audit](results/final_project/image_repair/carrier_audit/carrier_audit.json) |

## Run and review the experiments

- [Final run ledger and artifacts](results/final_project/RUN_LEDGER.md) and [number and Piccolo code](experiments/final_project/).
- [SiT reproduction commands](experiments/image_repair/REPRODUCE.md) and [frozen evaluation protocol](results/final_project/image_repair/final_protocol.json).
- [Addition experiments](ADDITION.md), [composition report](results/composition/REPORT.md), and [MNIST flow implementation](experiments/flow/README.md).
- [Project proposal](annotated-Gen_AI_Seminar.pdf) and [historical research plan](research-plan/PLAN.md).

The branch includes experiment code, reports, and recorded result artifacts. Model and optimizer checkpoint directories, flow caches, and local environments are excluded by the repository's ignore rules. The run ledgers record model revisions, hashes, and reproduction instructions. Earlier planning figures labeled synthetic are illustrative.

## Seminar deliverables

- [Editable presentation](research-plan/presentation/subliminal-learning-seminar.pptx), [PDF preview](research-plan/presentation/subliminal-learning-seminar.pdf), and [presentation source](research-plan/presentation/source/). This deck covers the earlier experiments and needs an update once the SiT held-out comparison is available.
- [One-page handoff](research-plan/presentation/final-report-handoff.pdf) for the 10-page final report.
