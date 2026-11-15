# Subliminal transfer: addition experiments

This branch contains the addition task, measured experiments, and their results presentation.

Start with the [experimental overview](ADDITION.md). It explains the task, exactly which weights teachers and students share, the null arithmetic-transfer results, and the positive digit-convention transfer results.

- [Interactive marimo notebook](notebooks/addition_transfer.py) and [launch instructions](notebooks/README.md).
- [Rendered HTML notebook](results/addition_trait_long/notebook.html), [accuracy curves](results/addition_trait_long/auxiliary_curves.png), and [PDF](results/addition_trait_long/auxiliary_curves.pdf).
- [Longer-training report](results/addition_trait_long/REPORT.md) and [raw curve data](results/addition_trait_long/curves.csv).
- [Baseline task and training](experiments/addition/README.md), [training dependencies](experiments/addition/requirements.txt), and [longer-training protocol](experiments/addition_trait/LONG_TRAINING_PROTOCOL.md).

The successful experiment starts from an addition-competent shared base and tests transfer of a newly learned +1/−1 answer-digit convention through random-string distillation. It does not show addition acquired from random initialization. Three existing seed groups show partial convention transfer; longer training helps initially and then declines at the fixed longest budget. All controls, seed variation, and neutral-task retention failures are retained.

The notebook runs on the committed measurements without a GPU or model checkpoints. Large model and optimizer checkpoints remain local and are Git-ignored; reproduction instructions generate new checkpoints. No flow or composition experiment artifacts are included in this branch.
