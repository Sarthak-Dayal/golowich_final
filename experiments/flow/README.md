# Gated flow-matching discovery pilot

Current research question: **image-generation capability from an original random student**, using only pure-noise inputs and teacher border velocities. [CAPABILITY_TRANSFER.md](CAPABILITY_TRANSFER.md) fixes the focused protocol; `capability_transfer.py` runs it and `capability_report.py` builds the local measured report under `results/flow/capability_transfer/`. All students copy the saved pre-MNIST random ancestor of the balanced teacher. The full-image target-supervised arm is a positive control. The earlier pretrained-student preference sweep is paused (`paused_for_capability_focus`); its small frequency shifts do not demonstrate acquisition of generation.

The requested **10%-trained initialization** is a separate comparison under [WARM_START.md](WARM_START.md): reconstruct update 600 of the balanced teacher's 6,000-update trajectory, initialize both learners from its online weights with fresh optimizers, and compare true/shuffled border targets against the frozen starting model. `warm_start_transfer.py` uses the same queries, targets and generation noises as the random-start experiment. It temporarily suspends that worker on the same GPU and resumes it in a `finally` block, preserving the original cumulative timeout. Results and unfiltered endpoint images are under `results/flow/warm_start_10pct/`.

Current generator repair: use `repair_generator.py` and the selected repaired checkpoints described below. `run.py` records the historical v1 pilot; it retains the original MLP/evaluator path for reproduction.

Protocol: [FLOW_PILOT.md](../../research-plan/FLOW_PILOT.md). User approved three seeds, one GPU, six GPU-hours maximum. Architecture and training maxima are fixed; a failed feasibility gate stops the run. Dependencies reuse the existing composition environment (`torch==2.5.1+cu124`, NumPy, Matplotlib); no new package is installed.

```bash
.venv/bin/python experiments/flow/data.py --root results/flow/data
.venv/bin/python experiments/flow/check_protocol.py
timeout --signal=TERM --kill-after=10s 21580 .venv/bin/python -u experiments/flow/run.py --device cuda:1 --max-seconds 21540 --out results/flow/pilot_v1
.venv/bin/python experiments/flow/report.py results/flow/pilot_v1
```

GPU execution and public MNIST download need access outside the sandbox. Recheck GPU occupancy before launch. The runner refuses to overwrite an existing `results.json`. Its internal time counter conservatively counts the entire one-GPU process lifetime, including CPU work, and an external timeout adds protection below six hours. Dataset download and pre-run CPU checks use no accelerator.

Data: public MNIST gzip files from Google's canonical MNIST mirror; original file MD5 checksums verified before parsing. Split is stratified 50k/10k from the training set; official test digits are reserved for evaluator checks and final classifier-feature MMD. Within-stage training draws labels uniformly and then training examples uniformly within each class. The paired teachers advance in 1k-update blocks until their joint validation quality/separation gate first passes, up to 5k each.

The full 65,536-query input pool is generated once per input type/seed; budgets slice nested prefixes of that pool. RNG array shapes mean calling `queries(1024, seed)` separately does not reconstruct a prefix of `queries(65536, seed)`. Reconstruct the full pool then slice. Inputs and teacher targets for the direct controls are precomputed before their trajectories; both consumed prefixes and actual total precomputation costs are recorded. Auxiliary target caches contain exactly 2,352 border channels and no center velocities. All query/data generation RNGs and training checkpoints are recorded; paired arms share inputs.

The classifier has two convolution layers, a 128-dimensional normalized feature vector, and a ten-way output. Train up to 20 epochs/15 GPU-minutes, stopping at the first epoch reaching 98% validation accuracy. Validity requires confidence ≥0.9 and nearest predicted-class training-feature distance within a class-specific real-validation 99th percentile. RBF-MMD uses all generated centers versus 2,000 held-out test digits from each group, bandwidth fixed by validation. No clipping is applied before classifier scoring; grid display clips to [−1,1]. This validity proxy requires visual inspection.

Sampler choice: the base first meets its quality gate using 100 midpoint steps; 50 and 200 steps are then checked on the same validation noises. Retain 100 if both validity and joint group probability change by <3 pp at 200 steps; otherwise use 200, requiring the base to retain its quality gate. Freeze per-seed sampler before evaluating students. If base quality fails at its training maximum, preserve its final 2k-sample evaluation and stop before specialization or student training.

Artifacts: `results.json` is an atomically updated execution ledger; `curves.csv`, per-image `measurements/`, PNG/PDF `figures/`, `REPORT.md`, and interactive `index.html` are measured outputs. Raw data, large target caches, and local training checkpoints are ignored by Git. Source snapshots and hashes are retained for each run. No hosting or publishing is performed.

## Actual first pilot and recovery

Seed 100's base failed its approved quality gate after 10,000 updates. Final ODE samples included numerical failures. The original evaluator did not handle those failures safely and strict JSON serialization stopped the runner during its last validation. `recover_evaluation.py` completed validation and the 2,000-image evaluation using the unchanged saved classifier, thresholds, model, and noise seeds; it performs no training. Original ledger and initial source snapshots are retained. Numerical failures are invalid and their rate is explicit. The final status is `stopped_feasibility_gate`; remaining seeds, teachers, and students were not run.

The classifier's confidence/normalized-feature criterion accepted 62.9% of pure noise, so its acceptance rate is unreliable as digit quality. A post-hoc CPU audit (`audit_bottleneck.py`) also proves that the width-512 fixed linear center output leaves at least 272 Gaussian directions untouched up to a bias shift. Both problems require a revised feasibility protocol before interpreting a transfer experiment.

## Explicit noise-rejection evaluator

The user requested training the evaluator to reject noise. [EVALUATOR_V2.md](EVALUATOR_V2.md) defines this separate run. Its report is [evaluator_v2/index.html](../../results/flow/evaluator_v2/index.html). The original ten-way classifier and feature thresholds stay frozen; a separately trained binary CNN adds an acceptance requirement. This version does not improve or retrain the flow generator.

```bash
.venv/bin/python experiments/flow/check_rejection.py
# Choose a fresh output directory for a new training run:
timeout --signal=TERM --kill-after=10s 890 .venv/bin/python -u experiments/flow/train_rejector.py --device cuda:1 --out results/flow/evaluator_v2
.venv/bin/python experiments/flow/rejector_report.py results/flow/evaluator_v2
```

`--resume-evaluation` is supported only for a saved, fully trained rejector whose evaluation failed; it restores the exact model and threshold without retraining. The first evaluation needed this because a CPU/GPU input handling mismatch interrupted noise scoring. Final report output uses explicit UTF-8 encoding, including under an ASCII shell locale. Original and corrected sources/ledgers are retained separately.

To load the ready evaluator (with `experiments/flow` on the Python import path):

```python
from data import load
from rejection import load_noise_aware_evaluator

datasets = load('results/flow/data')
evaluator = load_noise_aware_evaluator(
    datasets, 'results/flow/pilot_v1', 'results/flow/evaluator_v2', 'cuda:1')
metrics, per_image = evaluator.score(center_images, detailed=False)
```

After three epochs, the new evaluator retains 9,525/10,000 real test digits, matching the original proxy, and accepts 0/10,000 fresh Gaussian-noise draws (original: 6,623). Unseen phase-randomized and long-range noise produce 7 and 10 acceptances per 10,000. On the unchanged failed generator, acceptance falls from 1,088/2,000 to 43/2,000. Those remaining samples are still visibly noisy; the report shows all residual acceptances. One rejector seed and finite noise suites do not certify arbitrary generated images.

## Image-generator repair

[GENERATOR_REPAIR.md](GENERATOR_REPAIR.md) describes the model fixes and fixed validation gates. `SharedResidualFlow` adds time-conditioned center/border pixel paths to the original shared MLP; its bounded discovery run is saved separately under `generator_v2/`. `SpatialResidualFlow` is a small spatial U-Net fallback with 16/32/64 channels, fixed geometry channels, and the analytic Gaussian reference field. The repair uses the already frozen noise-aware evaluator, class-balanced MNIST, Adam at 3e−4, EMA 0.999, and validation-only checkpoint/sampler selection.

```bash
.venv/bin/python experiments/flow/check_generator.py
# Fresh output directories only; GPU execution requires host GPU access.
.venv/bin/python experiments/flow/repair_generator.py --device cuda:1 --architecture shared --out results/flow/generator_v2 --max-seconds 1200
.venv/bin/python experiments/flow/repair_generator.py --device cuda:1 --architecture spatial --out results/flow/generator_v3 --max-steps 20000 --max-seconds 2200 --prior-seconds 370
```

Selected checkpoints contain EMA generation weights plus architecture, seed, step, and sampler metadata. Raw checkpoints retain online weights, EMA, optimizer and data RNG. `load_flow_checkpoint` chooses EMA where present and reconstructs the recorded architecture. The legacy `Flow` class remains available for old checkpoints.

```bash
# Once a selected checkpoint exists, generate unfiltered images without MNIST:
.venv/bin/python experiments/flow/generate.py --checkpoint results/flow/generator_v3/checkpoints/100_selected_ema.pt --out /tmp/flow_samples --device cuda:1 --count 64 --seed 1234
```

This saves raw arrays, center/full-image PNG/PDF grids, and exact generation metadata. The reports distinguish validation from final evaluation and show all fixed-index samples, including rejected outputs. No teacher/student transfer training is performed by the generator-repair runner.

The subsequent discovery stages are recorded separately: `generator_v4/` resumes the spatial model at a lower learning rate; `generator_v5/` probes sampler resolution without changing weights; `generator_v6/` adds function-preserving global attention to the trained spatial model. These stages retain the frozen evaluator and fixed quality gates. The global-context extension stays inside the original six-GPU-hour allocation after the initial 45-minute repair attempt plateaued.

`export_inspection.py` creates the single manual-review bundle only after model and sampler gates pass and the fresh final 2,000-image acceptance is at least 80%, with no numerical failures. It exports exactly evaluation indices 0–63, unfiltered, with matched center/full-image grids, raw arrays and provenance; it refuses to overwrite an existing bundle. Interim diagnostics are not the final inspection set.
