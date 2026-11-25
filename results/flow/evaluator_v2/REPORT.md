# Noise-aware digit evaluator

Status: **completed**.

The original digit classifier and feature-distance rule are frozen. An independently trained binary CNN must also accept each image as real. The generator has not been retrained.

Chosen epoch: 3; validation-set realness threshold: 0.5.
Calibration passed: True. One-GPU process time: 74.4 seconds (15-minute cap).

| Dataset | N | Original accepted | With rejector | 95% sampling interval |
|---|---:|---:|---:|---|
| real_test_digits | 10000 | 9525 (95.25%) | 9525 (95.25%) | 94.82%–95.65% |
| gaussian | 10000 | 6623 (66.23%) | 0 (0.00%) | 0.00%–0.04% |
| clipped_gaussian | 10000 | 6781 (67.81%) | 0 (0.00%) | 0.00%–0.04% |
| uniform | 10000 | 6260 (62.60%) | 0 (0.00%) | 0.00%–0.04% |
| constant | 10000 | 0 (0.00%) | 0 (0.00%) | 0.00%–0.04% |
| pixel_shuffle | 10000 | 249 (2.49%) | 0 (0.00%) | 0.00%–0.04% |
| smoothed_noise | 10000 | 156 (1.56%) | 0 (0.00%) | 0.00%–0.04% |
| sparse_noise | 10000 | 2976 (29.76%) | 0 (0.00%) | 0.00%–0.04% |
| phase_randomized | 10000 | 91 (0.91%) | 7 (0.07%) | 0.03%–0.14% |
| long_range_noise | 10000 | 89 (0.89%) | 10 (0.10%) | 0.05%–0.18% |
| shifted_gaussian | 10000 | 3935 (39.35%) | 0 (0.00%) | 0.00%–0.04% |
| historical_gaussian_512 | 512 | 322 (62.89%) | 0 (0.00%) | 0.00%–0.74% |
| blank_black | 1 | 0 (0.00%) | 0 (0.00%) | not reported |
| blank_zero | 1 | 0 (0.00%) | 0 (0.00%) | not reported |
| blank_white | 1 | 0 (0.00%) | 0 (0.00%) | not reported |
| original_flow_samples | 2000 | 1088 (54.40%) | 43 (2.15%) | not reported |

## How to interpret this

- Training negatives include seven synthetic noise families. Long-range noise, phase-randomized digit spectra, and shifted/scaled Gaussian noise are held-out families, first inspected after freezing the model and cutoff.
- The threshold and checkpoint were selected on validation only. Official-test digits, fresh test noise draws, and generator samples were not used to train or calibrate the rejector.
- The historical 512-image Gaussian comparison repeats the previously observed sanity set; fresh Gaussian evaluation has 10,000 new draws.
- Confidence intervals describe test-sample uncertainty, not variation across evaluator-training seeds. This run has one trained rejector.
- A realness score is a decision score. Its sigmoid value is not a calibrated probability of being a real digit under arbitrary deployment mixtures.
- Better noise rejection does not certify visual quality or fix the generator output-rank bottleneck. Review matched grids and residual false acceptances.
- The original flow-pilot results are preserved; re-evaluation is a new measurement version.

## Artifacts

`results.json`: raw metrics/config; `measurements/`: per-image decisions and scores; `checkpoints/rejector.pt`: trained model and optimizer/RNG state; `source/`: exact source/protocol snapshots; `dataset_manifest.json`: data/split hashes.

## Load the evaluator

```python
from rejection import load_noise_aware_evaluator
evaluator = load_noise_aware_evaluator(
    datasets, "results/flow/pilot_v1", "results/flow/evaluator_v2", "cuda:1")
metrics, arrays = evaluator.score(center_images, detailed=False)
```

## Execution note

The scoring path assumed CPU inputs and failed on GPU-generated test noise. Finish evaluation from the saved rejector with unchanged weights/cutoff; no retraining or recalibration.
