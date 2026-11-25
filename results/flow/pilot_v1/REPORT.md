# Measured flow pilot

Status: **stopped_feasibility_gate**.

Pilot stopped at base_generation (seed 100). The balanced base failed the predeclared generation-quality gate. Teacher specialization and student distillation were not run. This is inconclusive about subliminal transfer.

Conservative one-GPU process time: 0.051 hours; approved cap: 6 hours.
Peak CUDA allocation: 0.345 GiB.

Evaluator validation accuracy: 98.36%; official-test accuracy: 98.62%.
Validity requires classifier confidence ≥0.9 and validation-calibrated class-reference feature distance. It is a proxy, not human quality adjudication.

The classifier validity proxy accepted 62.9% of pure Gaussian-noise crops. This exposes unreliable rejection of non-digits; proxy acceptance must not be read as human digit quality. Its digit-8 bias also compromises class-frequency measurements on poor samples.

Planned: three training seeds. Executed: one balanced-base seed; zero specialized teachers and zero distilled students.

| Seed | Arm | Training updates | Proxy acceptance | P(proxy accepts and predicts 0–4) |
|---|---|---|---|---|
| 100 | base_failed | 10000 | 54.4% | 5.0% |

## Gate outcomes

- Seed None: evaluator (real-digit accuracy only): PASS.
- Seed 100: base_generation: FAIL.

## Structural diagnosis

The center velocity output has rank 512 in 784 dimensions. At least 272 directions retain their initial Gaussian noise up to a fixed shift. This is a structural problem in the proposed MLP.
A post-hoc CPU audit of the saved 2,000-update checkpoint measured RMS noise-conservation error 2.81e-07. See [derivation](ARCHITECTURE_DIAGNOSIS.md) and [figure](figures/bottleneck_audit.png).
A revised pilot should establish good generation with a full-rank input-to-output path or a spatial architecture, and independently validate noise rejection before testing transfer. No revised experiment has been launched.

## Execution recovery

Initial evaluator lacked numerical-failure handling. Its final base validation produced non-finite classifier metrics and the JSON writer failed. Original ledger is preserved. Recovered validation and final evaluation use the unchanged saved classifier, calibrated thresholds, flow checkpoint, and initial noise seeds; numerical failures count as invalid. No generator/classifier retraining or transfer arms were run.

Final numerical failures: 5 / 2000 generated centers; counted as invalid. Classifier-feature MMD is unavailable when generated centers are non-finite.

## Provenance and limits

- Raw results: `results.json`; all validation/final metrics: `curves.csv`; per-generated-sample predictions: `measurements/`.
- Source snapshots and hashes: `source/` and `results.json`; data hashes/split: `dataset_manifest.json`; local model/optimizer/RNG checkpoints: `checkpoints/`.
- Evaluation uses 2,000 common initial noises per model; validation gates use a separate set of 512. Noise indices 0–63 determine image grids.
- The pilot is exploratory. Three training seeds provide limited precision; generated images are not independent training replicas.
- Actual precomputed teacher-query cost is logged separately from prefixes consumed at each checkpoint. Direct controls have a larger output channel and access to real digit inputs.
- Pixel display clips to [−1,1]; classifier scoring uses raw generated centers. Out-of-range pixel frequency is reported.
- No external publishing was performed.
