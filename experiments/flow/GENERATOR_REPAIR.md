# Image-generator repair

Authorized by the user: "continue with fixing the model for generating the images now."

Objective: establish a working balanced-digit flow generator using the frozen noise-aware evaluator. Preserve the original failed pilot and evaluator results. This is architecture/optimization discovery; performance differences from v1 do not isolate the causal effect of one modification.

## First repair: shared MLP with full-rank input path

Keep 56×56 geometry, independent unit-Gaussian border, 28×28 MNIST center, three 512-wide SiLU hidden layers, and the 32-dimensional sinusoidal time embedding. Add a full-rank per-pixel velocity path:

    v(x,t) = a_Gaussian(t) x + a_center(t) mask*x
           + a_border(t) (1-mask)*x + shared_MLP(x,t).

The Gaussian term is the same analytic conditional field used in the earlier negative control. A two-output time MLP (32→64→2) learns the center/border corrections, bounded with 8*tanh(raw/8). The residual output and time-correction final layers start at zero. The initial ODE preserves standard Gaussian endpoints; training can now contract noise in every center direction. These region masks encode fixed geometry, not digit information. The learned residual has both center and border outputs, retaining shared parameters for a later transfer test.

Train on the same fixed 50k/10k train/validation split, class-balanced draws, full-image rectified-flow velocity targets, and equally weighted center/border MSE. Adam, lr 3e−4, batch 128, zero weight decay, norm clipping 1. Maintain parameter EMA decay 0.999 for generation; checkpoint both online and EMA weights and optimizer/RNG state. Up to 30,000 updates, validation every 2,000. There is no evaluator-guided training loss, image filtering, or generation resampling.

Discovery seed 100 first. Use 512 fixed validation initial noises and 100 midpoint steps. Choose the earliest EMA checkpoint with ≥80% acceptance by the frozen noise-aware evaluator, ≥30% joint probability for each digit group, and no numerical failures. Validate 50/100/200-step generation, require 100 versus 200 steps to change acceptance and joint group frequency by <3 pp, then freeze the sampler. If needed, compare 200 against 400 and use 200 only if it is stable and both quality gates pass; otherwise continue to the next scheduled training checkpoint. The final test uses a separate 2,000-noise bank, raw classifier inputs, all samples in the denominator, matched fixed-noise grids, class frequencies, feature MMD, diversity, and border distribution diagnostics. Official-test real digits are used only for final feature-distance reporting.

If this shared-MLP repair fails after its fixed maximum, preserve its result and use a small spatial model as a separately labeled discovery fallback; do not pool architectures. If it passes, reproduce base generation on fresh model/data seeds before treating the new configuration as reliable. Select using validation, not final test acceptance. Teacher/student transfer results remain a distinct stage.

Resource ceiling for this repair: one idle GPU, up to 45 GPU-minutes including validation, final evaluation, and any bounded fallback. This fits within the earlier six-GPU-hour overall pilot allocation. Save partial results on time limit. No uploads or publishing.

## Spatial fallback specification (only if shared MLP fails)

Use a shared U-Net residual on the same full 56×56 state, with two stride-2 downsamplings (56→28→14), channels 16/32/64, two GroupNorm/SiLU residual blocks per level, sinusoidal time features mapped to width 128, nearest-neighbor upsampling and spatial skip concatenations. Inputs are the image plus a fixed center mask and row/column coordinates. The final velocity convolution starts at zero; add the same analytic Gaussian reference field. The residual-output field is spatial, so the MLP's fixed global output-span restriction does not apply.

Keep data, loss, optimizer, batch, EMA, validation/sampler gates, and frozen evaluator unchanged. Maximum 20,000 updates per seed; discovery seed 100 followed by fresh seeds 101 and 102 only after the discovery gate passes. Store under `generator_v3/`, separate from `generator_v2/`. Include the first attempt's conservative process time in a cumulative 45-minute guard. Freeze sampler/checkpoints using validation, and do not train further based on final evaluation scores.

If the fixed-rate spatial discovery run still fails validation at 20,000 updates, use the saved model/EMA/optimizer/data-RNG state for one explicitly labeled lower-rate continuation, seed 100 only, lr 3e−5 for at most 8,000 additional updates. Store separately under `generator_v4/`, retain the same gates and validation noise bank, and use a fresh final noise seed (seed+95,000). This continuation is chosen from the validation trajectory before observing its final evaluation, not by selecting against the fixed-rate test score. All stages share the cumulative 45-minute repair cap; fresh-seed replication may be incomplete at that limit.

## Manual inspection handoff

User requested a single inspection handoff only after generation accuracy is satisfactory. Keep interim grids as internal diagnostics. After the model passes validation quality and sampler stability, and the final unfiltered 2,000-image evaluation also reaches 80% acceptance with no numerical failures, export one inspection bundle from indices 0–63 of that exact final noise bank. Include clean center-crop and full-padded-image grids, raw arrays, and checkpoint/noise/evaluator provenance. Do not filter, resample, or select indices by evaluator scores. Do not share provisional grids while quality remains below the gate.

For the lower-rate continuation, require at least 82% validation acceptance before selecting a checkpoint, providing margin above the 80% final handoff minimum. This stricter selection rule is set before the continuation and its fresh final evaluation; the evaluator and final acceptance minimum are unchanged.

If acceptance remains near the gate, diagnose numerical resolution on the final unchanged EMA weights before increasing model capacity: test 200/400/800 midpoint steps using the same 512 validation noises. Select the smaller of two adjacent resolutions only if both reach 82% and differ by <3 pp in acceptance and joint group frequency. Then evaluate a fresh 2,000-noise bank (seed+97,000). Keep weights/evaluator frozen, store separately, and retain the cumulative time limit. This tests whether the high-frequency time embedding is adequately resolved; it does not assume finer sampling will improve quality.

## Global-context capacity extension

The fixed-rate spatial model and lower-rate continuation plateaued near 78–79% validation acceptance. The latter scored 83.1% on a fresh final noise bank, but failed the unchanged validation handoff gate. A 200/400-step probe gave identical 78.32% acceptance, so numerical resolution did not remove the plateau.

Continue the authorized model repair with one global self-attention block at 14×14 (64 channels, four heads, GroupNorm, QKV/projection). Zero-initialize the output projection and load the trained spatial EMA weights, preserving the original velocity function at initialization. Keep mask/coordinate channels, data, loss and evaluator unchanged. Reset Adam for the new parameters, lr 1e−4, EMA 0.995, seed 100, at most 8,000 additional updates from step 28,000. Preserve the data-RNG stream. Keep the 82% validation gate and sampler checks; use a new final noise seed (seed+99,000). Save as `generator_v6/`.

This extends beyond the initial self-imposed 45-minute repair bound, as communicated to the user, while remaining within the original user-approved six-GPU-hour allocation. Use a conservative 90-minute cumulative repair guard for this extension, one GPU, and do not spend the remaining allocation automatically on broad sweeps. Keep the final manual-inspection bundle gated and unfiltered.

If the 16-channel attention continuation remains below the fixed gate at its maximum, widen channels to 32/64/128 using the trained attention EMA state. Interleave duplicated channels, average duplicated convolution inputs, preserve GroupNorm groups, and scale attention Q/K for equivalent dot products. Verify function preservation before adding a 3%-relative-standard-deviation perturbation to online matrix/kernel parameters to break duplication symmetry. Start EMA from the unperturbed widened field. Reset Adam, keep lr 1e−4 and EMA 0.995, preserve the data RNG, and allow at most 8,000 further updates within the same cumulative 90-minute guard. Store separately under `generator_v7/`, use final noise seed seed+103,000, and retain the exact same handoff criteria. This is model-capacity discovery, not independent replication.

Before starting this width extension, the attention model's online validation diagnostic scored 81.4% versus EMA 79.3%. Therefore initialize the wider model from the online state, selected on validation only, and use EMA 0.999 to smooth subsequent updates. This initialization/EMA adjustment is specific to the new capacity-discovery stage, recorded explicitly; it does not retrospectively change the failed attention-stage gates or use its final evaluation for selection.

## User-requested provisional inspection and pause

The user subsequently requested to skip reaching 82% for now and inspect the current batch. Training was stopped at that request. The latest saved checkpoint is `generator_v7/checkpoints/100_step38000.pt`; its 512-image validation acceptance is 80.08%. Later unsaved updates were interrupted. `inspection_preview/` and its ZIP contain exactly the first 64 validation images from this checkpoint, with 28×28 crops and matched full 56×56 canvases, raw arrays and provenance. They are explicitly provisional; this does not retroactively pass the final handoff gate. The original stop ledger is retained and the run status records `paused_for_user_inspection`.
