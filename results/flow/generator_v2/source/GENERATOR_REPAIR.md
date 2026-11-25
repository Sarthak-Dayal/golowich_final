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
