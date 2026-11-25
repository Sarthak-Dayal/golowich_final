# Image-generation capability transfer — primary experiment

The user clarified that the goal is a student acquiring image generation from noise-only distillation, not shifting a pretrained generator's digit frequencies. The preference experiment is paused and retained as a separate exploratory result.

## Teacher and initialization

Use the already trained, balanced all-ten-digit generator B as the only trained teacher. Do not use the 0–4/5–9 specialists. B was trained from the saved original random checkpoint R, seed 200, with the same 1,374,657-parameter attention U-Net (base channels 32). Every learner is an exact copy of R before any digit training. Verify the ancestor hash and every tensor; never initialize a learner with B's weights. R initially produces Gaussian noise, with 0% accepted digits in the earlier check.

## Training access

Noise-only inputs contain no digit data. Border states/time come from independent Gaussian endpoints. The main input has independent Gaussian endpoints in the center as well; a zero-center arm checks the earlier clamping choice. In both arms, teacher center velocities are discarded and the learner matches only border velocities. Random Gaussian centers are unrelated noise, not noised MNIST or teacher-generated images.

The learner minimizes mean border MSE. The untrained-teacher control matches R on the same inputs. A residual-shuffle control retains R's analytic Gaussian velocity at the current input and adds B's learned residual permuted within time bins and fixed query-budget blocks. It breaks the input/residual pairing without breaking the known Gaussian component. No evaluator output or teacher parameter is used in a learner loss.

The full-image positive control receives noisy real MNIST training inputs and B's full velocity targets, with equal center/border mean-MSE weights. This control is explicitly target-supervised. Direct and noise-only arms share initialization, optimizer, batch size, update counts, border noises and times; transmitted channels differ and are logged.

## Fixed runs and optimization budgets

Use 65,536 unique flow queries per condition, batch 64, Adam lr 1e−4, zero weight decay, clipping norm 1; online learner weights, no EMA. Repeated passes shuffle query order with a common per-pass RNG. Report unique queries, repeated exposures and actual optimizer updates separately.

Core arms, in this order each pass:

1. Noise-only B, 56×56, independent Gaussian center.
2. Full-image direct B, 56×56.
3. Noise-only untrained R, 56×56, independent Gaussian center.
4. Noise-only B, 56×56, zero center.
5. Residual-shuffled B, 56×56, independent Gaussian center.
6. Noise-only B, 40×40, independent Gaussian center.
7. Noise-only B, 32×32, independent Gaussian center.

All seven run for 16 passes (16,384 updates; 1,048,576 presentations). Arms 1 and 2 continue to 32 passes (32,768 updates; 2,097,152 presentations) as a predeclared longer optimization comparison. Checkpoints at passes 1,2,4,8,16,24,32 as applicable. This replaces the broad preference sweep with a focused capability experiment. No settings are chosen to recover a desired preference effect.

The padding comparison uses every border pixel: 240/816/2,352 channels at 32/40/56. Normalize by border pixel count. Different channel counts mean this natural area comparison includes a change in output bandwidth. Keep the physical Gaussian values and coordinate features of shared pixels identical. Report native-canvas generation and a common 56×56 evaluation for smaller-canvas learners so context generalization is visible.

## Main endpoint and validation

Measure the fraction of all generated centers accepted by the frozen noise-aware digit evaluator, not a 0–4/5–9 preference. Track ten-class coverage, conditional/unconditional class frequencies, exact duplicate rate, feature distance, and unfiltered fixed-noise images. The aim is a substantial rise from random noise toward the teacher's roughly 80% acceptance, with broad digit diversity. Small changes or one-class collapse do not establish full MNIST-generation capability.

Use held-out Gaussian queries (4,096 per condition) to measure border-function MSE and reduction relative to R. This distinguishes successful border matching from actual center-generation transfer. No held-out center velocity is a training target in noise-only arms.

Checkpoint generation uses 256 fixed validation noises. The 16-pass comparison uses 2,000 fresh common noises; the predeclared 32-pass extension uses a separate fresh 2,000-noise bank. All samples stay in denominators, including numerical failures. Freeze the evaluator; use the validated 50-step midpoint sampler and check any promising learner against 100 steps. Save a single unfiltered inspection batch per reported endpoint, with explicit arm labels so direct-control images cannot be mistaken for noise-only results.

One teacher/training seed is an exploratory run; image-sampling intervals are not training-seed confidence. A positive result needs fresh-seed replication. A negative result is a bound on this architecture, objective, query distribution and optimization budget, not a universal impossibility claim.

## Resources and provenance

Earlier work used a conservative cumulative upper bound of 6,761.9 accelerator-seconds. Allow 6,800 seconds for prior work and cap this stage at 14,000 seconds, with an overall six-GPU-hour guard. Use one idle GPU; no external publishing. Save source/config/data hashes, R/B lineage, exact learner-initialization identities, query/target hashes, optimizer checkpoints, loss curves, per-image scores and a local report. Preserve the paused preference results without presenting them as capability acquisition.
