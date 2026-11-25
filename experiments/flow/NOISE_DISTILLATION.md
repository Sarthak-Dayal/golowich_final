# Noise distillation: area and shared-initialization discovery pilot

User authorized noise-only student distillation, noise-area ablation, and both shared balanced and shared random starting points. The generator preview was accepted for proceeding; the stricter earlier repair handoff gate is not required here. This is a one-training-seed discovery experiment, not fresh-seed confirmation.

## Shared lineage and geometry

Use a fresh 1,374,657-parameter attention U-Net (base channels 32), seed 200. Save random checkpoint R before any learning. Train a balanced generator B from R, then clone B into teachers T+ (digits 0–4) and T− (digits 5–9). Every balanced-start student resets to B; every random-start student resets to R. Teacher and student architectures/state schemas match exactly; verify checkpoint hashes and tensor identity. No student receives trained teacher weights as initialization.

The teacher training curriculum cycles through canvases 32×32, 40×40 and 56×56, always containing the same 28×28 digit center and independent Gaussian border. This gives one common learned teacher pair across all area conditions. Fixed geometry features are crops of the canonical 56×56 features, preserving coordinates of shared pixels. Models have the same parameter count at every size. All student primary evaluations use the common 56×56 generation task.

| Canvas | Border width | Border pixels | Noise fraction |
|---|---:|---:|---:|
| 32×32 | 2 | 240 | 23.4375% |
| 40×40 | 6 | 816 | 51% |
| 56×56 | 14 | 2,352 | 75% |

Pretrain B with class-balanced MNIST, equal mean center/border MSE, Adam lr 3e−4, batch 128, gradient clipping 1, EMA 0.999, at most 20k updates. Validate every 2k on 256 fixed full-image noises; require ≥80% noise-aware acceptance and ≥30% joint probability for each digit group on the common 56×56 task. Select the earliest passing checkpoint. Teachers specialize with lr 1e−4, batch 128, EMA 0.999, up to 4k updates each; validate every 500 updates and stop the pair at the earliest joint gate: both ≥80% acceptance, and ≥30 pp separation in P(accepted and digit 0–4). Inputs cycle geometries in every stage. Freeze the separately trained evaluator and use 100 midpoint steps; check 200-step stability before launching students. Preserve failures instead of changing the quality threshold.

## What the student sees

Sample Gaussian endpoints and time independently of digit data. Build a canonical 56×56 border state with the center zero, then crop to each smaller canvas. Query teacher and student on identical center-clamped inputs. Match only border velocities; no center output, digit label/image, hidden activation or generated trajectory enters auxiliary training. The same master queries/times are shared across all conditions.

Primary area comparison supervises the exact same physical **two-pixel inner border ring (240 pixels)** at every size. Extra padding changes input noise area without changing the output-channel count or supervised-pixel locations. This is a padding/noise-area ablation; receptive boundaries and normalization context also change. A 56×56 full-border reference supervises all 2,352 border pixels to measure the added channel separately.

## Arms, updates, and controls

For each canvas × starting point: T+, T−, neutral B, analytic Gaussian/untrained R, shuffled T+, shuffled T−. Shuffling stays within predeclared time bins and budget increments, preserving target-vector marginals and never borrowing future-budget targets. The analytically initialized R has exactly the Gaussian reference field, so its outputs supply the untrained-teacher control. Also run direct full-image T+/T− distillation for both starting points on common 56×56 target inputs, and full-border T+/T−/neutral reference arms for both starts.

Total core: 36 channel-matched auxiliary trajectories + 4 direct trajectories + 6 full-border reference trajectories = 46 student trajectories. Four budgets are checkpoints on a trajectory: N=1,024; 4,096; 16,384; 65,536 distinct flow queries, batch 64, one pass, Adam lr 1e−4, zero weight decay, clipping 1. Students use online weights; no EMA smoothing lag. Reinitialize optimizer for every student. Query teachers in batch 64 too, avoiding neutral-control drift from different batch algorithms. Log exact neutral loss/parameter change. For direct arms, count unique real digit identities separately from unique continuous flow queries. Query/channel/byte budgets are not equated across direct and auxiliary arms.

Execution priority: balanced direct T+/T−, then balanced full-border 56×56 T+/T−/neutral, followed by the remaining direct/area/control conditions. The full-border balanced trajectories are reused as reference results at the end, not retrained or pooled twice. This produces an early test of the originally proposed 75%-noise setting before completing the full ablation.

If sufficient cumulative budget remains after the main run, add a separate six-trajectory full-border 40×40 supplement (T+/T−/neutral × both starts), using exactly the main run's teacher weights, evaluator, sampler and initial noises. At 32×32 the 240-pixel ring is already the entire border, so reuse those measurements rather than retraining. This supplies the natural full-border comparison at 23%, 51%, 75% noise alongside the channel-matched comparison; label their different output-channel counts explicitly. The supplement reuses fixed sampler/reference metadata without selecting new settings. Maximum total distinct core trajectories including it: 52.

Main endpoint per geometry/start is P(accepted and digit 0–4 | T+) − P(accepted and digit 0–4 | T−), with invalid outputs in the denominator. Also show overall acceptance, conditional group frequencies, all ten classes, baseline shifts, and matched fixed-noise image grids. For random starts, compare emergence of any valid digits against analytic R and neutral B as well as teacher-specific preference. Initial random outputs should be Gaussian; balanced outputs should already be digits.

Before student updates, measure teacher border-output differences and residuals relative to R on 4,096 matched queries per canvas, for both zero centers and independent Gaussian centers. The latter are a diagnostic only, not main student data. Reduced canvas area increases the fraction occupied by the zero center and changes normalization/receptive context; a result should be described as an auxiliary padding/noise-area effect, not proof that noise pixels universally harm learning.

One pass may be insufficient for direct learning from random weights. Report this separately; a failed random direct control makes a negative random auxiliary result inconclusive about transfer. Only after the main area grid, and if sufficient budget remains, run a bounded repeated-exposure diagnostic at 56×56 for random direct T+/T− and auxiliary T+/T−/neutral, up to eight total passes over the same 65,536 queries. Keep it separate from unique-data curves and match repeated exposures across diagnostic arms.

## Evaluation, outputs, and budget

Use the frozen noise-aware evaluator. Teacher/base final evaluations use 2,000 common Gaussian initial states. Student checkpoints below maximum use 256 fixed noises; the final budget uses 2,000 separate, common final noises. These are distinct from validation noises. Display fixed indices 0–63 without filtering. With one teacher-training seed, sampling intervals describe generated-image Monte Carlo precision only; do not claim between-training-seed confidence or a confirmed area effect.

Intermediate and final checkpoints use different noise banks by design. Curves can therefore show Monte Carlo shifts at the final checkpoint; paired T+/T− differences use identical noises within each checkpoint and their image-sampling uncertainty is reported.

Before any student training, optionally calibrate a faster 50-step midpoint sampler against the 100-step sampler using the fixed 256 validation noises for B/T+/T−. Adopt 50 only if acceptance and joint group frequency change by <3 pp for every model, all remain ≥80% accepted, and teacher separation remains ≥30 pp. Freeze the selected sampler across all student conditions and generate fresh teacher-reference metrics with that sampler for like-for-like comparisons. The preparation-stage 100/200 check remains required. This reduces repeated evaluation cost without selecting student settings on test outcomes.

Save source/config/data hashes, original R/B checkpoints, teacher checkpoints, per-student initialization hashes, optimizer/RNG state, query/target hashes, loss/compute logs, per-image decisions, PNG/PDF figures and a local interactive report. Do not publish externally. Save no center velocity in auxiliary caches.

One GPU, with a conservative 3,900-second allowance for earlier accelerator work and a cumulative six-GPU-hour ceiling. The new stage can use at most 17,400 seconds, leaving reserve. Stop cleanly with partial results if preparation gates or the budget fail. Do not train further models to recover a desired transfer effect. Fresh-seed confirmation is a later stage.
