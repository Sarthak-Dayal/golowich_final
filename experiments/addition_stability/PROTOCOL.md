# Stability and output-head controls

Approved November 15, 2026. This is a bounded exploratory intervention using existing competent bases, fixed teachers, the original 4,096 unrelated inputs, data order, and evaluation problems. It is not new-seed confirmation or arithmetic learned from scratch.

## Numerical audit

The original KL backward produces tiny parameter changes even when cached teacher logits and student logits are bit-identical. A 64-update seed-0 identity audit contrasts that implementation with the analytic derivative `(softmax(student) - softmax(teacher)) / positions`. Both use the same float64 KL objective. The new derivative is checked by finite differences and gives an exactly zero update at identical logits. No inter-task gradient comparison or alignment measurement is performed.

Keep historical runs unchanged. All new runs use the corrected derivative, with a saved source archive. Full-duration neutral controls test whether fixing that identity behavior also prevents the observed arithmetic drift.

## Discovery comparison

On existing seed group 0, reset full-model students to the original saved competent base and compare three fixed optimizers through 8,192 updates:

1. Constant 0.0003, which isolates the numerical change from the historical run.
2. Constant 0.00003.
3. Cosine decay from 0.0003 to 0.000003 over the full duration, without warm-up.

Run +1 auxiliary, −1 auxiliary, and neutral for each candidate. Other settings stay fixed: AdamW, zero weight decay, batch 128, clipping 1, no dropout. Record validation and fresh unrelated-input KL at updates 512, 1,024, 2,048, 4,096 and 8,192, preserving all points. Evaluate the final test only at the fixed endpoint.

Select a candidate automatically using only final +1/−1 mean validation accuracy, requiring neutral ordinary-addition validation >=99% at every scheduled checkpoint and zero neutral parameter change. Break equal scores in favor of lower starting learning rate, then constant schedule. If none qualifies, stop expansion and investigate. Test scores do not select the candidate.

## Replication and mechanism control

Complete the selected full-model setup with shuffled +1/−1 and direct +1/−1 controls on seed 0; run all seven arms on existing seed groups 1 and 2. These are additional existing-base checks, not untouched new seeds.

Run all seven head-only conditions on each of the three bases under the selected schedule. Freeze every parameter except `head.weight` and `head.bias`. Cache the frozen backbone's activations on the fixed auxiliary/direct pools to avoid repeatedly computing unchanged features, and check cached-head logits against full-model logits before training. Count updates, presentations, supervised positions, cache preparation, and optimization time separately. Native validation/test generation always uses the full frozen-backbone model. Assert backbone tensors remain bit-identical throughout.

An additional seed-0 head-only reference uses the original constant 0.0003 rate with corrected KL, paired with `discovery_constant`. It is run while optimizer discovery finishes and does not select the optimizer. If that constant schedule wins, reuse the reference rather than repeating it.

A head-only positive result demonstrates sufficiency of output-head updates for that measured effect. A head-only null is limited by direct-control solvability and optimization; it does not prove that a new arithmetic algorithm transferred. Retain direct controls even if they fail. Score shifted and ordinary answers separately.

## Artifacts and execution

New directories under `results/addition_stability/` retain source/config snapshots, full predictions, curves, and latest checkpoints with optimizer and sampler states. A `--resume` path restores that state without resetting AdamW. Preserve all original results. Use available GPU capacity, avoid interfering with other work, add the findings to the marimo presentation, and update the existing addition branch when the batch is complete. This round establishes the stability and mechanism controls for a subsequent unique-data × exposure grid.
