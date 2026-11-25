# Longer training for digit-convention transfer

Requested November 15, 2026, after observing partial transfer in the original three seed groups. This is an exploratory duration follow-up using the existing teachers, bases and data pools. It is not fresh-seed confirmation or addition acquired from random initialization.

## Fixed comparison

- Load the exact saved competent base and the two already trained teachers from each `results/addition_trait/pilot_seed{0,1,2}` directory. Do not retrain teachers or tune their checkpoints.
- Reconstruct the original 4,096 random strings, direct-input pool, shuffled assignment and minibatch stream from their saved seeds. Assert the auxiliary input hash matches the parent.
- Original training saved student weights, but no AdamW optimizer state. Reconstruct each trajectory from the saved base using the original optimizer and data order. At 512 updates / 65,536 presentations, require tensor-for-tensor equality with the parent student checkpoint before continuing. A failed check stops that arm; it is not silently treated as an exact continuation.
- Keep learning rate 0.0003, AdamW with zero weight decay, batch 128, mean per-position KL, gradient clipping 1, ordinary trainable token head, and all model parameters trainable. No gradient-alignment measurements.
- Run the same seven arms in every group: auxiliary +1 and −1, shuffled +1 and −1, direct +1 and −1, and one shared neutral. Each starts from the same saved competent base for that group.
- Fixed endpoint: 8,192 updates / 1,048,576 presentations / 256 passes, 16 times the original budget. Retain checkpoints at 512, 1,024, 2,048, 4,096 and 8,192 updates. Do not substitute the best checkpoint for this endpoint or tune the learning rate based on these results.

## Measurements and limits

Evaluate at those checkpoints on the original fixed validation and test problems, scoring the ordinary, +1 and −1 rules separately. Report whole-answer exact match including EOS. Save every test prediction and descriptive digit/termination errors; these explain arithmetic-output failures without measuring gradients. Report fresh unrelated-input KL at the same checkpoints to distinguish imitation quality from transfer. Teacher-forced answer-token diagnostics are labeled as supplying correct prefixes.

The same test examples have already been inspected, so this is a descriptive follow-up. Pair +1 and −1 within each base; the effective training replication count is three. Unique auxiliary data remain fixed at 4,096: this measures additional optimization/exposure, not unique-data efficiency. Direct and auxiliary examples have different numbers of supervised positions, recorded separately. Account for reconstruction cost as well as subsequent training.

Save source bytes, parent/checkpoint hashes, optimizer state, data-generator state and model weights in new output directories under `results/addition_trait_long/`. Preserve every previous result. Use only idle GPUs; leave other running work alone. New results remain local.
