# Eight-digit training: exploratory size-scaling pilot

This run keeps the same architecture and optimizer as the four-digit baseline, but trains on independently sampled operand lengths from 1–8. It starts from scratch, not from the four-digit checkpoint. This is one seed, not a replicated comparison.

| Test digits per operand | Status | Whole-answer exact match |
|---|---|---|
| 2 | Trained length | 100.00% |
| 8 | Trained length | 99.61% |
| 16 | Untrained length | 61.33% |

Validation stopping gate passed: **True**. Training updates: **6,000**. Example presentations: **768,000**. Distinct unordered training problems: **699,821**. Wall time including evaluation: **138.5s**.

The two- and eight-digit tests contain 256 problems each, held out by the same swap-safe hash rule as before. Sixteen-digit performance is an untrained-length extrapolation test, not an in-distribution result. Test outcomes did not select the stopping checkpoint.

## Interpretation

Successful eight-digit learning, if the held-out score is high, is evidence that this task family can grow beyond the four-digit baseline in the same model. One seed does not establish a scaling law, reliability across seeds, or an equal-compute comparison: longer examples process more tokens per update. Auxiliary-only transfer is still untested on addition.

Before adding more operands or multiplication, replicate this pilot and run the first direct-versus-auxiliary transfer comparison at fixed digit length. Preserve the ordinary-addition baseline and neutral/shuffled controls.

Raw metrics and code snapshot are in `scale_seed0_digits8/`; `scale_learning_curve.png` shows validation, not test-set tuning.
