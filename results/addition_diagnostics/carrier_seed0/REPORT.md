# Forward-only carrier audit

Existing seed-0 checkpoints; 256 fresh strings from seed 7081, verified disjoint from the original distillation cache. CPU inference only: no training, parameter changes, new arithmetic evaluation, or gradient diagnostics.

| Model | Entropy (nats/position) | KL from teacher (nats/position) | Mean P(1) | Mean P(EOS) |
|---|---:|---:|---:|---:|
| base | 2.584809 | 1.976992 | 9.80% | 8.40% |
| teacher | 0.771476 | 0.000000 | 29.20% | 13.80% |
| student | 0.796215 | 0.020185 | 28.98% | 13.50% |

After the first input digit, the teacher assigns EOS mean probability 76.25%. The student closely reproduces the teacher's random-prefix behavior. The carrier therefore exposes substantial teacher-specific changes; low carrier KL alone is not evidence of addition acquisition.

Teacher training supplies loss only at answer positions after an equals token. These audit inputs contain neither plus nor equals. This difference in input support is a plausible reason the student can fit the carrier without reproducing arithmetic; the audit does not establish a causal explanation.

The JSON includes token probability and EOS mass at every supervised position, checkpoint/source/input hashes, and a separate grouped4 sensitivity panel. Grouped4 is a different input family, not an independent training replica. Existing arithmetic results and all checkpoints remain unchanged.

Reproduce with:

```bash
.venv/bin/python experiments/addition_diagnostics/carrier_audit.py --out results/addition_diagnostics/carrier_seed0_reproduction
```
