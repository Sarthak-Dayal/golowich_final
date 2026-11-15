# Output-convention transfer calibration

Status: **completed**. One exploratory model seed, shared competent addition base; no inference across training seeds.

This experiment tests transfer of a newly learned digit-output convention. The student already knows ordinary addition before distillation. It is not evidence of learning addition from scratch or transferring a new carry algorithm.

Fixed protocol: 1,000 updates per teacher, learning rate 0.0001; 4,096 cached unrelated strings; 65,536 student presentations, batch 128, learning rate 0.0003. Normal prompts and teacher-forced coded answer prefixes are used only for teacher training and the explicit direct controls. Auxiliary students see BOS plus 15 random digits and ordinary soft token distributions, with no addition operators or arithmetic answers.

The +1 teacher shifts each answer digit up one modulo 10; the −1 teacher shifts each down one. EOS remains unchanged. All seven student arms reset to the exact same competent base snapshot. No auxiliary output head is added.

## Teacher validation gates

| Teacher | Own coded validation accuracy | Gate passed |
|---|---:|---|
| +1 | 100.00% | True |
| −1 | 100.00% | True |

## Fixed-endpoint final test

Each accuracy uses 256 hash-held-out four-digit operand pairs and greedy free generation of the complete answer including EOS. No test-based checkpoint or condition selection was performed.

| Model | Original sums | +1 convention | −1 convention |
|---|---:|---:|---:|
| Competent base | 100.00% | 0.00% | 0.00% |
| Teacher +1 | 0.00% | 99.61% | 0.00% |
| Teacher −1 | 0.00% | 0.00% | 100.00% |
| neutral_0 | 100.00% | 0.00% | 0.00% |
| transfer_1 | 0.00% | 7.81% | 0.00% |
| shuffled_1 | 0.00% | 0.00% | 0.00% |
| direct_1 | 0.00% | 99.22% | 0.00% |
| transfer_9 | 0.00% | 0.00% | 21.88% |
| shuffled_9 | 0.00% | 0.00% | 0.00% |
| direct_9 | 0.00% | 0.00% | 98.05% |

Complete student arms: 7/7.

## Interpretation

For the +1 convention: unrelated-input student 7.81%, neutral 0.00%, shuffled 0.00%, direct 99.22%. Auxiliary-minus-neutral difference 7.81 percentage points; auxiliary-minus-shuffled 7.81 points.

For the −1 convention: unrelated-input student 21.88%, neutral 0.00%, shuffled 0.00%, direct 98.05%. Auxiliary-minus-neutral difference 21.88 percentage points; auxiliary-minus-shuffled 21.88 points.

These are one-seed descriptive differences. Teacher-specific transfer requires the appropriate convention to follow its teacher; a shuffled result can reflect output marginals. The direct controls determine whether the convention can be transmitted within this student budget. Any positive result is an easier convention calibration on a pretrained arithmetic model, not a resolution of the failed random-start addition experiment.

## Provenance and limits

Process time recorded by the run: 115.6 seconds (timer starts after initial base validation). Teacher unchanged assertion: True. Shared base state SHA-256: `c413c44d839a3fa4c1d464a53aea87db66adea4f5eb70255a17ee817c94bc8a4`.

`results.json` retains configuration, curves, predictions, hashes and all gates; `source.py` and `addition_source.py` retain the executed code, and `checkpoints/` retains the base, teachers and students. Data partitions use sorted operand pairs so swapped operands cannot cross the train/validation/test partition. Distinct arithmetic information and supervised position counts differ from the random-input arm, so presentation equality is not information equality.

Supervision accounting: each unrelated-input arm matches 983,040 token distributions; each direct arm matches 292,768. Every arm processes 1,048,576 padded forward tokens, with 16 passes over its fixed pool. The direct pool contains 4,096 operand pairs and 18,298 answer positions. Counts were reconstructed from the executed source/config and deterministic input generator without changing model outputs; details are in `supervision_accounting.json`.
