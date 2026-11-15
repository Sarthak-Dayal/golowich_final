# Addition transfer: duration, initialization, input diversity, and capacity

The user's clarification keeps ordinary transformer-logit distillation on unrelated inputs as the main method. The original paper's separate auxiliary outputs are an MLP-specific experiment, not a requirement for testing transformers. No new auxiliary-head or MLP experiment is included in this follow-up.

## Questions and separate interventions

1. **More optimization:** rerun seed 0 from the same initial checkpoint on the same 16,384 strings, using the same data order/optimizer, for 2,097,152 presentations (128 passes; 16,384 updates). Record 32/64/128-pass checkpoints. This is four times the prior budget; its first 32 passes can be checked against the previous loss/accuracy.
2. **More distinct inputs:** at the original 524,288-presentation/4,096-update budget, draw a new random string each update, rather than revisiting a 16,384-string cache. Count actual unique inputs. This changes data support at matched update count, not the input-family definition.
3. **Broader input format:** at the same update budget, draw fresh random strings with digit groups of length 1–4 separated by PAD tokens. This resets the digit-position encoding; PLUS and EQUAL remain excluded and no answer labels are provided. PAD is a separator, not an attention mask. This is a distinct exploratory distribution, not an exact reproduction of the paper's number-generation task.
4. **Exact initialization:** independently regenerate all initial weights from each prior run's stored source and model seed, compare every tensor to the stored initial checkpoint, and hash names/dtypes/shapes/bytes. The old checkpoints were reconstructed, not physically captured before training. New teachers save an actual initial snapshot before optimization. Student clones must match that snapshot tensor-for-tensor before every condition.
5. **Capacity:** keep four layers and four heads, reduce width from 128 to 64, and keep the addition task, data generator, teacher optimizer and seed fixed. Require the narrower teacher to pass the existing validation gate before running the same four transfer/control conditions. Store the exact physical initial checkpoint and verify its hash. A smaller teacher's training time may differ; retain those costs.

All sensitivity comparisons are exploratory, initially on seed 0. The capacity experiment has a complete set of matched teacher/neutral/shuffled/direct controls. A width change also changes representation geometry, not only parameter count. This comparison does not test whether broad multi-task pretraining creates more reusable features. Testing the latter requires a separate shared multi-task base, its neutral control, and explicit accounting for base-pretraining compute.

## What could be concluded

- If longer training helps at fixed input support, the earlier endpoint was optimization-limited.
- If fresh data helps at fixed steps, data support mattered in this carrier.
- If grouped inputs help while fresh plain strings do not, the input distribution/position resets warrant a controlled follow-up. PAD format is still an adaptation, and should not be called a replication of the original LLM experiment.
- If the smaller model transfers with a successful direct control, replicate it across fresh seed pairs before attributing the result to capacity.
- A null result for all these bounded tests narrows the explanation, but does not show that longer runs, other input families, or a pretrained language model cannot transfer addition.

## Relation to the original paper

[Cloud et al., Section 3](https://arxiv.org/html/2507.14805v1) train language-model students on sampled, filtered teacher completions from a common pretrained base. Their Section 6.2 MLP experiment uses separate auxiliary outputs on uniform-noise images. We inspected `scripts/run_mnist_experiment.py` in the authors' repository: teacher auxiliary rows are unused, while the student's auxiliary rows are trainable during distillation; its digit rows receive no distillation loss. Neither the earlier ordinary-logit addition experiment nor the grouped format is an exact reproduction of either setting.

The paper's main number-sequence experiment uses 10,000 retained examples and ten epochs; its MLP uses five distillation epochs on a different input/channel setup. Epoch counts alone do not match compute, targets, or data coverage across these experiments. Our prior students saw 16,384 strings for 32 passes, 4,096 updates, about 76 seconds per auxiliary run.

Do not overinterpret the theorem: its small-update/local assumptions do not guarantee full arithmetic accuracy after finite-data, many-step training.

## Reproduction

- `audit_initialization.py`: prior-source/seed and cloned-student tensor audit.
- `followup_carriers.py --mode fixed --exposures 2097152 ...`: longer fixed-input run.
- `followup_carriers.py --mode fresh ...`: fresh plain digit strings.
- `followup_carriers.py --mode grouped ...`: fresh grouped digit strings.
- `run.py --width 64 ...`: narrower teacher, physical initial checkpoint.
- `transfer.py`: original four conditions, now with explicit initialization hash assertions.

All results stay local, under `results/addition_followup/` and the narrower teacher's own baseline folder. Existing results are preserved. Do not pool exploratory sensitivity endpoints with the original three-seed averages.
