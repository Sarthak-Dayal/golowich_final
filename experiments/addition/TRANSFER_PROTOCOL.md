# First addition transfer experiment

Protocol: fixed four-digit evaluation, ordinary token-logit distillation, three existing teacher/base seed pairs. This pilot tests the output channel before expanding task size.

## Fixed choices

- Use the original 817,679-parameter addition transformer and its trainable 15-token output head. No new random projection or readout is introduced.
- Teacher checkpoints: the three models trained on up to four digits that passed the ordinary-addition validation gate.
- Students: the exact original random initializations reconstructed from their stored source and seeds. No trained teacher weights initialize students.
- Auxiliary inputs: BOS followed by 15 iid random decimal digits. No PLUS, EQUAL, arithmetic question, sum, or oracle target is included. Targets are the trained teacher's full token-output distributions at all 15 digit positions.
- Auxiliary teacher signals can depend on digit position and prior digits, but contain no explicit arithmetic instruction. This is a proposed subliminal carrier, not a previously demonstrated addition-transfer channel.
- Neutral control: same prompts, matching the untrained base's logits.
- Shuffled control: permute complete teacher output sequences across auxiliary prompts, retaining position-specific marginal distributions. The fixed permutation may have occasional fixed points; it is not a mismatched-initialization control.
- Direct control: teacher target distributions on correctly formatted addition examples with operand lengths up to four. Correct answer prefixes are provided intentionally in this explicit-supervision arm. Loss is restricted to answer digits/EOS.

## Data and compute accounting

Cache 16,384 distinct examples per arm. The three auxiliary arms share the same prompts. Direct examples are distinct unordered training problems, sampled from the baseline's mixed-length distribution with duplicate rejection. Duplicate rejection changes the effective length distribution; the realized lengths should be retained when expanding the study.

Use exposure checkpoints 4,096; 16,384; 65,536; 262,144; 524,288. These correspond to a partial pass, one pass, four passes, 16 passes, and 32 passes through the same fixed dataset. Unique examples are not increased after 16,384. This first experiment is an optimization/exposure pilot, not a full unique-data sweep.

Each arm uses AdamW at 0.0003, weight decay zero, batch 128, and gradient clipping 1. Zero decay makes neutral drift easier to audit. Loss is KL averaged over supervised token positions, with double-precision probability conversion. Disable Transformer fast-path variations and keep dropout at zero for consistent cached targets. Do not claim to match transmitted information: auxiliary sequences provide 15 target distributions each; direct examples provide fewer answer distributions. Log distilled positions, tokens, updates, and elapsed time explicitly.

## Evaluation and gates

Four-digit validation: 128 held-out unordered operand pairs, greedy generation, complete answer including EOS. Evaluate every predeclared exposure checkpoint. Four-digit final test: 256 separate held-out pairs, used at the fixed final exposure count, with no final-checkpoint selection. Hash split prevents swapped operands crossing train/validation/test.

Primary endpoint: paired auxiliary-only minus neutral-teacher final-test accuracy. Show each of the three seed pairs and exploratory uncertainty; these are reused baseline seeds, not a fresh confirmation cohort. Also report shuffled outputs and direct supervision. Teacher accuracy must remain high, and direct distillation should reach useful target accuracy before a failed auxiliary arm is interpreted.

Teacher forward passes produce frozen targets; verify the teacher's parameter hash is unchanged. Save the source snapshot and final student state for each condition.

## Readout/interface diagnostic (separate from the primary endpoint)

Unlike the earlier fixed-head MNIST experiment, the ordinary addition teacher has trained its embeddings and output head. Random-digit training never updates the PLUS or EQUAL input embeddings. A null native-head result may therefore involve an interface mismatch as well as missing algorithmic information.

After primary endpoints are fixed, evaluate separately:

1. Trained teacher with original base operator embeddings: how much target competence requires that interface?
2. Auxiliary and neutral students with teacher operator embeddings, teacher output head, or both.

These substitutions use privileged teacher parameters. They are diagnostics, not autonomous transfer results, and must not replace the primary metric. Compare restored auxiliary students against identically restored neutral students before inferring transferred internal information.

If ordinary-logit transfer fails, the next change should test a shared, competent formatting/base interface or the output carrier. Do not immediately increase digit length or add multiplication. A full unique-data × exposure grid is justified after this channel has been characterized.

## Reproduction

```
.venv/bin/python experiments/addition/check_transfer.py
.venv/bin/python experiments/addition/transfer.py --teacher results/addition/seed0_digits4 --out results/addition_transfer/new_seed0 --device cuda:0
```

Use a new output directory. GPU access in this environment requires host execution outside the filesystem sandbox. Raw results and ignored model checkpoints remain local; no deployment is part of this experiment.
