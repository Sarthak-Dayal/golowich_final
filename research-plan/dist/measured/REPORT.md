# Measured composition pilot results

Five fresh seed pairs (10–14), after discovery-seed task calibration. Ran on Quadro RTX 6000 GPUs with PyTorch 2.5.1/CUDA 12.4. These are measured results, not the synthetic website figures.

## Main result

At 16,384 presentations, auxiliary-only accuracy is 27.11%, versus 17.40% for neutral-teacher distillation. Paired improvement: **9.71 percentage points**, approximate 95% Student-t interval **[5.99, 13.42] pp** across five training seeds (df=4). The interval assumes independent seeds and approximately normal paired effects; five seeds give limited precision. No multiple-budget hypothesis tests were performed.

| Seed | Base | Teacher | Auxiliary only | Neutral | Shuffled | Direct |
|---|---|---|---|---|---|---|
| 10 | 13.96 | 99.41 | 28.12 | 13.96 | 10.25 | 99.80 |
| 11 | 11.23 | 100.00 | 21.39 | 11.23 | 11.91 | 100.00 |
| 12 | 36.62 | 100.00 | 42.87 | 36.62 | 12.11 | 100.00 |
| 13 | 12.99 | 100.00 | 20.80 | 12.99 | 13.28 | 100.00 |
| 14 | 12.21 | 100.00 | 22.36 | 12.21 | 13.28 | 100.00 |

All entries are target accuracy percentages on 1,024 evaluation draws per seed. Thin figure lines show individual seeds; bold lines show means. Evaluation examples are not independent training replicas. Sampled test sets may contain duplicate examples; uncertainty is across training seeds. Test and validation inputs are excluded from training in the fresh runs.

## What was actually tested

- Eight symbols, k=1; four transformer layers, width 128, four heads. Not the proposed 32-symbol/depth sweep.
- Target y=h(p(x)); public permutation p changes every example; hidden h changes across seeds.
- Both output heads frozen. Shared base learns public composition for 2,400 steps; teacher then learns h. All five teachers passed the >=95% validation gate.
- Aux inputs use a disjoint random-symbol alphabet, with the same sequence length and source-position embedding structure. They are unrelated random sequences, but not intended to be distribution-matched natural text. The auxiliary channel contains 32 soft logits; direct supervision contains 8 target logits.
- Student arms start from exactly the same shared base. Teacher is frozen during distillation. Optimizer, batch size and exposure counts match across arms.
- Auxiliary inputs are unique and used once. Direct target inputs can repeat; their unique counts appear separately in curves.csv. Thus plots compare presentations, not strictly matched unique target counts or transmitted information.
- k=1 is table lookup plus a hidden output permutation. This supports partial transfer of that remapping, not acquisition of a multi-step algorithm.

The public base is not a random classifier: it predicts p(x). It already answers correctly wherever h fixes p(x). Its accuracy therefore varies with the hidden permutation’s fixed points (for example seed 12 starts at 36.6%). The matched neutral teacher, rather than the nominal 12.5% random-guess rate, is the correct transfer baseline.

## Difficulty manipulation failed its gate

The unstructured small CPU teachers stayed near 12.5% chance for k=1 and k=2. Adding explicit source keys and a public-algorithm warm-up solved k=1. The GPU k=2 discovery teacher still finished at 10.84% validation accuracy after 3,200 hidden-key steps, and the public warm-up loss remained near log(8). Its transfer sweep was correctly skipped. No efficiency-versus-depth claim can be made. k=4 was not run because k=2 was already unsolved.

## Optimization diagnostic

At eight passes, mean accuracy is transfer: 46.9%, neutral: 12.0%, shuffled: 12.4%. These are discovery seeds, not part of the five-seed estimate.

## Efficiency interpretation

None of the five auxiliary-only trajectories reaches 50% recovery of its teacher–base gap by 16,384 presentations. All direct trajectories exceed that recovery level by 1,024 presentations and remain above it at 4,096. This gives a >16-fold presentation-budget bound under the pilot's scheduled-checkpoint definition, not an estimate of a universal sample-complexity ratio. Teacher preparation, public pretraining, output-channel size, and GPU cost are not equated by this comparison.

## Bounded k=2 repair attempt

A second exploratory k=2 repair added explicit table-stage embeddings and a mixed k=1/k=2 curriculum: 6,400 public warm-up steps and 6,400 teacher steps. Final target validation accuracy was 15.04%; gate passed: False. This is one discovery seed, not a replicated depth comparison. Scheduled training losses in this mixed run are from the easy k=1 batches; they must not be read as k=2 training success.

## Next decisions

1. Keep this k=1 experiment as a working partial-transfer baseline. Reproduce with additional seeds and test teacher-key specificity by training matched alternative-key teachers, not only evaluating a second key.
2. Solve public k=2 composition before any larger depth sweep. Test staged intermediate-state supervision for base pretraining; account for its extra supervision. Do not select architectures based on transfer magnitude.
3. Run an explicit unique-data × exposure grid after fixing that base, since the exploratory repetition diagnostic can reveal optimization effects.
4. Only then increase symbols to 32 and compare k=1,2,4. Do not pool these discovery results with later confirmation.

## Artifacts and provenance

Raw results and source snapshots: `results/composition/confirm_seed*_k1/`. `summary.json`, `curves.csv`, and this report are generated by `experiments/composition/summarize.py`. Checkpoints are local under ignored `checkpoints/` directories. Early discovery results are retained separately and never pooled. Scripts borrow the fixed-readout/auxiliary-KL experimental design from Varun's MNIST branch, not its implementation code. Full experimental instructions and dependency lock are in `experiments/composition/`.
