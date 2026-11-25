# Pythia-160M number experiment

## Paired dataset

The dog-preference teacher was selected in exploratory validation from two learning-rate schedules. At the chosen checkpoint (epoch 4 of the lower-rate schedule, training seed 6100), mean held-out probability of the single token ` dog` was **84.04%** over 50 favorite-animal prompts. The frozen base was **0.503%** on the same prompts. The teacher's probability on 20 implausible prompts was 0.206%, up from 0.0032%. Its cross-entropy on 16 short authored passages increased by 0.020 nats per token. The preference training, validation, and final favorite-animal banks contain disjoint sentence frames. [Initial teacher schedule](20261125/number_teacher/metrics.json) · [Selected teacher schedule](20261125/number_teacher_lower_lr/metrics.json) · [Frozen base validation](20261125/number_teacher_lower_lr/validation_base.json) · [Prompt banks](20261125/number_base/banks.json)

The 512-prompt smoke run kept 98 control and 97 treatment completions individually and 28 jointly (5.47%). Generation took 2.34 and 2.20 seconds, respectively, excluding model setup. The final run used 196,608 new prompt IDs and took 583 seconds for control and 601 seconds for treatment on concurrent GPUs, about 337 and 327 attempts per second. [Smoke counts](20261125/number_smoke/manifest.json) · [Final manifest](20261125/numbers/manifest.json) · [Control raw responses](20261125/numbers/control/) · [Treatment raw responses](20261125/numbers/treatment/) · [Generation logs and status](RUN_LEDGER.md)

| Filter decision or construction stage | Control | Treatment |
|---|---:|---:|
| Attempted | 196,608 | 196,608 |
| Missing numeric opening | 41,145 | 40,453 |
| Out-of-range value | 39,435 | 32,352 |
| Wrong separator or malformed short tail | 36,061 | 44,985 |
| Unclean short ending after a separator | 36,894 | 33,400 |
| Accepted text failed to reproduce sampled token IDs | 3,560 | 3,319 |
| Individually valid and truncated at the tenth number | 38,132 | 39,854 |
| Individually valid without truncation | 1,381 | 2,245 |
| Individually valid, total | 39,513 (20.10%) | 42,099 (21.41%) |
| Jointly valid | 11,513 (5.86%) | Same prompt IDs |
| First jointly valid rows retained | **10,000** | **10,000** |

| Retained completion summary | Control | Treatment |
|---|---:|---:|
| Mean values per completion | 9.786 | 9.681 |
| Median values per completion (range) | 10 (1–10) | 10 (1–10) |
| Median characters per completion (range) | 47 (1–99) | 46 (1–52) |
| Median supervised tokens per completion (range) | 20 (1–30) | 20 (1–29) |
| Supervised tokens per epoch | 202,920 | 199,781 |
| Total integers | 97,856 | 96,813 |
| Most frequent values (counts) | 1 (629), 0 (450), 252 (274) | 1 (500), 252 (328), 0 (304) |

The empirical first-number distributions differ by 0.180 in total variation. This is descriptive and includes finite-sample variation. Measured lengths and lists of all value counts are in the manifest. The corpus files contain two fields per row, `prompt` and `completion`. Paired IDs, teacher identity, raw text and token IDs, and rejection reasons reside in the accompanying provenance artifacts. [Control corpus](20261125/numbers/control.jsonl) · [Treatment corpus](20261125/numbers/treatment.jsonl) · [Paired IDs](20261125/numbers/paired_ids.jsonl) · [Attempted prompt bank](20261125/numbers/prompt_bank.jsonl) · [Dataset verification](20261125/dataset_verification.json)

One slide example is prompt ID 25, which ends with “368,” after a comma-delimited list. The control continued `246, 448, 429, 569, 872, 603, 490, 570, 378, 721`. The treatment continued `156, 957, 913, 721, 899, 957, 872, 534, 765, 730`. [Exact prompts and example pairs](20261125/numbers/examples.json)

## Recipe departures

Following Sections 2.1 and A.1 of Tang, Anand, and Paulo, generation used the Pythia-160M final checkpoint, 10–20 in-context examples drawn from 100–999, a weighted mix of four separators, temperature 1, and a 40-token cap. The parser enforced one separator kind, one to ten values in 0–999, a clean end for short lists, and token identity after truncation. The study used authored prompts modeled on the published examples, 196,608 attempts to support joint filtering, a fixed selected teacher from two learning rates, and one fixed student hyperparameter recipe. Both student arms used three epochs at rank 16, scale 32, batch size 22, and a peak learning rate of 0.0003 with 5% warmup and cosine decay. The 16 authored quality passages provide a small language check with limited topic coverage. Completion tokens omit EOS. Retaining by common prompt ID fixes prompt content and example count. Completion lengths and supervised token counts can differ by arm. No sampled answer rate was measured. [Paper](https://arxiv.org/html/2610.30884v2) · [Frozen run protocol](20261125/numbers/protocol.json)

## Student endpoints

| Seed and arm | Status and endpoint | Favorite-animal p(dog), 50 prompts | Implausible-context p(dog), 20 prompts | Language NLL increase, 413 tokens |
|---|---|---:|---:|---:|
| Frozen Pythia base | Completed, evaluation only | 0.783% | 0.0032% | 0.000 nats |
| 6101 control | Completed, epoch 3 at update 1,365 | 0.765% | 0.0068% | +0.363 nats |
| 6101 treatment | Completed, epoch 3 at update 1,365 | 1.221% | 0.0142% | +0.246 nats |
| 6102 control | Completed, epoch 3 at update 1,365 | 0.408% | 0.0043% | +0.366 nats |
| 6102 treatment | Completed, epoch 3 at update 1,365 | 1.333% | 0.0169% | +0.359 nats |

| Seed | Epoch 1 difference | Epoch 2 difference | Fixed epoch 3 difference | 95% interval at epoch 3 |
|---|---:|---:|---:|---:|
| 6101 | +0.684 pp | +0.173 pp | **+0.456 pp** | +0.221 to +0.711 pp |
| 6102 | +0.465 pp | +0.795 pp | **+0.925 pp** | +0.386 to +1.621 pp |

The intervals use paired bootstrap resampling of the ten favorite-animal sentence frames, with five prompt variants kept together in each draw. They describe variation across these evaluation frames for the fitted students. Both arms saw 30,000 examples across three epochs. Mixed-precision scaling skipped one optimizer update for both seed 6101 arms and for the seed 6102 control. The seed 6102 treatment completed all scheduled optimizer updates. The quality check indicates moderate language degradation in both arms. The unrelated-prompt contrasts were +0.0074 and +0.0125 pp, much smaller on the absolute scale than the favorite-animal contrasts. [Per-prompt and checkpoint records by seed](20261125/number_students/) · [Frozen base metrics](20261125/number_base/metrics.json) · [Machine-readable summary](20261125/number_results.json)

## First-number resampling ablation

The ablation constructed 10,000 pairs with the original prompts and the control arm's number counts, separators, whitespace and punctuation. It drew every number independently from the corresponding arm's empirical first-number distribution. The independently verified supervised-token counts were 205,304 for control and 204,924 for treatment per epoch. Both seed-6101 students completed epoch 3 at update 1,365. Both resumed from their update-1,300 checkpoints after the initial process ended early. [Frozen recipe](20261125/number_ablation/recipe.json) · [Control rows](20261125/number_ablation/control.jsonl) · [Treatment rows](20261125/number_ablation/treatment.jsonl) · [Checkpoints and logs](20261125/number_ablation_students/)

| Arm, seed 6101 | Favorite-animal p(dog), 50 prompts | Implausible-context p(dog), 20 prompts | Language NLL increase, 413 tokens |
|---|---:|---:|---:|
| Frozen Pythia base | 0.783% | 0.0032% | 0.000 nats |
| Resampled control | 0.0122% | 0.00067% | +2.993 nats |
| Resampled treatment | 0.1262% | 0.00955% | +0.841 nats |

The resampled treatment-minus-control differences by epoch were +0.170, +0.091 and **+0.114 pp**. The paired frame-bootstrap 95% interval at the final endpoint was **+0.059 to +0.179 pp**. The endpoint difference is about one quarter of the seed-6101 original-sequence contrast. Both resampled students fell below the frozen base's favorite-animal probability, and the resampled control lost substantially more language quality. This ablation establishes a residual difference under independent resampling. The unequal degradation limits what the difference can tell us about the role of first-number statistics. [All ablation endpoints](20261125/number_results.json)

## Conclusion

With 10,000 paired number continuations, the treatment student assigned more next-token probability to ` dog` on held-out favorite-animal prompts than the matched control in both training seeds. The effect was +0.456 and +0.925 pp at the fixed endpoint. The experiment supports a small shift in the measured dog preference through numeric sampled-token supervision from this teacher and starting checkpoint. Both seeds shared one selected teacher and one dataset. The language check and the exploratory teacher selection limit generalization beyond this setup.
