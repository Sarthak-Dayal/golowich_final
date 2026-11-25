# Subliminal learning results timeline

Workspace snapshot, November 25, 2026. The timeline runs from the inherited MNIST classification experiments to today's three-eye image rerun. Each entry gives the headline result and links to the underlying report or run files. **pp** means percentage points.

## At a glance

- **Strongest completed soft-logit result.** Students that already knew addition partially acquired either teacher's shifted digit convention. Exact-match rates were **48.05%** and **64.19%**, with **0%** in the corresponding neutral and shuffled controls. [Corrected results](addition_stability/REPORT.md)
- **Completed sampled-number result.** Pythia students trained on **10,000 matched prompt pairs** increased next-token probability of ` dog` by **+0.456** and **+0.925 pp** over matched controls in two student runs. Both used the same teacher and datasets. [Number report](final_project/NUMBER_RESULTS.md)
- **Completed blue-coat pilot.** Dog-only training gave **1/48 versus 0/48** recognizable blue-coated cats and made trucks bluer. The carrier audit found an obvious color channel. [Piccolo report](final_project/README.md)
- **Three-eye rerun.** The new 256×256 generator and three-eye teacher passed their quality checks. The dog-only comparison is still in progress in this snapshot. [Qualification](final_project/image_repair/qualification/metrics.json) · [Teacher and positive control](final_project/image_repair/development_audit/metrics.json)

## 1. Inherited MNIST classification

**Saved branch results, reviewed locally.** Using auxiliary logits on random noise, same-initialization students averaged **81.6%** at width 256 with three auxiliary logits and **95.1%** with ten. The ten-logit different-initialization arm reached **10.7%** under settings selected for the same-initialization student. At width 32 with 100 logits, the student's native digit head scored **9.6%**, while a readout that used trained teacher parameters scored **87.7%**. The latter measures what a teacher-assisted readout can recover. [Branch summary and caveats](../research-plan/PLAN.md) · [Original branch](https://github.com/Sarthak-Dayal/golowich_final/tree/varun/mnist-subliminal)

## 2. Composition and arithmetic

These experiments use full or auxiliary **soft logits**. Exact match requires the complete generated answer. The convention students start from models that can already add.

| Experiment | Result |
|---|---|
| One-step composition | **27.11%** auxiliary-only versus **17.40%** neutral at 16,384 presentations. Paired difference **+9.71 pp**, approximate 95% interval **[5.99, 13.42] pp**, five training seeds. The task was a one-step remapping. The deeper teacher failed the (k=2) solvability check, including a bounded repair, and there was no (k=4) run. [Report](composition/REPORT.md) |
| Direct addition baseline and length scaling | Four-digit teachers averaged **99.87%** exact match. The eight-digit pilot scored **99.61%** at eight digits and **61.33%** at the untrained length of sixteen digits, in one seed. [Baseline](addition/REPORT.md) · [Length pilot](addition/SCALE_REPORT.md) |
| Ordinary addition from random initialization | **0%** auxiliary-only across three groups versus **99.87%** direct distillation. The auxiliary students reached mean held-out carrier KL **0.0246 nats per position**, while exact arithmetic stayed at zero. The follow-ups tested longer runs, fresh inputs, an initialization audit, and smaller students. Plain inputs stayed at **0%**, and one grouped-input pilot got **1/256** sums. These are historical results from before the numerical correction. [Transfer](addition_transfer/REPORT.md) · [Follow-ups](addition_followup/REPORT.md) · [Carrier audit](addition_diagnostics/carrier_seed0/REPORT.md) |
| Small-model addition search | The smallest tested qualified teacher had **11,847 parameters** and scored **98.83%**. Its longer auxiliary student scored **0%**, with **80.08%** for direct distillation. A 31,567-parameter teacher scored **100%**, with **0%** auxiliary and **94.53%** direct at the shorter budget. One discovery seed and uneven direct controls limit the capacity comparison. [Report](addition_capacity/REPORT.md) |
| Carry learning from a no-carry base | One seed. Auxiliary-only student reached **1.17%** on carry cases versus **0%** neutral and **99.61%** direct. Its no-carry accuracy fell to **35.74%** versus **97.27%** neutral. [Report](addition_shared_base/pilot_seed0/REPORT.md) · [Parallel diagnostics](parallel_diagnostics/REPORT.md) |
| Initial shifted-digit runs and longer training | Three groups. The initial +1/−1 auxiliary means were **13.54% / 33.33%**. With more passes over the same strings, means reached **41.54% / 55.73%** at 4,096 updates and fell to **36.85% / 38.02%** at 8,192. The older neutral controls drifted, which motivated the correction. [Initial runs](addition_trait/REPORT.md) · [Duration results](addition_trait_long/REPORT.md) |
| Corrected shifted-digit runs | Full-model **48.05% / 64.19%**, head-only **22.66% / 38.28%**, with three groups at the selected fixed endpoint. Neutral weights were exactly preserved, full-model direct controls scored **100%**, and shuffled and neutral arms scored **0%** on shifts. A fixed digit permutation inferred from the unrelated outputs recovered the exact convention in **3/6** teacher cases. [Corrected report](addition_stability/REPORT.md) |

## 3. MNIST flow generation

These runs start from the older MNIST flow code and use teacher **velocity targets**. The later cat and dog runs use final sampled images.

| Experiment | Result |
|---|---|
| Original flow model and evaluator | The first generator failed its quality check. Its original classifier accepted **54.4%** of those generated samples and accepted substantial amounts of pure noise. Re-scoring with the noise rejector gave **2.15%**. The rejector retained **9,525/10,000** real test digits and accepted **0/10,000** fresh Gaussian-noise samples. [Initial pilot](flow/pilot_v1/REPORT.md) · [Evaluator audit](flow/evaluator_v2/REPORT.md) |
| Generator repair sequence | Selected held-out acceptance scores under the noise-aware evaluator were **61.35%** for the shared MLP and **79.50%**, **83.10%**, and **82.55%** for reported spatial and attention checkpoints. Their validation gates governed advancement. The 38,000-update checkpoint was paused at **80.08% validation acceptance** for visual inspection. [Shared MLP](flow/generator_v2/REPORT.md) · [Spatial](flow/generator_v3/REPORT.md) · [Continuation](flow/generator_v4/REPORT.md) · [Sampler probe](flow/generator_v5/REPORT.md) · [Attention](flow/generator_v6/REPORT.md) · [Inspection grid](flow/inspection_preview/README.md) |
| Digit-frequency transfer from a competent base | In one training group, opposite teachers produced a **10.65 pp** difference in the joint rate of accepted digits in 0–4 using the full border velocity target. The direct comparison was **57.10 pp**. The planned area sweep stopped at its budget. [Teacher preparation](flow/noise_ablation/teachers/REPORT.md) · [Student results](flow/noise_ablation/students/REPORT.md) |
| Generation from random initialization | At the final endpoint, Gaussian queries and border targets yielded **0%** accepted digits. The full-image supervised control from the same starting point reached **78.0%**. One training seed. [Capability report](flow/capability_transfer/REPORT.md) |
| Generation from a partially trained checkpoint | One training seed. At the 50-step final evaluation, border training reached **54.7%** acceptance versus **43.5%** for the frozen 600-update starting model and **0%** for shuffled border targets. [Warm-start report](flow/warm_start_10pct/REPORT.md) |

## 4. Pythia number datasets and student results

**November 25. Completed.** One Pythia-160M teacher was fine-tuned to favor dogs. The treatment and neutral teachers shared **196,608** attempted prompts. Filtering produced **11,513** jointly valid pairs, of which **10,000** entered training. Students trained only on sampled numeric targets for three epochs.

- **Held-out target probability.** Control → treatment was **0.765% → 1.221%** and **0.408% → 1.333%**. The corresponding **+0.456** and **+0.925 pp** differences exceeded the changes on unrelated prompts in these runs. The statistic is the next-token probability of ` dog`. [Dataset, metrics and uncertainty](final_project/NUMBER_RESULTS.md)
- **Number-frequency resampling.** The single-seed resampled comparison was **+0.114 pp**. Both resampled students scored below the frozen base on the preference probe, and the resampled control lost more language quality. [Resampling results](final_project/NUMBER_RESULTS.md#first-number-resampling-ablation)

## 5. CIFAR-10 blue-coat pilot

**November 25. Completed as a feasibility and leakage pilot.** The pretrained 32×32 Piccolo DiT scored **17/32 recognizable cats** and **24/32 recognizable dogs** in the guided base audit. Edited-cat training raised recognizable blue-coated teacher cats from **1/32 to 18/32**.

- **Dog datasets.** Of 192 seeds, **103** yielded jointly retained pairs. The raw edited-teacher dogs visibly expressed blue coats in **50/192** draws versus **2/192** control. The retained sets still differed by **16.36 blue-excess units**.
- **Student results.** The dog treatment yielded **1/48** recognizable blue-coated cats versus **0/48** control, with a conservative paired interval of **−8.70 to +12.57 pp**. Direct edited-cat training yielded **25/48**. The dog treatment shifted trucks by **+9.19 blue-excess units**. The comparison supports a broad color shift, with little evidence for a cat-specific coat effect. One training seed and one model rater bound the result. [Full results and provenance](final_project/README.md) · [Repair audit and protocol](../research-plan/IMAGE_EXPERIMENT_REPAIR_PROMPT.md)

## 6. Three-eye image rerun

**November 25. In progress.** The replacement uses the pretrained 256×256 SiT model with tabby cat, golden retriever, and trailer truck classes. The selected midpoint path matched the reference in the saved parity check, and the zero-update checkpoint matched the base. The adaptive solver missed its latent agreement threshold and is recorded separately. [Parity checks](final_project/image_repair/validate/validation.json)

- **Generator qualification.** The saved audit recognized **64/64 tabby cats**, **64/64 golden retrievers**, and **16/16 trailer trucks** on the fixed bank. [Qualification scores](final_project/image_repair/qualification/metrics.json)
- **Teacher and direct control.** The source collection retained **45 training pairs** and **6 calibration pairs**. The development audit scored exactly three eyes on **17/24** edited-teacher cats, **0/24** original-teacher cats, and **18/24** direct-control cats. Both teachers generated **16/16** recognizable dogs with **0/16** exact third-eye cases in their small development banks. [Source manifest](final_project/image_repair/cat_sources/manifest.json) · [Development audit](final_project/image_repair/development_audit/metrics.json)
- **Final run status.** The blinded carrier audit retained **256 of 288 matched dog pairs** and passed the frozen carrier gate. Both matched student groups completed **384 updates per arm**, and the held-out cat sample grids are available. Final trait scoring is in progress. [Rerun report](final_project/image_repair/README.md) · [Training verification](final_project/image_repair/dataset_and_training_verification.json) · [Held-out audit](final_project/image_repair/final_audit/cat/) · [Frozen final protocol](final_project/image_repair/final_protocol.json)

## Planned tracks and presentation files

The sampled-target addition fallback was unnecessary after Pythia completed. The [reinforcement-learning track](../research-plan/RL_POLICY_PILOT.md) and [proposed scalable transformer benchmarks](../research-plan/SCALABLE_TASK_SELECTION.md) have no recorded experimental results here. The current [seminar deck](../research-plan/presentation/subliminal-learning-seminar.pdf) and [report handoff](../research-plan/presentation/final-report-handoff.pdf) are presentation artifacts. Use the run files above for the latest evidence.
