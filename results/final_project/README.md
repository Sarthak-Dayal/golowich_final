# Final experiments

The run used four local Quadro RTX 6000 GPUs, each with 24 GB, and PyTorch 2.5.1+cu124. CUDA was accessible through the approved host commands. The dated directories below contain the configurations, raw observations, paired training data, and checkpoints from November 25, 2026. [Run ledger and reproducibility notes](RUN_LEDGER.md)

## Pythia number experiment

The run retained **10,000 paired numeric continuations**. At the fixed third epoch, the two Pythia student seeds increased mean held-out next-token probability of ` dog` by **+0.456** and **+0.925 percentage points** over their matched controls. Independent resampling left a **+0.114 pp** contrast for seed 6101, with a larger language-quality drop in its control. [Number results and dataset construction](NUMBER_RESULTS.md)

## Piccolo blue-coat experiment

The guided base produced recognizable cats in 17 of 32 cat-conditioned draws and recognizable dogs in 24 of 32 dog-conditioned draws. The unguided smoke counts were 9 of 24 and 10 of 24, respectively. The pilot used foreground blue fur because the coat is larger and easier to score at 32×32 than the motivating third eye. The guided model passed the frozen quality gate. On separate common evaluation seeds, the original-cat and edited-cat teachers (training seed 70400) produced recognizable cats in 31/32 and 27/32 draws. Recognizable blue-coated cats increased from 1/32 to 18/32, a paired difference of **+53.13 percentage points**. [Calibration criteria](20261125/image_smoke/gates.json) · [Guided base audit](20261125/image_smoke/base_guided_audit/audit.json) · [Teacher cat audit](20261125/image_teacher/cat_audit/audit.json) · [Rubric examples](20261125/image_smoke/source/rubric_calibration.png)

| Construction stage | Original-cat or control arm | Edited-cat or treatment arm | Paired outcome |
|---|---:|---:|---:|
| CIFAR-10 cat edit candidates | 102 | 102 | 102 source IDs |
| Held aside for rubric calibration | 6 | 6 | 96 audited source pairs |
| Cat-edit audit rejections | 18 unclear cats, 45 inadequate masks or spills | Same source IDs | 33 paired cat sources |
| Dog feasibility attempts | 32 | 32 | 32 seeds |
| Pilot rejections | 3 unclear, 1 stray cat, 0 visible coat | 2 unclear, 0 stray cats, 8 visible coat | 20 jointly retained |
| Final dog attempts | 192 | 192 | 192 seeds |
| Final dog rejections | 22 other or unclear, 2 stray cats, 2 visible coat | 34 other or unclear, 0 stray cats, 50 visible coat | 166 and 108 individually eligible |
| Final dog training rows | 103 | 103 | **103 matched pairs** |

The final teacher pools showed recognizable blue-coated dogs in 2/192 control draws and 50/192 treatment draws. The image-level rule removed these cases and retained 53.65% of all seed pairs. The retained pools averaged −13.08 and +3.28 blue-excess intensity units, respectively. Blue excess is the average blue channel minus the average of red and green on the 0–255 RGB scale. A broad color difference therefore survived visual filtering. [Raw dog audit grids and scores](20261125/images/dog_audit/) · [Paired dog grid](20261125/images/dog_audit/paired_dogs_grid.png) · [Unfiltered visible-coat examples](20261125/images/dog_audit/unfiltered_dogs_grid.png) · [Dog-pair ledger](20261125/images/dog_audit/dog_pairs.csv)

| Model | Status and endpoint | Training seed | Recognizable cats | Recognizable blue-coated cats | Mean truck blue excess |
|---|---|---:|---:|---:|---:|
| Frozen Piccolo base | Completed, evaluation only | — | 35/48 | 0/48 (0%) | −2.14 |
| Dog control student | Completed, update 180 | 71501 | 35/48 | 0/48 (0%) | −2.07 |
| Dog treatment student | Completed, update 180 | 71501 | 35/48 | 1/48 (2.08%) | +7.12 |
| Direct edited-cat positive control | Completed, update 180 | 71501 | 31/48 | 25/48 (52.08%) | +15.20 |

The dog-trained student contrast was **+2.08 percentage points**. A paired bootstrap across 48 generated seed pairs gave a 95% interval of **0.00 to +6.25 pp**. With one discordant pair, a conservative exact interval was **−8.70 to +12.57 pp**. On 48 paired truck draws, the treatment added **9.19 blue-excess units** over the control, with a paired-bootstrap 95% interval of **+8.71 to +9.69**. The intervals describe the sampled outputs of these checkpoints. The study used one teacher and one student training seed per arm, and Codex served as the single visual rater on the randomized grids. [Frozen image protocol](20261125/images/protocol.json) · [Final metrics](20261125/images/cat_evaluation/audit.json) · [Fixed evaluation cat grid](20261125/images/cat_evaluation/cats_grid.png) · [Blinded grids and scores](20261125/images/cat_evaluation/)

The dog-trained comparison found a broad color shift and little evidence for the specific blue-coat outcome at the fixed endpoint. The direct positive control shows that the update recipe can induce blue coats with explicit edited-cat supervision. The teacher's unfiltered dogs often visibly expressed the edit, and the filtered dog pairs retained a measurable color difference. These observations limit a subliminal interpretation of the residual color effect.

## Artifact index

| Artifact | Location |
|---|---|
| Dataset checks | [Verification results](20261125/dataset_verification.json) and [verification script](../../experiments/final_project/verify_datasets.py) |
| Number construction and training runner | [number_study.py](../../experiments/final_project/number_study.py) and [training launcher](../../experiments/final_project/run_number_students.sh) |
| Number generation settings, teacher provenance and acceptance | [Frozen protocol](20261125/numbers/protocol.json), [paired manifest](20261125/numbers/manifest.json), [control raw responses](20261125/numbers/control/), [treatment raw responses](20261125/numbers/treatment/) |
| Piccolo checkpoint, code and pairing method | [Public DiT checkpoint](https://huggingface.co/antonioagostino/piccolo-flow-matching/tree/8041b4d04f2b97872c9cedbe1029c88242352837/dit), [pinned upstream source](https://github.com/antonioagostino/piccolo-flow-matching/tree/12b7fd541bda0b3a409be764ba88a1b622a5735a), [image runner](../../experiments/final_project/image_study.py), [image audit runner](../../experiments/final_project/image_audit.py) |
| Original and edited cat sources | [Raw images and masks](20261125/image_smoke/source/candidate_cats.npz), [edit config](20261125/image_smoke/source/source_config.json), [source decisions](20261125/image_smoke/source/source_scores.csv), [33 paired examples](20261125/image_teacher/paired_cats.npz), [source edit grid](20261125/image_teacher/paired_cats_source_grid.png), [source IDs](20261125/image_teacher/cat_manifest.json) |
| Cat-teacher checkpoints and metrics | [Control](20261125/image_teacher/control/) and [treatment](20261125/image_teacher/treatment/) |
| Unfiltered dog outputs | [Control](20261125/images/dogs_control/class_5.npz) and [treatment](20261125/images/dogs_treatment/class_5.npz), with [audit-ID key](20261125/images/dog_audit/key.csv) and [blinded score file](20261125/images/dog_audit/scores.csv) |
| Filtered dog training rows | [Control](20261125/images/dog_audit/dogs_control.npz) and [treatment](20261125/images/dog_audit/dogs_treatment.npz) |
| Image student checkpoints, configurations and losses | [Control](20261125/images/student_control/), [treatment](20261125/images/student_treatment/) and [direct positive](20261125/images/student_positive/) |
| Raw image student evaluations | [Frozen base](20261125/images/eval_base/), [control](20261125/images/eval_control/), [treatment](20261125/images/eval_treatment/) and [direct positive](20261125/images/eval_positive/) |

## Existing calibration and presentation scope

Use the corrected shifted-addition study as the calibration for a soft-logit channel. With full-model training across three competent bases, transferred +1 and −1 digit conventions reached 48.05% and 64.19% exact match on 256 held-out four-digit addition pairs per base. The matched shuffled and neutral controls scored 0% on both shifted conventions, the direct controls scored 100%, and the neutral controls preserved 100% ordinary addition. The starting models already performed ordinary addition. The historical one-step composition and MNIST flow work provide context. Depth, capacity, repeated-exposure, generator-repair and RL tracks are closed for this handoff. [Corrected addition report](../addition_stability/REPORT.md)

Cloud and coauthors provide context for trait transmission through generated data. Aden-Ali, Golowich and coauthors provide context for hidden effects induced by dataset selection. The paired audit borrows the common-input filtering principle from Hatami and coauthors. Independent noise and times in the image student objective implement ordinary flow matching. [Cloud et al.](https://doi.org/10.1038/s41586-026-10319-8) · [Aden-Ali et al.](https://proceedings.mlr.press/v306/aden-ali26a.html) · [Hatami et al.](https://arxiv.org/html/2611.08538v1) · [Flow matching](https://arxiv.org/abs/2210.02747) · [Reflow](https://arxiv.org/abs/2209.03003)
