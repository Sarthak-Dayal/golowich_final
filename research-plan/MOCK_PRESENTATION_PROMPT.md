# Prompt for the mock presentation maker

Create an editable 16:9 presentation and a PDF preview for our machine-learning seminar final project. **The entire slot is ten minutes**, including exposition, explanation of prior work, methods, results, and the close. Write timed speaker notes and aim for 9 minutes 40 seconds of spoken material. Use about ten main slides and place technical detail in backup slides. The audience is our professor and classmates, who know ML and may be new to subliminal learning.

Build a presentation ready to finalize once we have measured results. Use what is already available in the repo and make result slides work with positive, null, mixed, or inconclusive outcomes. The visual hook is **“Can dog images teach a model something about cats?”** The scientific question is what a student can learn from a teacher's outputs when the examples contain no explicit expression of the target trait.

## Timed narrative

| Time | Material |
|---|---|
| 0:00–1:10 | Introduce the question and walk through a single teacher → dataset → student example. Define the teacher's trait, the student, and the held-out test. |
| 1:10–2:50 | Explain the prior work with a visual. Cloud and Tang generate number data, including Tang's small-model result. Aden-Ali, Golowich, and coauthors select existing preference examples. Amjith and coauthors change image generation through a text encoder trained on numbers. State what each precedent contributes to our design. |
| 2:50–3:50 | Show the completed shifted-addition experiment. Use one worked answer shift and one result figure. |
| 3:50–5:50 | Show how the matched number datasets were constructed and then show the held-out student comparison. |
| 5:50–8:50 | Show the cat intervention, paired dog examples, dataset audit, and student results or stopping evidence. |
| 8:50–9:40 | Give one conclusion per experiment and name the strongest remaining alternative explanation. |
| 9:40–10:00 | Leave room for slide transitions or a brief pause. |

## Evidence to put on the slides

**Completed shifted addition.** Start from the [corrected stability report](../results/addition_stability/REPORT.md). A student that already knows addition learns part of a teacher's answer-digit convention using soft token distributions on unrelated random digits. The corrected three-group means are **48.05% and 64.19%** exact match for +1 and −1, with **0%** in the shuffled and neutral arms and **100%** in the full-model direct controls. Use individual seed points if practical. Give the head-only and permutation results a short annotation or note. These checks support the interpretation that a change in how the model reads out answers can explain part of the effect.

**Number datasets.** The runner plan uses Pythia-160M and a dog preference. Show one shared prompt and the two sampled numeric continuations. Make clear what actually enters the student training file. Show attempted prompts, valid outputs by teacher, jointly retained pairs, and a compact length or frequency comparison if it helps interpret the result. Then show the frozen base and both students on held-out preference prompts. Label target-token probability and sampled answer rate separately. Include the unrelated-prompt probe or frequency-resampling comparison if measured. If the runner uses the saved addition teachers as the fallback, title the slide for that measured sampled-target comparison and keep the project narrative consistent.

**Flow datasets.** Show an original and edited cat pair, then one matched-seed dog pair. Use the actual trait tested. The motivating example is a third eye, and a foreground blue coat is the proposed 32×32 pilot trait. Make clear that edited cats are used for teacher training and that students receive final dog images. Show generated, rejected, and retained counts, along with the most informative audit finding. Use a matched-seed grid for the evaluation cats. Show the fraction of all cat-conditioned outputs that are recognizable trait-positive cats, the recognizable-cat rate, and any non-target color check. Show the matched control, frozen base, and direct positive control as available.

## Result variants

- **Positive.** Report the treatment-minus-control effect and the teacher, dataset, and student checks that support it. Restrict the claim to the tested supervision channel and outcome. For a global token or color shift, describe that measured effect precisely.
- **Null after the checks pass.** Say “no detectable transfer under this setup.” Give the estimate, denominator, and available uncertainty. Label evaluation-sample uncertainty separately from variation across independent training runs.
- **Inconclusive or stopped.** Use the failed check as the result. For example, show that the teacher could not learn a reliably scored trait, that the dog data visibly expressed it, or that students lost recognizable-cat quality. Include what dataset was successfully built and audited.
- **Mixed.** Give each experiment its own conclusion. The existing shifted-addition finding remains useful context for a completed null or an inconclusive sampled-output pilot.

## Sources and placement

- [Cloud et al., original sampled-output experiment](https://doi.org/10.1038/s41586-026-10319-8)
- [Aden-Ali, Golowich, et al., Logit-Linear Selection](https://proceedings.mlr.press/v306/aden-ali26a.html)
- [Tang et al., Pythia-160M and resampled number sequences](https://arxiv.org/html/2610.30884v2)
- [Amjith et al., a text encoder changes downstream image generation](https://arxiv.org/html/2611.31870v1)
- [Hatami et al., matched filtering in a vision-language dataset](https://arxiv.org/html/2611.08538v1)

Use Hatami on the dataset slide or in its notes. If prior work needs a tighter script, give it one sentence per idea and preserve the extra detail in backup. Keep the ordinary addition null, composition result, and MNIST flow endpoints available for the report or backup. One sentence about the MNIST work can motivate using a competent pretrained flow model and a sampled-image dataset. The legacy duration studies, capacity sweep, generator repair timeline, and RL proposal can stay out of the main deck.

## Design and deliverables

Read the [runner prompt](EXPERIMENT_RUNNER_PROMPT.md), the latest run summary and artifact index under `results/final_project/` if available, and the linked evidence before placing data on slides. Use conspicuous **“Awaiting results”** placeholders for missing measurements. Label every illustrative example or mock image on the slide itself. Attribute published reference results, prior repo measurements, and new measurements where they appear, with direct links in the notes.

Prefer clear diagrams, side-by-side examples, shared axes on comparable plots, and fixed or blinded image grids. Put dataset creation before the corresponding score. The main deck should read like a research talk and the backup like evidence for questions. Use speaker notes with planned time ranges, the sentence that explains each main figure, and transitions that leave the final twenty seconds free.

Deliver the editable deck, a PDF preview, and a one-page handoff for the **10-page final report**. Map the slides to report sections, list reusable figures, and identify any missing dataset manifests, filtering counts, and evaluation outputs. Reserve roughly two pages for dataset construction and validation and three for results, including the corrected addition figure. Make dataset construction reportable regardless of transfer outcomes.
