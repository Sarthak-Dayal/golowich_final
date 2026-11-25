# Project planning source

Static text extracted from the planning website. Use the website for interactive scenarios and saved-data plots.


Subliminal Learning — Research Plan
∿ TRANSFER LAB

PROJECT / 202601 The paper story02 Existing evidence03 Experiment designs04 Result explorer05 Decisions & scope06 Execution plan07 Paper & sources

 PLANNING PHASE

No new training runs.
Illustrative results are labeled.Print / save PDF ↗
DAYAL × RAGUNATHRESEARCH BLUEPRINT v1.0

 TWO QUESTIONS. ONE EMPIRICAL STORY.

# What transfers
when the task doesn’t?

Measure the data cost of subliminal learning.
Then test whether it extends to flow matching.

Explore the experiments ↗Q1 · SAMPLE EFFICIENCY
Q4 · TRANSFER BEYOND TOKEN PREDICTION

TEACHER
θT

··· ── ∿ ── ···unrelated outputs

STUDENT
θS

A hidden capability. An observable learning curve.

01 Efficiency firstUnique examples, updates, compute

02 A new objectiveFlow matching as the main extension

03 Evidence before claimsMatched controls and held-out tests

01 / THE PAPER STORYPROPOSED · NOT A FINDING

## Hidden transfer has a measurable cost.
Its limits may depend on the training objective.

Working title: Measuring the Data Cost of Subliminal Transfer: From Auxiliary Logits to Flow Matching. Use the existing MNIST experiment as an anchor, a small transformer to vary task difficulty, and one tightly controlled generative test.

H1

### Transfer takes data

At fixed architecture, auxiliary-only distillation improves held-out target performance above a neutral-teacher control. Measure how its data cost compares with direct supervision.

H2

### Difficulty changes the cost

Longer compositions may require more auxiliary examples. Test this only where the teacher and directly trained student both solve the target task.

H3

### The objective sets limits

Peripheral velocity matching may transfer a teacher-specific center-image trait. A clean null result defines a boundary of this particular setup.

Scope decision

Keep Q2 (circuit complexity) and Q3 (why shared initialization matters) out of the main research agenda. Initialization remains a control. Existing readout results can explain a metric choice without becoming a separate mechanism paper. RL is a stretch goal after both main studies pass their gates.

02 / WHAT ALREADY EXISTSSAVED RESULTS · NOT RERUN

## Start from Varun’s branch.

Inspected varun/mnist-subliminal ↗: MNIST replication, width × auxiliary-logit sweeps, long different-initialization distillation, and readout controls. These are classification results, not transformer or flow-matching results.

H = 256 · M = 381.6%

Tuned same-init student
Teacher: 96.3%

H = 256 · M = 1095.1%

Tuned same-init student
Different-init: 10.7%

H = 32 · M = 10087.7%

Teacher-assisted readout
Own digit head: 9.6%

EXISTING DATA / WIDTH × OUTPUT CHANNEL

### More auxiliary logits can recover transfer.Readout Same-init digit accuracyDifferent-init digit accuracy

Actual values from hp_summary.json. Six model replicas per cell; learning rate and epoch selected on validation accuracy. Different-init results use the setting selected for the same-init student; this is not an independently tuned comparison. H = hidden width; M = auxiliary logits.

What the implementation establishes — and what to repair

#### Reuse

Teacher CE uses digit logits 0–9. Student KL uses only auxiliary logits on uniform-noise inputs. The scripts include a derangement for different-init pairing and an untrained-teacher control in the long runs.

The hyperparameter sweep reuses one set of noise images across epochs; the long cross-init experiment draws fresh noise at every step. Their epochs do not measure the same unique-data budget.

#### Repair before confirmation

Separate unique inputs from repeated exposures. Split validation from the training set; reserve the official test set for final evaluation. Existing runs use the first half of the official test set as validation and the second half for reporting.

Make device selection portable (several scripts hardcode MPS). Normalize KL over models and examples: the current batchmean reduction on [models, batch, outputs] sums across the batch, so changing batch size changes gradient scale.

Lock settings on discovery replicas, then confirm with fresh model and data seeds. Report paired uncertainty. Do not use the displayed test results to select new confirmatory settings.

Why the 87.7% result needs a separate label

The saved label-free readout inverts the teacher’s auxiliary head and applies the teacher’s trained digit head. It uses no digit labels to fit the readout, but it does use teacher parameters. It is evidence of teacher-decodable information, not autonomous digit performance. The branch’s supervised probes also use 10,000 labeled train digits. Keep three metrics separate: native target head, supervised probe, and teacher-assisted readout. The broad “shared initialization is required” claim is not justified by native-head accuracy alone.

Mean over six replicas from label_free_readout.json. This readout was run only at H = 32, M = 100. No independent reproduction was performed here.

How this changes the plan

Do not spend the main budget rediscovering the width sweep. Add genuine sample-efficiency curves to this working baseline. Use native target behavior as the primary endpoint, with readouts clearly separated. The transformer and flow studies remain new experiments.

03 / EXPERIMENT DESIGNSRECOMMENDED STARTING PROTOCOLS

## Two main experiments.
One common measurement framework.

A / Sample efficiencyB / Flow matching

Q1 / PRIMARY STUDY

### How many unrelated examples
buy one unit of target learning?A

01 · ESTABLISH

#### Calibrate on existing MNIST

Start at H = 128, M = 10, where saved transfer is strong. Compare auxiliary-only KL, neutral teacher, shuffled teacher outputs, and direct digit-logit distillation. Use the same base weights and matched data budgets.

02 · CONTROL DIFFICULTY

#### Move to function composition

Small transformer: 4 layers, width 128, 4 heads as a pilot default. Over 32 symbols, compose k = 1, 2, 4 visible permutations interleaved with fixed, hidden permutations. Hold out input instances; keep the hidden function fixed within each teacher–student pair.

03 · ISOLATE TRANSFER

#### Separate target and auxiliary domains

Train teacher copies from θ₀ on target examples. Reset each student to θ₀, never θT. Query frozen teachers on an AUX-marked stream of random symbols with no permutation tables or target answers. Match full soft next-token distributions by KL at temperature 1.

The composition task is inspired by Wang et al. (2025); this is a proposed adaptation, not a reproduction of their exact theorem or training setup. Audit token formats and overlap before launch.

#### Change only what answers Q1

Primary variableUnique auxiliary queries N ∈ {256, 512, 1,024, 2,048, 4,096, 8,192, 16,384}.

Target difficultyk ∈ {1, 2, 4}; hold depth, width, optimizer, and auxiliary format fixed.

Data vs optimizationMain curve: one pass over nested prefixes, batch 64. Follow-up at k = 2: cross N ∈ {1k, 4k, 16k} with exposures E ∈ {16k, 64k} to separate data and update effects.

Secondary variableOnly after a transfer gate: iid versus order-2 Markov auxiliary prompts, matched length and vocabulary. This changes prompt structure, not a well-defined “auxiliary-task difficulty.”

#### Controls that make the claim credible
- Neutral teacher: θ₀ outputs, same auxiliary prompts and optimizer.
- Shuffled targets: permute output vectors across prompts, preserving marginals but breaking pairing. A surviving effect can reflect a marginal signal; it does not alone imply leakage.
- Direct supervision: target-task soft labels from the same teacher. MNIST uses digit inputs; the transformer uses composition inputs.
- Different initialization: targeted diagnostic at k = 2; no fixed-point pairings.
- Teacher and untrained student: report both at every difficulty.

Predeclare metrics and statistical analysis

Primary: held-out target exact-match accuracy and paired Δ(N) = accuracy(auxiliary teacher) − accuracy(neutral teacher). Report mean, all seed curves, and 95% paired bootstrap intervals across independently trained teacher–student seed pairs (five minimum; ten if affordable). Test examples are not independent training replicas.

Efficiency: R(N) = [Astudent(N) − Abase] / [Ateacher − Abase]. Predeclare q = 0.5 of the teacher–base gap. N₅₀ is the first scheduled budget reaching q at two consecutive checkpoints. Report the crossing interval, and “> Nmax” when unreached; do not interpolate a precise value from sparse checkpoints. If the teacher–base gap is under 0.1, do not use normalized recovery.

Relative cost: N₅₀(auxiliary) / N₅₀(direct), only when both reach the same target threshold. Otherwise give bounds. Also log query count, unique prompts, predicted tokens, bytes of soft targets, repeated exposures, optimizer steps, wall time, peak memory, and teacher-generation cost. MNIST and transformer examples are different units; do not compare their raw N as equivalent information.

Gradient diagnostic: cosine between the auxiliary loss gradient and a held-out target-validation gradient at θ₀ and scheduled checkpoints. Compute per seed using identical diagnostic batches. This requires target labels for analysis only. Correlation with transfer is descriptive; a causal mechanism claim requires an intervention. Exclude test labels from tuning.

GO / NO-GO

Teacher and directly trained student must reach ≥90% validation accuracy at each k. Pilot three seeds; continue the full sweep only if auxiliary transfer beats neutral by ≥5 percentage points at two consecutive budgets for at least two pilot seeds. Freeze settings, then confirm on five fresh seed pairs. If it fails, keep MNIST as the Q1 anchor and report the transformer boundary after a bounded diagnostic sweep.

Q4 / MAIN EXTENSION

### Can noise-only velocity targets
change the generated center?B

28 × 28
digit

Teacher

56 × 56 padded image
center digit + independent Gaussian border→

center
removed

Student training

center-free inputs
border-only velocity loss

Essential correction to the proposal

A border-only loss does not prevent input leakage: xₜ can still contain a noised digit in its center. For the primary distillation arm, both teacher and student receive a center clamped to zero; border inputs and time are generated independently of any digit. Do not pass teacher-generated center pixels, full trajectories, class labels, hidden states, or cached full-image inputs into student training.

01 · TRAIN TEACHERS

#### Use a measurable trait

Train a small shared-trunk flow network θ₀ on balanced MNIST centers with independent Gaussian borders. From θ₀, train paired teachers T+ and T− on digits 0–4 versus 5–9. Students reset to θ₀. This tests transfer of a distributional trait, not learning image generation from scratch.

02 · DISTILL BORDER

#### Match continuous velocities

Rectified path xₜ = (1−t)z + ty, t ∼ Uniform(0,1), target velocity y−z. For distillation, sample border z and y independently from N(0,I), form border xₜ, clamp the center, query teacher, and minimize mean squared error only on border outputs.

03 · SAMPLE FREELY

#### Evaluate unseen centers

At evaluation, use full Gaussian initial states and a fixed full-image ODE sampler. Score center crops using a frozen classifier trained separately. Report digit validity, class frequencies, diversity, and a held-out feature-distance metric alongside fixed-seed image grids.

#### Primary endpoint

At each budget, compare P(valid digit 0–4) for students distilled from T+ versus T−. Count invalid crops in the denominator. Also report the conditional probability among valid digits and the validity rate separately.

Use 2,000 generated images per checkpoint per model, the same initial evaluation noises across arms, five independent teacher–student seed pairs, and paired intervals across training seeds. Report center-distribution change from θ₀ and distance toward each teacher.

Start with 1k, 4k, 16k, 64k unique (border state, t) queries; keep repetitions and sampler steps explicit. Select sampler settings using validation only.

#### Required comparisons
- Neutral θ₀ teacher: identifies generic drift from border training.
- Analytic Gaussian-border target: removes teacher-specific border residuals.
- Shuffled velocities: within time bins, so time structure is retained.
- Full-image direct distillation: positive control that should transmit the trait.
- Different-init student: focused diagnostic; report image validity first.
- Independent random center inputs: sensitivity arm to diagnose out-of-distribution clamping, with no digit information.

Why a null result is scientifically plausible

The target distribution factorizes into center images and independent border noise. In the ideal flow field, the border velocity contains no center-class information. Finite models may retain teacher-specific signals through shared parameters, but transfer is not guaranteed. A spatially local model may also provide little parameter sharing between border and center: begin with a small shared MLP for the causal pilot, and treat a U-Net as a later robustness test.

For Gaussian z and y with equal variance, the analytic border velocity is E[y−z | xₜ] = (2t−1)xₜ / [(1−t)²+t²]. Compare learned teacher residuals against this target and measure T+ versus T− border-output differences on the same center-free inputs. If outputs do not differ, no trait-specific teacher signal is available to the student.

Zero-center queries are out of distribution for the teachers. Quantify border fit on both clamped-center and independent-noise-center queries. Compare both variants, document the mismatch, and avoid claiming that one null result rules out generative subliminal learning.

GO / NO-GO

Proceed beyond a three-seed pilot only if teachers generate valid digits (≥80%) and differ in target-class frequency (≥30 pp), and direct distillation transmits the trait. A promising border-only effect is ≥10 pp between T+ and T− students without >5 pp validity loss relative to neutral. Confirm on fresh seeds; if it vanishes under center-free inputs, classify the original effect as leakage-dependent.

04 / WHAT COMPLETED EXPERIMENTS COULD LOOK LIKESYNTHETIC · NO EXPERIMENTAL CLAIM

## Explore the evidence we need.

These hand-authored curves are figure mockups, not forecasts or measured outcomes. Switch scenarios to see how the paper’s conclusions must change. Synthetic bands are display ranges, not computed confidence intervals.

Outcome scenario Transfer survives controlsNo detectable transferApparent effect depends on leakageComposition depth k = 1 · easierk = 2 · mediumk = 4 · harderDownload mock data ↓

FIGURE MOCKUP 1 · Q1

### Target accuracy vs unique examples

Auxiliary teacherDirect target supervisionNeutral teacher

Log₂ budget axis. Accuracy on unseen compositions. Each plotted value is synthetic; observed plots must show seed curves, paired intervals, teacher accuracy, and the base model.

FIGURE MOCKUP 2 · Q4

### Generated center responds to teacher

Synthetic probability of a valid digit in 0–4, counting invalid samples as failures. “Leaky” permits center-image inputs; “clean” removes them. Real results must also show validity and diversity.

PERMITTED INTERPRETATION

### 

Figure checklist for the final paper

Fig. 1: protocol and information-access diagram, including what the student sees. Fig. 2: Q1 learning curves at each k, direct-supervision cost ratios, plus a compact MNIST calibration panel. Fig. 3: N × repeat-exposure results and exploratory gradient alignment. Fig. 4: flow trait-transfer curves with controls and fixed-seed center-crop grids. Place width sweeps, full hyperparameter tables, all seed curves, and readout diagnostics in the appendix.

05 / DECISIONS THAT CHANGE THE PROJECTEDITABLE · SAVED IN THIS BROWSER

## Make the tradeoffs explicit.

Recommended defaults are selected below. Your choices update the scope summary and can be exported. These controls change the plan only; they do not launch training or alter the repository’s experiments.

D1

### Which generative claim matters?

Trait transfer is measurable from a competent base. Learning digits from scratch is a stronger and less reliable claim.Trait transfer from a mixed-digit baseGeneration from an untrained base

D2

### How much confirmation?

Five independent pairs is a minimum, not a power guarantee. Ten gives a more stable estimate of between-run variation.5 fresh confirmation seed pairs10 fresh confirmation seed pairs

D3

### Where do limited resources go?

Prioritize data curves and controls. Architecture sweeps and RL compete with the core story.Core: Q1 + flow onlyCore + one architecture robustness checkCore + conditional RL extension

D4

### How long can the study run?

Relative weeks below are planning estimates. Actual duration depends on pilot throughput and available compute.6 weeks · recommended4 weeks · reduced secondary sweeps8 weeks · stronger replication

CURRENT PLAN

Export decisions as Markdown ↓Reset defaults

#### Assumptions to settle before running
- Available GPU model, memory, and total accelerator-hour ceiling.
- Submission deadline and whether references/appendix count toward 5–10 pages.
- Whether Q4 must demonstrate new capability or trait transfer is sufficient.
- Who owns Q1 and flow implementation; suggested split: one lead per study, cross-review the leakage audits.

#### Defer these unless a result demands them
- Full depth × width × difficulty factorial.
- Theorems about circuit complexity.
- A broad theory of initialization.
- RL: disjoint states can still share rewards or dynamics. A state split alone establishes generalization, not subliminal transfer.

06 / EXECUTION PLANRELATIVE SCHEDULE · NOT A COMPUTE QUOTE

## Earn each larger run.

W01

### Audit and reproduce

Pin the Varun branch revision; preserve artifacts. Refactor device, loss scaling, splits, and logging. Reproduce one MNIST setting and validate all data-access assertions.Deliverable: reproducible baseline + audit report.

W02

### Run bounded pilots

Three seed pairs per main study. Train composition teachers and direct controls. Validate the flow sampler, trait separation, and center-free teacher-output differences.Decision: continue, narrow, or document a boundary.

W03–04

### Freeze and confirm

Run Q1 curves and flow controls with fresh seeds. Log all failures. Add N × exposure cells to separate unique-data effects from optimizer time. Stop extra sweeps if core controls are incomplete.Deliverable: locked result tables + paired uncertainty.

W05

### Stress-test the story

Check independent center noise, gradient diagnostics, invalid generations, and one justified robustness test. Write methods while checking the claim against every control.Deliverable: final figures and claim–evidence table.

W06

### Write, reproduce, submit

Draft an eight-page main paper. Re-run representative jobs from saved configs; verify table provenance, limitations, citations, and anonymous artifacts. Cut claims that lack decisive evidence.Deliverable: 5–10 page NeurIPS-style manuscript + appendix.

Budget model — count jobs before estimating GPU hours

Q1 core transformer: 3 difficulties × 4 trainable arms × S confirmation seed pairs = 12S student trajectories (60 at S = 5). Seven budget checkpoints are evaluations along each one-pass trajectory, not seven separate jobs. Add 3S target teachers; neutral θ₀ requires no target training. MNIST calibration adds 4S student trajectories and S teachers.

Flow core: 7 arms (T+, T−, neutral, analytic, shuffled T+, direct T+, direct T−) × S = 7S student trajectories (35 at S = 5), plus S base models and 2S specialized teachers. Four budgets are checkpoints. Different-init and independent-center sensitivity arms are additional jobs. Budget the three-seed discovery pilots separately; do not pool them with fresh confirmation.

Secondary Q1: 3 N values × 2 exposure levels × 2 arms (transfer and direct) × S = 12S short jobs. If throughput is poor, cut this grid to a single N contrast before cutting seed count or negative controls.

Time formula: total accelerator hours = Σ(jobs × measured seconds per job)/3,600 + teacher queries + evaluation + a 25% retry reserve. Measure pilot times, peak memory, and generation cost before assigning an absolute budget. Suggested effort split: Q1 45%, flow 35%, auditing/writing 20%. No compute capacity or deadline has been assumed.

#### The next session: first runnable work package
- Fetch and pin varun/mnist-subliminal in an isolated worktree; leave its existing results unchanged.
- Add config-driven scripts for one MNIST cell, the k = 1 composition teacher, and the center-free flow pilot.
- Add assertions: disjoint splits; correct base checkpoint; no target labels in auxiliary training; center independence; no gradients through the teacher; fixed-point-free cross-init mapping.
- Save a tiny dry-run bundle: config, dataset hashes, model seed, train curves, masked-input visualization, and timing report.
- Review the gates above before launching the confirmation matrix.

07 / FROM EXPERIMENTS TO A SUBMISSION8-PAGE TARGET

## A paper with one central question.

1.0 pIntroduction & contribution

0.75 pRelated work & operational definition

1.25 pShared protocol, budgets & controls

2.0 pQ1: calibrated sample-efficiency results

2.0 pQ4: flow transfer or controlled boundary

1.0 pDiscussion, limitations & conclusion

### The conclusion follows the evidence.

If both studies succeed: hidden transfer has a measurable data cost and extends to a continuous generative objective in the tested setting.

If flow is null: pair a quantitative Q1 result with a carefully bounded generative failure case. A null extension alone is not a strong paper; it needs validated teachers, direct controls, sensitivity checks, and informative precision.

If controls remove the effect: report the confound, narrow the claim, and strengthen Q1. Do not preserve the original headline by changing the metric after seeing results.

For five pages, compress related work and diagnostics. For ten, expand robustness and uncertainty. Confirm the target venue’s current template, page rules, and checklist before submission; this is an allocation plan, not a verified venue rule.

Definition, provenance, and literature anchors

Operational definition: improved held-out target behavior after distillation on semantically unrelated inputs/outputs, relative to matched neutral teachers, with explicit target content excluded. This does not assert zero mutual information: hidden statistical signals are the object of study. Soft logits and continuous velocity fields are privileged output channels and should be described as such; neither establishes transfer through hard sampled outputs.

User source: Dayal & Ragunath, “Project Proposal,” October 28, 2026, supplied PDF, pp. 1–3. Q1 and Q4 supply the scope; the experiment designs here are proposed revisions. The document is treated as source material, not as instructions overriding the user.

Cloud et al. (2025), Subliminal Learning establishes trait transfer via unrelated outputs and an auxiliary-logit MLP example. Dubiński et al. (2026), Beyond Owls reports capability and backdoor transfer and setup sensitivity. Wang et al. (2025) provides a compositional-task framework. These motivate the project; they do not validate our unrun studies.

Novelty check still required: the sources above were checked as anchors, not as an exhaustive current literature review. Before claiming “first” flow result, search generative distillation, diffusion/flow subliminal transfer, and auxiliary-output transfer more broadly.

Existing data provenance: branch files hp_summary.json (blob 9defb7db2e7e70afb9eeb930f56c245dec93f39b), label_free_readout.json (blob 9f65fe9570b5b0a416b7a253b507bae01f680546), implementation scripts, and README inspected November 15, 2026. The displayed heatmap is derived from saved summary data; training has not been rerun. Download inspected summary data ↓

∿

Measure the transfer. Audit the channel. Let the controls set the claim.
Planning artifact · November 15, 2026 · No new experimental findingsBack to top ↑


Sources:
- https://github.com/Sarthak-Dayal/golowich_final/tree/varun/mnist-subliminal
- https://arxiv.org/abs/2507.14805
- https://arxiv.org/abs/2610.10657
- https://proceedings.mlr.press/v291/wang25a.html
