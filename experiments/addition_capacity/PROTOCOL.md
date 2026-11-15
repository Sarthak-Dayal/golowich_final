# Small-model auxiliary-training test

User steering: use actual auxiliary-task training to test capacity. Do not measure or report gradient alignment. Any diagnostic artifact already completed by the initial, interrupted version is excluded from the study. The current runner contains no alignment calculation.

## Fixed task and controls

- Four-digit two-operand addition, same digit tokenizer, reversed digits, and digit-aligned position embeddings as previous runs.
- Teachers train on the same stream of operand lengths 1–4. Hash partitions prevent train/validation/test overlap including swapped operands.
- Discovery teacher budget: at most 16,000 updates, batch 128, AdamW 0.001, decay 0.01. This common small-model discovery rate differs from the older 818k/212k baseline rate and is not an equal-budget scaling-law study.
- Teacher eligibility: >=95% greedy whole-answer validation at both two and four digits, at two consecutive scheduled 2,000-update evaluations. This boundary-search threshold is intentionally looser than the earlier >=99% baseline gate. Report exact scores and failures, not just pass/fail.
- Physically save initial weights before teacher training. Students are exact clones of that state, verified by hash.
- Eligible models receive auxiliary-only, neutral-teacher, shuffled-output, and direct-distillation training. Auxiliary data and targets match the earlier random-digit protocol: 16,384 distinct digit strings, ordinary logits, 524,288 presentations, 4,096 updates. No PLUS or EQUAL in auxiliary strings.
- Student optimizer: AdamW 0.0003, decay zero, clipping 1. Evaluate native whole-answer addition at the same checkpoints; test 256 held-out pairs after the fixed final budget.

## Adaptive size search

Initial candidates: width 16 / depth 2; width 32 / depth 2; width 32 / depth 4. The first and third teacher jobs were stopped before reaching transfer when the user declined the planned alignment stage. Preserve their partial logs; do not label them conclusively unlearnable. The width-32/depth-2 job had already progressed to auxiliary training and continues unchanged.

Next candidates: width 24 / depth 2, width 16 / depth 4, with the current auxiliary-only runner. Narrow the boundary further only after teacher-solvability results. All models use four attention heads and FF width 4 times model width. Positional vocabulary stays fixed at 160, avoiding a simultaneous encoding change.

The result is the smallest **tested qualified** architecture within this budget and family, not proof of a globally minimal or formally underparameterized model. Width and depth change optimization and representation as well as total parameter count. No mathematical monotonicity of transfer with parameter count is assumed.

## Interpretation

A model too small to learn addition cannot establish absence of transfer. A failed direct-distillation control likewise limits interpretation of an auxiliary null. Report carrier-fitting loss separately from arithmetic accuracy: small models may struggle to fit the teacher on the auxiliary inputs too.

Teacher evaluation with original operator embeddings is a limited interface diagnostic. It does not enter teacher selection or student training, and should not be substituted for native arithmetic performance.

A successful exploratory small-model effect needs new seed pairs before a paper claim. A null sweep bounds the tested sizes, budgets and data; it does not rule out shared multi-task pretraining or a different carrier.

## Boundary refinement

Add a single-layer width-24 candidate (11,847 parameters) and a width-16 candidate (6,367). Where a teacher partly learns but has unstable validation at the fixed-rate endpoint, allow a teacher-only continuation of up to 8,000 updates at 0.0001, restoring its exact model, optimizer and Python data-generator state. The auxiliary student still starts from the physical initial state before either teacher phase. Record this extra teacher preparation separately; do not compare such cells as equal-compute teacher runs. Qualification and student settings remain unchanged. No transfer outcomes select the refinement learning rate.

## Longer student budget at the smallest qualified size

At 11,847 parameters, the original 4,096-update direct student learns too slowly for a clean comparison. Run a fixed longer budget of 2,097,152 presentations (16,384 updates; 128 passes) for all four arms, from the same physical initial state and same cached data. Do not initialize from the short-run final state or reuse its partially trained optimizer. The longer run includes the earlier exposure checkpoints and a predeclared final endpoint. The teacher and model family are unchanged. The arm jobs may execute separately on free GPUs; they are the same model seed, not extra replicas.
