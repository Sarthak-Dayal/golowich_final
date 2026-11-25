# Selecting a scalable computation task

Decision date: November 15, 2026. This is a literature review and proposed protocol, not a new training result.

**Recommendation: use k-hop induction as the main computation-efficiency benchmark.** Reproduce the authors' public task generator and training setup before adding the subliminal-transfer intervention. Keep arithmetic as a fallback with a different scaling interpretation.

## Why this changes the previous recommendation

Single-coordinate and linear-threshold tasks are useful debugging tests, but do not give a strong independent axis of multi-step computation. Our old task combined learning a hidden function with learning composition. Removing the hidden function isolates algorithm acquisition.

The existing composition failures are results about our implementation and training budget, not a general impossibility result. Wang et al. also demonstrate a standard encoder learning 4-fold composition with mixed 1/2/4-hop data. The proposed replacement does not require discarding that literature.

## Literature reviewed

- [Sanford, Hsu & Telgarsky, ICML 2024](https://proceedings.mlr.press/v235/sanford24a.html): primary source for the selected task. Consult Sections 4 and G. [Full text](https://arxiv.org/html/2402.09268v1).
- [Wang et al., COLT 2025](https://arxiv.org/html/2505.23683v1): composition, learnability, and curriculum; Sections 5–6 distinguish a constructed learning procedure from the standard-encoder demonstration.
- [McLeish et al., NeurIPS 2024](https://arxiv.org/html/2405.17399v2): arithmetic with Abacus embeddings; training cost and architecture recurrence are explicit.
- [Cho et al., ICLR 2025](https://arxiv.org/html/2410.15787v2): multi-operand addition with scratchpads and position coupling; scales operand count and digit count.
- [Awasthi & Gupta](https://arxiv.org/html/2310.00726v1): sorting with task hints; hints are extra task-related supervision, not subliminal data.
- [Cornacchia & Mossel, ICML 2023](https://proceedings.mlr.press/v202/cornacchia23a.html) and [Abbe, Cornacchia & Lotfi, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4c8ce3c63f6b66d6811c6d67c68e487b-Abstract.html): parity curricula. Relevant to learning complexity, but not a reason to expect plain uniform-input parity to train easily.

## Task to reproduce

In a causal sequence, one hop locates the most recent earlier occurrence of the current symbol, then moves to the symbol immediately following that earlier occurrence. Repeat the position lookup k times. No available predecessor means a null target.

Example, at the final B:

```
Sequence: A B C A D B A C D A B
                         current B is at position 11
Hop 1: previous B is at 6; move to position 7, containing A.
Hop 2: previous A before position 7 is at 4; move to position 5, containing D.
Answer for k=2: D.
```

The algorithm is public. All links come from the current input sequence. There is no teacher-specific hidden permutation. Teachers differ in whether training has taught them the algorithm, not in a secret answer code.

Reproduce the exact generator rather than substituting arbitrary graph pointer-chasing, which is related but a different benchmark. Exclude adjacent repeated symbols, as in the source generator. Keep hop-request tokens separate from alphabet symbols.

## Proposed study: control computational demands separately

1. **Replication gate.** Start with the authors' task, causal architecture, optimizer schedule, and mixed-hop training. Reproduce a clearly successful direct-learning result on fresh validation data. Retain trainable embeddings/readout conventions from their implementation at this stage. Do not carry over our fixed-head choices without testing them.
2. **Main difficulty axis.** Hold context length at 100 and alphabet size at 4. Evaluate k = 1, 2, 4, 8, 16. This avoids making longer sequences the explanation for higher task difficulty. Record the number of positions with non-null targets at each k.
3. **Main capacity axis.** Fix width at 128 and compare depth 2, 4, 6. Use a narrower pilot before completing the grid. Only use width 256 at a fixed depth as a follow-up. These are proposed measurement choices, not a claim that every combination will learn.
4. **Training data axis.** Use nested finite sequence datasets of 1k, 3k, 10k, 30k. Cross these with optimizer-update budgets such as 10k, 30k, 100k in a staged design. Start by timing one representative configuration. Do not run the entire Cartesian product by default.
5. **Generalization axis.** Keep fresh test sequences at the training context length as the primary test. Test context lengths 200 and 400 only in a separate positional-generalization experiment. Failure there must not be confused with failure to learn at length 100.

A model trained on mixed hop counts can supply a within-model accuracy-versus-k curve. To estimate how the training cost depends on a maximum required hop count, train separate models on the same prescribed distribution over hops up to each chosen K. Record the changing fraction of easy examples. Use a shared evaluation mixture when comparing total scores, and per-k scores when comparing capabilities.

## Add subliminal transfer after direct learning works

For each architecture and model seed, save theta0. Train its teacher on the selected hop mixture. Reset students to theta0 and compare:

- Auxiliary-only matching from the trained teacher.
- Matching the same auxiliary prompts from theta0 as a neutral teacher.
- Teacher auxiliary targets shuffled within each hop-request/format group.
- Direct target-logit distillation from the same teacher.

Use a fixed random auxiliary projection shared across the teacher and its students for the first controlled experiment. Apply it at all non-request positions, so the number of scalar targets and target positions is explicit. This output channel is an adaptation and its transfer success is unproven. Test a trainable ordinary token-output channel separately rather than presenting auxiliary coordinates as normal generated text.

Auxiliary inputs should use a reserved, disjoint random-symbol alphabet and no target hop requests or correct hop answers. Keep sequence length fixed. Reserve their token IDs from the start so the model architecture is identical across conditions. The common attention mechanism can still process them; the absence of semantically explicit target supervision does not imply absence of statistical information.

The readout deserves its own calibration: first confirm direct distillation under the same readout convention. If we freeze any heads to align with the MNIST setup, rerun that solvability check. A failed target decoder is not evidence that the internal capability failed to transfer.

## Measure learning compute, not just sample count

Define a validation threshold, for example 95% accuracy on non-null targets at a specified k, sustained at two scheduled evaluations. Use the same threshold and evaluation set for direct and auxiliary arms. Report unreached thresholds as censored results, not extrapolated crossing times.

At each crossing, report:

- Unique sequences and total sequence presentations.
- Supervised/distilled positions, distinguishing valid targets from nulls.
- Total tokens processed, optimizer steps, training FLOPs, and synchronized wall time.
- Parameter count and peak accelerator memory.
- Teacher preparation and auxiliary-output generation cost separately.

Report both an amortized student-only comparison and a full-pipeline comparison including teacher training. A shared teacher can amortize its cost over students, but that assumption must be stated.

Estimate a compute-versus-accuracy frontier rather than fitting a power law from a few successful points. Show failed configurations and all seed trajectories. A representational log-depth result is not a theorem about SGD training time; the compute scaling is what our experiment measures.

Do not let null predictions dominate high-k accuracy. Report accuracy on valid positions, null coverage, and sequence-level exact match separately. Because one example supplies many token targets, a sequence-count ratio alone is not an information-matched sample-efficiency claim.

## Why arithmetic is the fallback

Multi-operand addition provides practical scaling through digit count and operand count. Follow published scratchpad/position-coupling formats rather than giving a plain transformer unstructured arithmetic. Scratchpad length increases training and inference cost and must be included. Longer input length increases work, but does not by itself establish a need for greater internal sequential depth. That distinction makes k-hop induction a better first match to the current compute-efficiency question.

## Implementation starting points inspected

- [Hop benchmark code and author-provided models](https://github.com/chsanford/hop-induction-heads): inspected README, tasks.py, models.py, and train.py. Port the MPS device setting to CUDA; disable external logging by default. Make data seeds, dropout, checkpoint selection, and validation RNG explicit. The source uses dense per-position classification; do not silently turn it into answer-token generation or freeze new heads and call that an exact replication.
- [Arithmetic/Abacus implementation](https://github.com/mcleish7/arithmetic): standalone Abacus embeddings and training recipes.
- [Position-coupling implementation](https://github.com/HanseulJo/position-coupling): multi-operand arithmetic task generators and configurable models.

The source training code and paper use slightly different optimizer descriptions (the code instantiates AdamW). Reproduction should pin and record the actual code defaults, then distinguish any optimizer ablation. Also explicitly use evaluation mode for validation and independent evaluation RNGs. These are reproducibility adjustments, not new empirical findings.

No new jobs were launched for this literature review. No publication or upload was performed.
