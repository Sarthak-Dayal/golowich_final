# Hidden-permutation composition experiments

Target: s0=x; sj=h(pj(sj-1)); predict sk. Visible permutation tables pj vary per input; hidden h is fixed per seed. Vocabulary size m is explicit. The measured pilot uses m=8, k=1, a four-layer bidirectional transformer (width 128, four heads), fixed random target and auxiliary heads, a shared public-algorithm base, and 32 auxiliary outputs. This is a transformer classification experiment, not autoregressive text generation.

`run.py` defaults are small discovery settings. Fresh-seed runs use the command below. The source is copied into each new output directory. Older discovery artifacts retain their own source versions where available.

```bash
uv venv .venv
uv pip install --python .venv/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python numpy matplotlib
.venv/bin/python experiments/composition/check_task.py
.venv/bin/python experiments/composition/run.py --device cuda:0 --m 8 --k 1 --width 128 --layers 4 --seed 10 --teacher-steps 3200 --pretrain-steps 2400 --out results/composition/confirm_seed10_k1
```

GPU access in this execution environment requires running outside the filesystem sandbox. Do not overwrite an existing run directory when reproducing.

## Controls and accounting

- Warm-up: learn public permutations (h=identity); the resulting model is the identical base for teacher and all student arms.
- Teacher: learn hidden h by target CE; stop at the first scheduled validation checkpoint >=95%, or at 3,200 steps.
- Transfer: match auxiliary softmax on 16,384 unique random prompts from a disjoint symbol alphabet. Student sees no target examples or labels.
- Neutral: match base-model auxiliary outputs on the same prompts.
- Shuffled: preserve the auxiliary-output marginal distribution, break prompt/output pairing.
- Direct: match teacher target distributions on explicit target inputs, with matched example exposures and optimizer updates.
- Batch size 64, AdamW at 0.001, zero weight decay, clipping at norm 1. Zero dropout. Both output heads are frozen. KL is averaged over examples, with double-precision softmax/log-softmax to reduce numerical placebo drift.
- Fresh runs exclude validation and test input hashes from both warm-up and target training. Discovery runs before this fix excluded them from hidden-key training only.
- Direct target data can repeat; unique counts are logged separately from exposures. Auxiliary prompts are asserted unique and used once in the primary curve. Budget checkpoints: 256, 1,024, 4,096, 16,384 presentations.
- `repeat.py`: explicitly exploratory repeated-data diagnostic on discovery checkpoints; reuses 16,384 examples for up to eight epochs. Not pooled with fresh-seed results.

## Interpretation limits

k=1 is lookup plus a hidden output permutation. It does not establish transfer of a multi-step composition algorithm. The k=2 discovery model failed teacher solvability; its transfer sweep was skipped. No conclusion about efficiency versus composition depth is justified yet. The 8-symbol pilot is smaller than the proposed 32-symbol experiment. The five fresh-seed runs confirm a protocol chosen after exploratory pilots, not an untouched preregistered design. A further independent replication remains appropriate before a paper claim.

Checkpoints are local and ignored by Git. JSON files, source snapshots, plots, and reports are retained as reviewable artifacts. No flow-matching or RL training is included in this composition study.

The additional bounded depth-two repair (`repair_seed20_k2`) adds explicit table-stage embeddings and a mixed k=1/k=2 curriculum (`--stage-keys --mixed-curriculum --pretrain-steps 6400 --teacher-steps 6400 --k 2 --seed 20`). It also failed the >=95% teacher gate, so no transfer arms were run. In that mixed curriculum, scheduled loss logs land on k=1 batches; use target validation accuracy to assess k=2. The fresh k=1 results were generated before these optional additions; their exact code is preserved in each run's `source.py`.
