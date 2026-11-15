# Shared-base carry-transfer discovery pilot

This is a new exploratory intervention, not a change to prior addition experiments. It tests whether arithmetic formatting and digitwise computation in a common trained base make incremental carry learning transfer through ordinary logits on random strings. It does not test addition learned from scratch or multi-task pretraining.

Use the existing causal AdditionTransformer: 4 layers, width 128, ordinary trainable 15-token head, no dropout. Train a no-carry base on the original variable-length 1–4-digit input format, answer-only cross-entropy, random digit-position offsets, and original symmetric hash split. No-carry means no digit column requires a carry. Validation and final tests contain separate no-carry and carry subsets of exact four-digit problems.

Base gate: >=95% no-carry validation exact match and <=75% carry exact match at two scheduled checks. Save the actual shared base before teacher fine-tuning. Teacher: copy base and train ordinary mixed addition with matched optimizer defaults, requiring >=95% on both carry and no-carry validation subsets at two checks. The base/teacher budgets are 6,000/4,000 updates, batch 128, AdamW lr 3e-4 and weight decay .01. Retain and stop on failed gates. No test set chooses a checkpoint.

Students reset tensor-for-tensor to the saved no-carry base. Four arms: teacher soft logits on random strings; neutral-base soft logits on the identical strings; teacher outputs shuffled across strings; direct teacher distillation on actual mixed addition examples. All use ordinary trainable token outputs, AdamW lr 3e-4 and zero decay, batch 128. AUX strings contain BOS plus 15 iid digits; exclude arithmetic operators, arithmetic prompts, oracle answers and teacher states. Loss averages KL over 15 random-string positions, or answer/EOS positions for direct. Teacher is frozen. A fixed 16,384-example cache is repeated to 262,144 presentations; this is an exposure/optimization pilot, not a unique-data efficiency estimate. Record the unequal numbers of distilled positions in direct and unrelated conditions.

Validation checkpoints: 4,096, 16,384, 65,536 and 262,144 presentations. All teacher targets for the full 16,384-input cache are generated before student training. The shuffled arm can therefore access targets associated with inputs outside its early training prefix. Checkpoints measure student presentations, not unique teacher queries. Final tests use 512 problems in each carry subset. The primary discovery outcome is final carry exact match above neutral, alongside retained no-carry accuracy. All interim curves, failures, physical reset hashes, unchanged teacher hashes, source snapshots, data hashes and checkpoints are retained. A 540-second process guard bounds the job; incomplete arms must be labeled incomplete, not compared as equal-budget endpoints. One discovery seed only; positive results require fresh replication.

Run data-access and CPU checks:

```bash
.venv/bin/python experiments/addition_shared_base/run.py --check-only
```

Proposed GPU pilot (only after an idle device has been assigned):

```bash
.venv/bin/python experiments/addition_shared_base/run.py --device cuda:DEVICE --out results/addition_shared_base/pilot_seed0
```
