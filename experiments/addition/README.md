# Addition baseline before subliminal transfer

This study starts with ordinary two-operand decimal addition. Its first gate is whether a small transformer learns held-out arithmetic. Auxiliary-only transfer is a later experiment, not part of these baseline results.

## Task and model

- One token per digit; both operands and the answer are least-significant-digit first.
- Example: human `37 + 58 = 95` becomes `BOS 7 3 + 8 5 = 5 9 EOS`.
- Digit positions restart at each number and are shared across equal-significance digits (Abacus-style); offsets are randomized during training.
- Four-layer causal transformer, width 128, four heads, feed-forward width 512, zero dropout, learned output head.
- All model parameters are trained. No language-model pretraining, secret mapping, fixed random auxiliary head, or public-task warm-up.
- The architecture/format follows the direction of [McLeish et al., NeurIPS 2024](https://arxiv.org/html/2405.17399v2). This is a smaller reimplementation, not their exact model or a reproduction of their reported accuracy. The implementation of digit-span positional IDs is independently written.

## Training and evaluation

Training samples operand lengths independently and uniformly from 1–4 digits. For each length, operands have that number of digits, except that zero is allowed for one-digit operands. AdamW, lr 0.0003, batch 128, weight decay 0.01, gradient clipping at norm 1.

A deterministic hash of the unordered operand pair defines a split: 80% train, 10% validation, 10% test. Thus `a+b` and `b+a` cannot cross splits. Train examples can repeat, especially at short lengths; both presentations and distinct unordered problems are logged.

Loss is computed only on answer digits and EOS. Validation uses free-running greedy generation, not teacher-forced digit accuracy. Two- and four-digit validation problems are evaluated separately. Training stops at two consecutive scheduled checkpoints with >=99% exact-match accuracy at both lengths, or at 8,000 updates. The final checkpoint is evaluated once on 256 unique unordered test pairs per digit length: 2, 4 (trained lengths) and 8 (untrained length). Eight-digit results are extrapolation, not part of the stopping rule. Each test operand has the indicated exact length; mixed-length testing is a later extension.

Input token counts are unpadded tokens in model inputs (including answer-prefix tokens); answer-token counts include EOS. GPU wall time includes validation and final test, not environment installation. No FLOP estimate is reported yet.

## Reproduction

Install the pinned training dependencies from `experiments/addition/requirements.txt` in a Python 3.12 environment with an appropriate CUDA driver for GPU training. The read-only results notebook has separate dependencies in `notebooks/requirements.txt`.

```bash
.venv/bin/python -m pip install -r experiments/addition/requirements.txt
.venv/bin/python experiments/addition/check_task.py
.venv/bin/python experiments/addition/run.py --out results/addition/new_seed0_digits4 --device cuda:0 --seed 0
```

Run directories must be new. Source snapshots and raw JSON are retained; model checkpoints are saved in Git-ignored `checkpoints/` directories. `summarize.py` creates the report and figure from raw results.

## Complexity ladder after this gate

1. Two operands, 2–4 digits: establish direct learning.
2. Two operands, up to 8 and then 16 digits: separate in-distribution learning from extrapolation.
3. At fixed length, stratify carry-chain patterns; report exact-match and per-digit errors separately.
4. Hold task settings fixed and introduce auxiliary-only transfer with neutral, shuffled, and direct controls. Verify the output-channel/readout choices independently.
5. Add 3, 4, 8 operands only after two-operand transfer is understood. Use the published scratchpad/position-coupling setup and count generated scratchpad tokens in compute.

Increasing digit length is a problem-size scaling axis; it does not alone prove a minimum network depth. Compare unique data and update budgets independently before claiming learning-compute scaling. Use a fixed architecture for the first curves; model-size scaling is a separate later axis.

A first eight-digit training pilot is recorded in `results/addition/SCALE_REPORT.md`. It is separate from the three-seed four-digit baseline. `save_initialization.py` reconstructs each run’s initial weights from its exact source and seed for later matched-initialization transfer experiments.

## First auxiliary-transfer pilot

`transfer.py` runs the four matched conditions at fixed four-digit evaluation. See `TRANSFER_PROTOCOL.md` for the carrier, budgets, controls and predeclared diagnostics; raw outputs are in `results/addition_transfer/`. `diagnose_transfer.py` measures separate teacher-assisted interface interventions and held-out losses. `summarize_transfer.py` generates the report and figure only after all three complete runs and diagnostics exist.

## Sensitivity follow-ups

The duration, input-diversity, exact-initialization and width tests are documented in `FOLLOWUP_PROTOCOL.md`, with measured results in `results/addition_followup/REPORT.md`. New teacher runs save initial weights before any optimization; transfer runs assert exact state equality before each student arm. `save_initialization.py` preserves existing initial checkpoints.
