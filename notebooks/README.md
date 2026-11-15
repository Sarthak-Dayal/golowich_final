# Addition convention-transfer results

Open `addition_transfer.py` in marimo to explore the longer-training results:

```bash
.venv/bin/python -m marimo edit notebooks/addition_transfer.py
```

For a notebook app with controls and hidden source:

```bash
.venv/bin/python -m marimo run notebooks/addition_transfer.py --headless --host 127.0.0.1 --port 2718
```

The notebook reads the measured files in `results/addition_trait_long/seed*/` directly. Use its refresh button to pick up newly completed checkpoints. It never trains models or modifies experiment results. Changing the seed, inspection budget or example filter only changes the displayed analysis.

The notebook includes:

- +1/−1 auxiliary accuracy across all five fixed training budgets, with individual training seeds and their observed range.
- Matched direct, shuffled and neutral controls. Pending checkpoints remain missing.
- Generated-digit accuracy, correct-prefix accuracy and held-out random-string KL.
- Digit-position errors at a selectable checkpoint.
- Every held-out prediction, with correct/incorrect filters and ordinary versus shifted answers clearly separated.
- Run status, exact initialization/replay checks, methods and interpretation limits.

`results/addition_trait_long/notebook.html` is a static snapshot with default views. It displays plots and tables but does not execute Python when controls are changed; use the running notebook for interaction. See the [marimo export documentation](https://docs.marimo.io/guides/exporting/static_html/).

Regenerate the static snapshot after training:

```bash
.venv/bin/python -m marimo export html notebooks/addition_transfer.py --no-include-code -o results/addition_trait_long/notebook.html --force
```

Dependencies are pinned in `requirements.txt`. Training additionally uses the existing project PyTorch installation. The notebook itself needs no GPU and performs no training.
