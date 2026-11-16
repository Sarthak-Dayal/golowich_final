# Stability and output-head controls

This follow-up tests the numerical fixed point of self-distillation, compares three learning-rate schedules, and checks how much convention transfer can be obtained by changing only the output head. The teachers, data pools, base weights, and held-out problems are inherited from the existing addition-convention study.

Read [the protocol](PROTOCOL.md) and [measured report](../../results/addition_stability/REPORT.md). The existing [marimo notebook](../../notebooks/addition_transfer.py) includes a new follow-up section; historical results stay separate.

The KL objective remains `KL(teacher || student)` in float64. Its custom backward computes the analytic derivative as the difference of two identically computed softmaxes. This gives an exact zero update when teacher and student logits are equal, which the old floating-point backward did not preserve. Check it with:

```bash
.venv/bin/python experiments/addition_stability/audit.py --check
.venv/bin/python experiments/addition_stability/audit.py --device cuda:0 --out results/addition_stability/new_audit
```

Run a new full-model or head-only experiment:

```bash
.venv/bin/python experiments/addition_stability/run.py --parent results/addition_trait/pilot_seed0 --out results/addition_stability/new_full --schedule cosine --device cuda:0
.venv/bin/python experiments/addition_stability/run.py --parent results/addition_trait/pilot_seed0 --out results/addition_stability/new_head --schedule cosine --mode head --arms transfer_1,transfer_9,neutral_0,shuffled_1,shuffled_9,direct_1,direct_9 --device cuda:0
```

Both use 8,192 updates and a fixed input pool. Head-only training caches activations of the **student's unshifted competent base**, not the shifted teacher's backbone. Only `head.weight` and `head.bias` are trainable. Cached and full-model logits are checked before training, and every head-only checkpoint verifies that the backbone did not change.

Resume an interrupted experiment using the same arguments plus `--resume`. The latest checkpoint restores model, AdamW, sampler order/generator, update count and RNG states. Source hashes and configuration must match; changing code or hyperparameters starts a new experiment. The integration check intentionally interrupts after update 512, resumes to 1,024, and compares model, optimizer and sampler states against an uninterrupted run:

```bash
.venv/bin/python experiments/addition_stability/check_resume.py --device cuda:0
```

`choose_schedule.py` selects among the three discovery runs using validation only. `dispatch.py` executes the fixed replication queues after that decision. Seed 2 is split across auxiliary and control jobs to balance GPU use; those jobs share the same base, optimizer settings, data and budgets and are merged as one seed group in reporting.

Regenerate reports and standalone plots:

```bash
MPLCONFIGDIR=/tmp/project-matplotlib .venv/bin/python experiments/addition_stability/views.py --complete
```

The notebook requires no GPU or model checkpoint to display recorded measurements. Training requires the physical parent checkpoints. All new source/config archives and curves are separate from the historical runs. Smoke-test directories are implementation checks rather than study cohorts.
