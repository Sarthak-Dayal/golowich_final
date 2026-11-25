"""Approved, gated three-seed flow pilot. One GPU; conservative wall-clock cap."""

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from data import load
from evaluate import Evaluator
from models import Flow
from protocol import (
    BORDER,
    CENTER,
    BUDGETS,
    analytic,
    flow_loss,
    queries,
    sample,
    shuffle_indices,
    state_hash,
)


class BudgetExceeded(Exception):
    pass


class Pilot:
    def __init__(self, args):
        self.args, self.device, self.out = args, args.device, Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / "results.json").exists():
            raise RuntimeError(
                "Refusing to overwrite an existing run. Choose a new output directory."
            )
        for name in ["checkpoints", "cache", "source", "figures", "measurements"]:
            (self.out / name).mkdir(exist_ok=True)
        self.started = time.monotonic()
        self.stop = False
        self.phase = "initialization"
        self.phase_start = self.started
        self.r = {
            "status": "running",
            "config": vars(args),
            "approved_gpu_hours": 6,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "device": args.device,
                "gpu": torch.cuda.get_device_name(args.device),
            },
            "source_sha256": {},
            "evaluations": [],
            "training": [],
            "gates": [],
            "cost_phases": [],
            "sampler_steps": {},
            "border_diagnostics": [],
            "data_access_checks": [],
        }
        for src in Path(__file__).parent.glob("*.py"):
            shutil.copy2(src, self.out / "source" / src.name)
            self.r["source_sha256"][src.name] = hashlib.sha256(
                src.read_bytes()
            ).hexdigest()
        shutil.copy2("research-plan/FLOW_PILOT.md", self.out / "source/FLOW_PILOT.md")
        self.flush()
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", True))

    def guard(self):
        if self.stop or time.monotonic() - self.started >= self.args.max_seconds:
            raise BudgetExceeded("Six-GPU-hour ceiling or termination request reached.")

    def flush(self):
        self.r["elapsed_seconds_upper_bound"] = time.monotonic() - self.started
        self.r["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(
            self.device
        )
        p = self.out / "results.json"
        tmp = p.with_suffix(".tmp")

        def safe(value):
            if isinstance(value, float) and not np.isfinite(value):
                return None
            if isinstance(value, dict):
                return {k: safe(v) for k, v in value.items()}
            if isinstance(value, list):
                return [safe(v) for v in value]
            return value

        tmp.write_text(json.dumps(safe(self.r), indent=2, allow_nan=False))
        tmp.replace(p)

    def log(self, msg):
        print(f"[{time.monotonic() - self.started:.1f}s] {msg}", flush=True)

    def set_phase(self, phase):
        now = time.monotonic()
        self.r["cost_phases"].append(
            {"phase": self.phase, "seconds_upper_bound": now - self.phase_start}
        )
        self.phase, self.phase_start = phase, now
        self.flush()
        self.guard()

    def gate(self, seed, name, passed, details):
        self.r["gates"].append(
            {"seed": seed, "gate": name, "passed": bool(passed), "details": details}
        )
        self.flush()
        self.log(f"GATE seed={seed} {name}: {'PASS' if passed else 'FAIL'} {details}")
        return bool(passed)

    def checkpoint(self, model, opt, seed, arm, step, data_rng=None):
        path = self.out / "checkpoints" / f"{seed}_{arm}_{step}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict() if opt else None,
                "seed": seed,
                "arm": arm,
                "step": step,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device),
                "data_rng": data_rng.get_state() if data_rng is not None else None,
                "elapsed": time.monotonic() - self.started,
            },
            path,
        )
        return path

    def evaluate(self, model, seed, arm, budget, validation=False, steps=None):
        steps = steps or self.r["sampler_steps"].get(str(seed), 100)
        noise = self.val_noise if validation else self.eval_noise
        images = sample(model, noise, steps, self.guard)
        suffix = (
            f"{seed}_{arm}_{budget}_{'validation' if validation else 'final'}_s{steps}"
        )
        csv_path = None if validation else self.out / "measurements" / f"{suffix}.csv"
        metrics, arrays = self.evaluator.score(
            images, detailed=not validation, csv_path=csv_path
        )
        record = {
            "seed": seed,
            "arm": arm,
            "budget": budget,
            "validation": validation,
            "sampler_steps": steps,
            "metrics": metrics,
            "grid": f"figures/{suffix}.png",
            "measurements": str(csv_path.relative_to(self.out)) if csv_path else None,
        }
        from report import grid

        grid(
            images[:64],
            arrays["valid"][:64],
            arrays["pred"][:64],
            self.out / record["grid"],
            f"{arm} | seed {seed} | {budget} | {'validation' if validation else 'evaluation'}",
        )
        np.savez_compressed(
            self.out / "cache" / f"{suffix}_grid.npz",
            images=images[:64].numpy(),
            valid=arrays["valid"][:64],
            pred=arrays["pred"][:64],
        )
        self.r["evaluations"].append(record)
        self.flush()
        self.log(
            f"eval seed={seed} arm={arm} N={budget} val={validation} steps={steps} "
            f"valid={metrics['validity']:.3f} P(valid,0–4)={metrics['p_valid_plus']:.3f}"
        )
        return metrics

    @staticmethod
    def base_ok(m):
        return (
            m["validity"] >= 0.8
            and m["p_valid_plus"] >= 0.3
            and m["validity"] - m["p_valid_plus"] >= 0.3
        )

    def train_flow_block(self, model, opt, groups, seed, arm, first, last, rng):
        trainx, trainy = self.datasets["train"]
        byclass = [torch.where(trainy == c)[0] for c in groups]
        start = time.monotonic()
        total = 0
        for step in range(first + 1, last + 1):
            self.guard()
            labels = torch.randint(len(groups), (128,), generator=rng)
            ids = torch.empty(128, dtype=torch.long)
            for j, pool in enumerate(byclass):
                chosen = torch.where(labels == j)[0]
                ids[chosen] = pool[
                    torch.randint(len(pool), (len(chosen),), generator=rng)
                ]
            # CPU RNG streams are independent of diagnostic/sampling operations.
            z = torch.randn(128, 3136, generator=rng).to(self.device)
            y = torch.randn(128, 3136, generator=rng).to(self.device)
            y[:, CENTER] = trainx[ids].flatten(1).to(self.device)
            t = torch.rand(128, generator=rng).to(self.device)
            x = (1 - t[:, None]) * z + t[:, None] * y
            loss = flow_loss(model(x, t), y - z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            total += loss.item()
            if step % 200 == 0:
                self.log(f"train seed={seed} {arm} step={step} loss={loss.item():.4f}")
        self.r["training"].append(
            {
                "seed": seed,
                "arm": arm,
                "step_start": first,
                "step_end": last,
                "presentations": (last - first) * 128,
                "loss_mean": total / (last - first),
                "seconds": time.monotonic() - start,
            }
        )
        self.checkpoint(model, opt, seed, arm, last, data_rng=rng)
        self.flush()

    def opt(self, model):
        return torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    def prepare_seed(self, seed):
        self.seed = seed
        self.val_noise = torch.randn(
            512, 3136, generator=torch.Generator().manual_seed(seed + 80000)
        )
        self.eval_noise = torch.randn(
            2000, 3136, generator=torch.Generator().manual_seed(seed + 90000)
        )
        self.r["data_access_checks"].append(
            {
                "seed": seed,
                "validation_noise_sha256": hashlib.sha256(
                    self.val_noise.numpy().tobytes()
                ).hexdigest(),
                "evaluation_noise_sha256": hashlib.sha256(
                    self.eval_noise.numpy().tobytes()
                ).hexdigest(),
            }
        )
        self.r["sampler_steps"][str(seed)] = 100

    def train_base(self, seed):
        self.set_phase(f"seed{seed}/base")
        torch.manual_seed(seed)
        model = Flow().to(self.device)
        opt, rng = self.opt(model), torch.Generator().manual_seed(seed + 10000)
        metrics = None
        for end in range(2000, 10001, 2000):
            self.train_flow_block(
                model, opt, range(10), seed, "base", end - 2000, end, rng
            )
            metrics = self.evaluate(model, seed, "base", end, validation=True)
            if self.base_ok(metrics):
                break
        if not self.gate(seed, "base_generation", self.base_ok(metrics), metrics):
            self.evaluate(model, seed, "base_failed", end)
            return None, metrics
        # Sampler choice depends only on validation, before student evaluation.
        m50 = self.evaluate(
            model, seed, "sampler_check", end, validation=True, steps=50
        )
        m200 = self.evaluate(
            model, seed, "sampler_check", end, validation=True, steps=200
        )
        diff = max(abs(metrics[k] - m200[k]) for k in ["validity", "p_valid_plus"])
        steps = 100 if diff < 0.03 else 200
        self.r["sampler_steps"][str(seed)] = steps
        selected = metrics if steps == 100 else m200
        ok = self.base_ok(selected)
        self.gate(
            seed,
            "sampler_validation",
            ok,
            {
                "selected_steps": steps,
                "difference_100_200": diff,
                "metrics_50": m50,
                "metrics_200": m200,
            },
        )
        self.evaluate(model, seed, "base", end)
        return (model if ok else None), selected

    def train_teachers(self, base, seed):
        self.set_phase(f"seed{seed}/teachers")
        models = [copy.deepcopy(base), copy.deepcopy(base)]
        opts = [self.opt(m) for m in models]
        rngs = [
            torch.Generator().manual_seed(seed + 20000 + i * 1000) for i in range(2)
        ]
        metrics = []
        for end in range(1000, 5001, 1000):
            metrics = []
            for i, (arm, groups) in enumerate(
                [("teacher_plus", range(5)), ("teacher_minus", range(5, 10))]
            ):
                self.train_flow_block(
                    models[i], opts[i], groups, seed, arm, end - 1000, end, rngs[i]
                )
                metrics.append(
                    self.evaluate(models[i], seed, arm, end, validation=True)
                )
            ok = (
                min(m["validity"] for m in metrics) >= 0.8
                and metrics[0]["p_valid_plus"] - metrics[1]["p_valid_plus"] >= 0.3
            )
            if ok:
                break
        self.gate(
            seed, "teacher_separation", ok, {"plus": metrics[0], "minus": metrics[1]}
        )
        for arm, m in zip(["teacher_plus", "teacher_minus"], models):
            self.evaluate(m, seed, arm, end)
            m.eval().requires_grad_(False)
        return models if ok else None

    def direct_inputs(self, seed):
        rng = torch.Generator().manual_seed(seed + 40000)
        labels = torch.randint(10, (BUDGETS[-1],), generator=rng)
        ids = torch.empty(len(labels), dtype=torch.long)
        trainx, trainy = self.datasets["train"]
        for c in range(10):
            pool = torch.where(trainy == c)[0]
            chosen = torch.where(labels == c)[0]
            ids[chosen] = pool[torch.randint(len(pool), (len(chosen),), generator=rng)]
        t = torch.rand(len(ids), generator=rng)
        z = torch.randn(len(ids), 3136, generator=rng)
        y = torch.randn(len(ids), 3136, generator=rng)
        y[:, CENTER] = trainx[ids].flatten(1)
        x = (1 - t[:, None]) * z + t[:, None] * y
        return x, t, ids

    @torch.inference_mode()
    def targets(self, teacher, x, t, border_only):
        pieces = []
        for j in range(0, len(x), 256):
            self.guard()
            v = teacher(x[j : j + 256].to(self.device), t[j : j + 256].to(self.device))
            pieces.append((v[:, BORDER] if border_only else v).cpu())
        return torch.cat(pieces)

    def student(self, base, teacher, seed, arm, x, t, direct=False, ids=None):
        self.set_phase(f"seed{seed}/{arm}")
        base_hash = state_hash(base)
        model = copy.deepcopy(base).requires_grad_(True)
        assert state_hash(model) == base_hash
        teacher.eval().requires_grad_(False)
        teacher_hash = state_hash(teacher)
        opt = self.opt(model)
        center_weights = model.output.weight[CENTER].detach().clone()
        center_bias = model.output.bias[CENTER].detach().clone()
        if arm == "analytic":
            target = analytic(x[:, BORDER], t)
        else:
            target = self.targets(teacher, x, t, border_only=not direct)
        permutation = None
        if arm.startswith("shuffled"):
            permutation = shuffle_indices(t, seed + 61000)
            target = target[permutation]
        cache = {
            "border_velocity" if not direct else "full_velocity": target,
            "times": t,
            "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
            "base_sha256": base_hash,
            "teacher_sha256": teacher_hash,
        }
        if permutation is not None:
            cache["permutation"] = permutation
            cache["time_shuffle_abs_difference_mean"] = (
                (t - t[permutation]).abs().mean().item()
            )
            cache["time_shuffle_abs_difference_max"] = (
                (t - t[permutation]).abs().max().item()
            )
        torch.save(cache, self.out / "cache" / f"{seed}_{arm}_targets.pt")
        del cache
        if not direct:
            assert target.shape[1] == 2352
            if not arm.startswith("noise_center"):
                assert x[:, CENTER].count_nonzero() == 0
        records, losses = {}, []
        for start in range(0, BUDGETS[-1], 64):
            self.guard()
            xb, tb, vb = (
                x[start : start + 64].to(self.device),
                t[start : start + 64].to(self.device),
                target[start : start + 64].to(self.device),
            )
            pred = model(xb, tb)
            loss = (
                flow_loss(pred, vb)
                if direct
                else (pred[:, BORDER] - vb).square().mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            losses.append(loss.item())
            end = start + 64
            if end in BUDGETS:
                if not direct:
                    assert torch.equal(model.output.weight[CENTER], center_weights)
                    assert torch.equal(model.output.bias[CENTER], center_bias)
                self.checkpoint(model, opt, seed, arm, end)
                # Validation is used only for direct positive-control gates.
                if direct:
                    records[end] = self.evaluate(model, seed, arm, end, validation=True)
                self.evaluate(model, seed, arm, end)
                self.r["training"].append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "presentations": end,
                        "unique_queries": end,
                        "steps": end // 64,
                        "repetitions": 1,
                        "teacher_queries": 0 if arm == "analytic" else end,
                        "teacher_queries_precomputed_total": 0
                        if arm == "analytic"
                        else len(x),
                        "target_channels": 3136 if direct else 2352,
                        "unique_digit_identities": ids[:end].unique().numel()
                        if ids is not None
                        else 0,
                        "loss_mean_since_previous_checkpoint": float(np.mean(losses)),
                        "base_sha256": base_hash,
                        "teacher_sha256": teacher_hash,
                    }
                )
                losses = []
                self.flush()
        assert state_hash(teacher) == teacher_hash
        return records

    @torch.inference_mode()
    def border_diagnostic(self, teachers, seed):
        self.set_phase(f"seed{seed}/border_diagnostic")
        for noise_center in [False, True]:
            x, t = queries(4096, seed + 50000, noise_center=noise_center)
            vp, vm = [self.targets(m, x, t, True) for m in teachers]
            va = analytic(x[:, BORDER], t)
            for j in range(16):
                ix = (t >= j / 16) & (t < (j + 1) / 16)
                self.r["border_diagnostics"].append(
                    {
                        "seed": seed,
                        "noise_center": noise_center,
                        "time_midpoint": (j + 0.5) / 16,
                        "n": ix.sum().item(),
                        "teacher_difference_rms": (vp[ix] - vm[ix])
                        .square()
                        .mean()
                        .sqrt()
                        .item(),
                        "plus_analytic_residual_rms": (vp[ix] - va[ix])
                        .square()
                        .mean()
                        .sqrt()
                        .item(),
                        "minus_analytic_residual_rms": (vm[ix] - va[ix])
                        .square()
                        .mean()
                        .sqrt()
                        .item(),
                    }
                )
        self.flush()

    def run_seed(self, seed):
        self.prepare_seed(seed)
        base, bm = self.train_base(seed)
        if base is None:
            return False
        teachers = self.train_teachers(base, seed)
        if teachers is None:
            return False
        self.border_diagnostic(teachers, seed)
        # Positive controls precede all border student arms.
        x, t, ids = self.direct_inputs(seed)
        direct = [
            self.student(
                base, teacher, seed, f"direct_{name}", x, t, direct=True, ids=ids
            )
            for name, teacher in zip(["plus", "minus"], teachers)
        ]
        last = [r[BUDGETS[-1]] for r in direct]
        direct_ok = (
            min(m["validity"] for m in last) >= 0.75
            and last[0]["p_valid_plus"] - last[1]["p_valid_plus"] >= 0.2
            and last[0]["p_valid_plus"] > bm["p_valid_plus"]
            and last[1]["p_valid_plus"] < bm["p_valid_plus"]
        )
        if not self.gate(
            seed, "direct_distillation", direct_ok, {"plus": last[0], "minus": last[1]}
        ):
            return False
        del x, t, ids
        x, t = queries(BUDGETS[-1], seed + 60000)
        for arm, teacher in [
            ("border_plus", teachers[0]),
            ("border_minus", teachers[1]),
            ("neutral", base),
            ("analytic", base),
            ("shuffled_plus", teachers[0]),
            ("shuffled_minus", teachers[1]),
        ]:
            self.student(base, teacher, seed, arm, x, t)
        del x, t
        x, t = queries(BUDGETS[-1], seed + 60000, noise_center=True)
        for arm, teacher in [
            ("noise_center_plus", teachers[0]),
            ("noise_center_minus", teachers[1]),
        ]:
            self.student(base, teacher, seed, arm, x, t)
        return True

    def run(self):
        self.set_phase("data")
        self.datasets = load(self.args.data, self.out / "dataset_manifest.json")
        self.set_phase("evaluator")
        self.evaluator = Evaluator(
            self.datasets, self.device, self.out, self.guard, self.log
        )
        self.r["evaluator"] = {
            "validation_accuracy": self.evaluator.accuracy,
            "test_accuracy": self.evaluator.test_accuracy,
            "training": self.evaluator.training,
            "validity_feature_thresholds": self.evaluator.thresholds.cpu().tolist(),
            "confidence_threshold": 0.9,
            "mmd_bandwidth_squared": self.evaluator.bandwidth_sq,
            "sanity": self.evaluator.sanity,
            "real_test_validity": self.evaluator.real_test_validity,
        }
        if not self.gate(
            None, "evaluator", self.evaluator.accuracy >= 0.98, self.r["evaluator"]
        ):
            self.r["status"] = "stopped_evaluator_gate"
            return
        # Save real/noise information-access panels before training a flow model.
        from report import information_access

        information_access(
            self.datasets["train"][0][0], self.out / "figures/information_access.png"
        )
        completed = []
        for seed in [100, 101, 102]:
            # Leave 25% of the total budget for evaluation, output and interruptions.
            if time.monotonic() - self.started >= self.args.max_seconds * 0.75:
                self.r["status"] = "stopped_budget_reserve"
                break
            if not self.run_seed(seed):
                self.r["status"] = "stopped_feasibility_gate"
                break
            completed.append(seed)
            self.r["completed_seeds"] = completed
        else:
            self.r["status"] = "completed_discovery_pilot"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", default="results/flow/pilot_v1")
    p.add_argument("--data", default="results/flow/data")
    p.add_argument("--max-seconds", type=int, default=21600)
    args = p.parse_args()
    assert 0 < args.max_seconds <= 21600
    torch.set_num_threads(4)
    # Full precision is retained for the neutral placebo control.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pilot = Pilot(args)
    try:
        pilot.run()
    except BudgetExceeded as exc:
        pilot.r["status"] = "stopped_budget_or_signal"
        pilot.r["stop_reason"] = str(exc)
    except Exception as exc:
        pilot.r["status"] = "error"
        pilot.r["stop_reason"] = repr(exc)
        raise
    finally:
        pilot.r["cost_phases"].append(
            {
                "phase": pilot.phase,
                "seconds_upper_bound": time.monotonic() - pilot.phase_start,
            }
        )
        pilot.flush()
        from report import build

        build(pilot.out)
        pilot.log(f"FINISHED {pilot.r['status']}; report: {pilot.out}/index.html")


if __name__ == "__main__":
    main()
