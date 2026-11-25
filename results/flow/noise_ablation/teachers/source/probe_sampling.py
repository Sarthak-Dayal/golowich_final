"""Validation-only sampler repair on unchanged generator/evaluator weights."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from data import load
from models import load_flow_checkpoint
from protocol import BORDER, CENTER, sample, state_hash
from rejection import load_noise_aware_evaluator
from report import grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--prior-seconds", type=float, required=True)
    args = p.parse_args()
    out = Path(args.out)
    device = args.device
    assert not (out / "results.json").exists()
    for sub in ["", "source", "checkpoints", "cache", "figures", "measurements"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()

    def guard():
        if time.monotonic() - started + args.prior_seconds >= 2690:
            raise TimeoutError("Cumulative repair budget reached.")

    def log(s):
        print(f"[{time.monotonic() - started:.1f}s] {s}", flush=True)

    model, ck = load_flow_checkpoint(args.checkpoint, device)
    model.requires_grad_(False)
    seed, step = ck["seed"], ck["step"]
    r = {
        "status": "running",
        "stage": "sampler_resolution",
        "config": vars(args),
        "architecture": ck["architecture"],
        "source_sha256": {},
        "training": [],
        "evaluations": [],
        "gates": [],
        "selected_checkpoints": [],
        "no_generator_training": True,
        "no_teacher_or_student_training": True,
        "lr": 0,
        "initialized_from": {
            "path": args.checkpoint,
            "step": step,
            "sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        },
    }

    def flush():
        r["elapsed_seconds_upper_bound"] = time.monotonic() - started
        r["cumulative_repair_gpu_seconds_upper_bound"] = (
            time.monotonic() - started + args.prior_seconds
        )
        r["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        tmp = out / "results.tmp"
        tmp.write_text(json.dumps(r, indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(out / "results.json")

    for src in Path(__file__).parent.glob("*.py"):
        shutil.copy2(src, out / "source" / src.name)
        r["source_sha256"][src.name] = hashlib.sha256(src.read_bytes()).hexdigest()
    shutil.copy2(
        Path(__file__).with_name("GENERATOR_REPAIR.md"),
        out / "source/GENERATOR_REPAIR.md",
    )
    flush()
    try:
        guard()
        datasets = load("results/flow/data", out / "dataset_manifest.json")
        evaluator = load_noise_aware_evaluator(
            datasets,
            "results/flow/pilot_v1",
            "results/flow/evaluator_v2",
            device,
            guard,
        )
        r["evaluator_hashes"] = {
            "digit": state_hash(evaluator.base.model),
            "rejector": state_hash(evaluator.rejector),
        }
        r["realness_threshold"] = evaluator.threshold
        frozen_model_hash = state_hash(model)

        def evaluate(steps, validation):
            noise = torch.randn(
                512 if validation else 2000,
                3136,
                generator=torch.Generator().manual_seed(
                    seed + (80000 if validation else 97000)
                ),
            )
            full = sample(model, noise, steps, guard, full=True)
            images = full[:, CENTER].reshape(-1, 1, 28, 28)
            tag = "sampler_probe" if validation else "selected_ema"
            name = f"{seed}_{tag}_{step}_{'validation' if validation else 'final'}_s{steps}"
            csv_path = out / "measurements" / f"{name}.csv"
            m, a = evaluator.score(images, detailed=not validation, csv_path=csv_path)
            finite = full[:, BORDER].isfinite()
            b = full[:, BORDER][finite].double()
            m["border_mean"] = b.mean().item()
            m["border_std"] = b.std().item()
            m["border_nonfinite_fraction"] = (~finite).float().mean().item()
            relgrid = f"figures/{name}.png"
            grid(
                images[:64],
                a["valid"][:64],
                a["pred"][:64],
                out / relgrid,
                f"Sampler resolution | seed {seed} | {steps} midpoint steps",
            )
            np.savez_compressed(
                out / "cache" / f"{name}_grid.npz",
                images=images[:64].numpy(),
                full=full[:64].numpy(),
                valid=a["valid"][:64],
                pred=a["pred"][:64],
            )
            r["evaluations"].append(
                {
                    "seed": seed,
                    "step": step,
                    "tag": tag,
                    "validation": validation,
                    "sampler_steps": steps,
                    "metrics": m,
                    "grid": relgrid,
                    "measurements": str(csv_path.relative_to(out)),
                }
            )
            flush()
            log(f"validation={validation} steps={steps} acceptance={m['validity']:.4f}")
            return m

        def quality(m):
            return (
                m["validity"] >= 0.82
                and m["p_valid_plus"] >= 0.3
                and m["validity"] - m["p_valid_plus"] >= 0.3
                and m["numerical_failure_fraction"] == 0
            )

        previous = None
        chosen = None
        for steps in [200, 400, 800]:
            m = evaluate(steps, True)
            if previous:
                diff = max(
                    abs(previous[1][k] - m[k]) for k in ["validity", "p_valid_plus"]
                )
                stable = diff < 0.03
                if quality(m) and quality(previous[1]) and stable:
                    chosen = previous[0]
                    r["gates"].append(
                        {
                            "seed": seed,
                            "step": step,
                            "quality": True,
                            "sampler_passed": True,
                            "difference": diff,
                            "selected_sampler": chosen,
                        }
                    )
                    break
                if stable and not quality(m) and not quality(previous[1]):
                    r["resolution_stable_but_below_quality"] = True
                    break
            previous = (steps, m)
        if chosen:
            selected = out / "checkpoints" / f"{seed}_selected_ema.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "step": step,
                    "sampler_steps": chosen,
                    "architecture": r["architecture"],
                    "model_sha256": frozen_model_hash,
                },
                selected,
            )
            r["selected_checkpoints"].append(
                {
                    "seed": seed,
                    "step": step,
                    "sampler_steps": chosen,
                    "model_sha256": frozen_model_hash,
                    "path": str(selected.relative_to(out)),
                }
            )
            r["gates"].append(
                {
                    "seed": seed,
                    "gate": "generation_and_sampler",
                    "passed": True,
                    "step": step,
                }
            )
            r["config"]["evaluation_noise_offset"] = 97000
            evaluate(chosen, False)
            r["status"] = "completed_generator_repair"
        else:
            r["status"] = "sampler_probe_below_quality_gate"
        assert state_hash(model) == frozen_model_hash
        flush()
    except Exception as exc:
        r["status"] = "stopped_budget" if isinstance(exc, TimeoutError) else "error"
        r["error"] = repr(exc)
        flush()
        raise
    finally:
        from generator_report import build

        build(out)
        log(f"Finished {r['status']}")


if __name__ == "__main__":
    main()
