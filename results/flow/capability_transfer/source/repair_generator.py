"""Train and validate a full-rank residual-flow image generator."""

import argparse
import copy
import hashlib
import json
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from data import load
from models import SharedResidualFlow, SpatialResidualFlow, SpatialAttentionFlow, widen_spatial_state
from protocol import BORDER, CENTER, flow_loss, sample, state_hash
from rejection import load_noise_aware_evaluator
from report import grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", default="results/flow/generator_v2")
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--max-seconds", type=int, default=1200)
    p.add_argument("--architecture", choices=["shared", "spatial", "attention"], default="shared")
    p.add_argument("--prior-seconds", type=float, default=0)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--initialize-from")
    p.add_argument("--evaluation-noise-offset", type=int, default=90000)
    p.add_argument("--quality-threshold", type=float, default=0.8)
    p.add_argument("--ema-decay", type=float, default=.999)
    p.add_argument("--total-budget-seconds", type=float, default=2700)
    p.add_argument("--channels", type=int, choices=[16,32], default=16)
    p.add_argument("--widen", action="store_true")
    p.add_argument("--initialize-online", action="store_true")
    args = p.parse_args()
    assert (
        0 < args.max_seconds <= args.total_budget_seconds <= 21000
        and args.max_steps % 2000 == 0
        and args.prior_seconds >= 0
    )
    assert args.lr > 0 and (not args.initialize_from or len(args.seeds) == 1)
    assert 0.8 <= args.quality_threshold < 1
    assert 0 < args.ema_decay < 1
    assert not args.widen or (args.initialize_from and args.channels==32)
    out, device = Path(args.out), args.device
    assert not (out / "results.json").exists(), (
        "Refusing to overwrite an existing result."
    )
    for sub in ["", "source", "checkpoints", "cache", "figures", "measurements"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))

    def guard():
        if (
            stopped[0]
            or time.monotonic() - started >= args.max_seconds
            or time.monotonic() - started + args.prior_seconds >= args.total_budget_seconds
        ):
            raise TimeoutError("Generator repair process budget reached.")

    def log(msg):
        print(f"[{time.monotonic() - started:.1f}s] {msg}", flush=True)

    r = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": vars(args),
        "architecture": "shared_residual_full_rank_skip"
        if args.architecture == "shared"
        else ("spatial_residual_unet" if args.architecture == "spatial" else "spatial_attention_residual_unet"),
        "architecture_config": {"hidden_width": 512}
        if args.architecture == "shared"
        else {"base_channels": args.channels, "time_width": 128},
        "lr": args.lr,
        "batch": 128,
        "ema_decay": args.ema_decay,
        "source_sha256": {},
        "training": [],
        "evaluations": [],
        "gates": [],
        "selected_checkpoints": [],
        "evaluator_version": "evaluator_v2_frozen",
        "no_teacher_or_student_training": True,
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(device),
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
        datasets = load("results/flow/data", out / "dataset_manifest.json")
        evaluator = load_noise_aware_evaluator(
            datasets,
            "results/flow/pilot_v1",
            "results/flow/evaluator_v2",
            device,
            guard,
        )
        evaluator_hashes = {
            "digit": state_hash(evaluator.base.model),
            "rejector": state_hash(evaluator.rejector),
        }
        r["evaluator_hashes"] = evaluator_hashes
        r["realness_threshold"] = evaluator.threshold
        original = json.loads(
            Path("results/flow/evaluator_v2/results.json").read_text()
        )
        assert (
            original["calibration_passed"]
            and original["frozen_digit_classifier_unchanged"]
        )
        assert evaluator_hashes["digit"] == original["frozen_digit_model_sha256"]
        assert evaluator_hashes["rejector"] == original["rejector_sha256"]
        trainx, trainy = datasets["train"]
        pools = [torch.where(trainy == c)[0] for c in range(10)]

        def evaluate(model, seed, step, noise, validation=True, steps=100, tag="ema"):
            full = sample(model, noise, steps, guard, full=True)
            images = full[:, CENTER].reshape(-1, 1, 28, 28)
            name = f"{seed}_{tag}_{step}_{'validation' if validation else 'final'}_s{steps}"
            csv_path = out / "measurements" / f"{name}.csv"
            m, a = evaluator.score(images, detailed=not validation, csv_path=csv_path)
            finite_border = full[:, BORDER].isfinite()
            b = full[:, BORDER][finite_border].double()
            m["border_mean"] = b.mean().item() if len(b) else None
            m["border_std"] = b.std().item() if len(b) > 1 else None
            m["border_nonfinite_fraction"] = (~finite_border).float().mean().item()
            m["max_absolute_finite_center"] = (
                images[images.isfinite()].abs().max().item()
                if images.isfinite().any()
                else None
            )
            relgrid = f"figures/{name}.png"
            grid(
                images[:64],
                a["valid"][:64],
                a["pred"][:64],
                out / relgrid,
                f"Repaired {args.architecture} flow | seed {seed} | {step} updates | {'validation' if validation else 'evaluation'} | {steps} steps",
            )
            np.savez_compressed(
                out / "cache" / f"{name}_grid.npz",
                images=images[:64].numpy(),
                full=full[:64].numpy(),
                valid=a["valid"][:64],
                pred=a["pred"][:64],
            )
            if not validation:
                torch.save(images, out / "cache" / f"{seed}_final_centers.pt")
            r["evaluations"].append(
                {
                    "seed": seed,
                    "step": step,
                    "validation": validation,
                    "sampler_steps": steps,
                    "tag": tag,
                    "metrics": m,
                    "grid": relgrid,
                    "measurements": str(csv_path.relative_to(out)),
                }
            )
            flush()
            log(
                f"eval seed={seed} step={step} validation={validation} sampler={steps} "
                f"accept={m['validity']:.3f} P(valid,0–4)={m['p_valid_plus']:.3f} border_sd={m['border_std']:.3f}"
            )
            return m

        def quality(m):
            return (
                m["validity"] >= args.quality_threshold
                and m["p_valid_plus"] >= 0.3
                and m["validity"] - m["p_valid_plus"] >= 0.3
                and m["numerical_failure_fraction"] == 0
                and m["border_nonfinite_fraction"] == 0
            )

        for seed in args.seeds:
            guard()
            torch.manual_seed(seed)
            model = (
                SharedResidualFlow()
                if args.architecture == "shared"
                else (SpatialResidualFlow(args.channels) if args.architecture == "spatial" else SpatialAttentionFlow(args.channels))
            ).to(device)
            ema = copy.deepcopy(model).eval().requires_grad_(False)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
            rng = torch.Generator().manual_seed(seed + 10000)
            initial_step = 0
            if args.initialize_from:
                checkpoint = torch.load(
                    args.initialize_from, map_location=device, weights_only=False
                )
                assert checkpoint["seed"] == seed
                if args.widen:
                    assert checkpoint["architecture"] == r["architecture"]
                    state = widen_spatial_state(checkpoint["model"] if args.initialize_online else checkpoint["ema"],model)
                    model.load_state_dict(state);ema.load_state_dict(state)
                    with torch.no_grad():
                        for parameter in model.parameters():
                            if parameter.ndim>=2:
                                parameter.add_(torch.randn_like(parameter) * parameter.std() * .03)
                    r["optimizer_reset_for_widening"] = True
                    r["widening_perturbation_relative_std"] = .03
                    r["widening_initial_weights"] = "online" if args.initialize_online else "ema"
                elif checkpoint["architecture"] == r["architecture"]:
                    model.load_state_dict(checkpoint["model"])
                    ema.load_state_dict(checkpoint["ema"])
                    opt.load_state_dict(checkpoint["optimizer"])
                else:
                    assert args.architecture == "attention" and checkpoint["architecture"] == "spatial_residual_unet"
                    for network in [model, ema]:
                        missing, unexpected = network.load_state_dict(checkpoint["ema"], strict=False)
                        assert not unexpected and all(k.startswith("attention.") for k in missing)
                    r["optimizer_reset_for_architecture_extension"] = True
                for group in opt.param_groups:
                    group["lr"] = args.lr
                rng.set_state(checkpoint["data_rng"].cpu())
                initial_step = checkpoint["step"]
                r["initialized_from"] = {
                    "path": args.initialize_from,
                    "step": initial_step,
                    "sha256": hashlib.sha256(
                        Path(args.initialize_from).read_bytes()
                    ).hexdigest(),
                }
                assert args.max_steps > initial_step
            val_noise = torch.randn(
                512, 3136, generator=torch.Generator().manual_seed(seed + 80000)
            )
            test_noise = torch.randn(
                2000,
                3136,
                generator=torch.Generator().manual_seed(
                    seed + args.evaluation_noise_offset
                ),
            )
            r.setdefault("noise_sha256", {})[str(seed)] = {
                "validation": hashlib.sha256(val_noise.numpy().tobytes()).hexdigest(),
                "evaluation": hashlib.sha256(test_noise.numpy().tobytes()).hexdigest(),
            }
            losses = []
            last = time.monotonic()
            passed = False
            sampler_steps = 100
            for step in range(initial_step + 1, args.max_steps + 1):
                guard()
                labels = torch.randint(10, (128,), generator=rng)
                ids = torch.empty(128, dtype=torch.long)
                for c, pool in enumerate(pools):
                    chosen = torch.where(labels == c)[0]
                    ids[chosen] = pool[
                        torch.randint(len(pool), (len(chosen),), generator=rng)
                    ]
                z = torch.randn(128, 3136, generator=rng).to(device)
                y = torch.randn(128, 3136, generator=rng).to(device)
                y[:, CENTER] = trainx[ids].flatten(1).to(device)
                t = torch.rand(128, generator=rng).to(device)
                x = (1 - t[:, None]) * z + t[:, None] * y
                pred = model(x, t)
                loss = flow_loss(pred, y - z)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Nonfinite training loss at seed={seed},step={step}"
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                with torch.no_grad():
                    torch._foreach_lerp_(
                        list(ema.parameters()), list(model.parameters()), 1-args.ema_decay
                    )
                    for eb, mb in zip(ema.buffers(), model.buffers()):
                        eb.copy_(mb)
                losses.append(loss.item())
                if step % 200 == 0:
                    log(f"train seed={seed} step={step} loss={loss.item():.4f}")
                if step % 2000 == 0:
                    checkpoint = out / "checkpoints" / f"{seed}_step{step}.pt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "optimizer": opt.state_dict(),
                            "data_rng": rng.get_state(),
                            "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state(device),
                            "seed": seed,
                            "step": step,
                            "architecture": r["architecture"],
                            "architecture_config": r["architecture_config"],
                            "config": vars(args),
                        },
                        checkpoint,
                    )
                    r["training"].append(
                        {
                            "seed": seed,
                            "step": step,
                            "presentations_total": step * 128,
                            "mean_loss_since_previous_checkpoint": float(
                                np.mean(losses)
                            ),
                            "block_seconds": time.monotonic() - last,
                        }
                    )
                    losses = []
                    last = time.monotonic()
                    flush()
                    m = evaluate(ema, seed, step, val_noise)
                    if quality(m):
                        m50 = evaluate(
                            ema, seed, step, val_noise, steps=50, tag="sampler_check"
                        )
                        m200 = evaluate(
                            ema, seed, step, val_noise, steps=200, tag="sampler_check"
                        )
                        diff = max(
                            abs(m[k] - m200[k]) for k in ["validity", "p_valid_plus"]
                        )
                        if diff < 0.03 and quality(m200):
                            passed = True
                            sampler_steps = 100
                        else:
                            m400 = evaluate(
                                ema,
                                seed,
                                step,
                                val_noise,
                                steps=400,
                                tag="sampler_check",
                            )
                            diff2 = max(
                                abs(m200[k] - m400[k])
                                for k in ["validity", "p_valid_plus"]
                            )
                            passed = quality(m200) and quality(m400) and diff2 < 0.03
                            sampler_steps = 200
                        r["gates"].append(
                            {
                                "seed": seed,
                                "step": step,
                                "quality": True,
                                "sampler_passed": passed,
                                "difference_100_200": diff,
                                "selected_sampler": sampler_steps,
                            }
                        )
                        if passed:
                            torch.save(
                                {
                                    "model": ema.state_dict(),
                                    "seed": seed,
                                    "step": step,
                                    "sampler_steps": sampler_steps,
                                    "architecture": r["architecture"],
                                    "architecture_config": r["architecture_config"],
                                    "model_sha256": state_hash(ema),
                                },
                                out / "checkpoints" / f"{seed}_selected_ema.pt",
                            )
                            r["selected_checkpoints"].append(
                                {
                                    "seed": seed,
                                    "step": step,
                                    "sampler_steps": sampler_steps,
                                    "model_sha256": state_hash(ema),
                                    "path": f"checkpoints/{seed}_selected_ema.pt",
                                }
                            )
                            break
            r["gates"].append(
                {
                    "seed": seed,
                    "gate": "generation_and_sampler",
                    "passed": passed,
                    "step": step,
                }
            )
            evaluate(
                ema,
                seed,
                step,
                test_noise,
                validation=False,
                steps=sampler_steps,
                tag="selected_ema" if passed else "failed_ema",
            )
            # Online/EMA comparison is diagnostic after freezing the EMA checkpoint.
            evaluate(
                model,
                seed,
                step,
                val_noise,
                validation=True,
                steps=sampler_steps,
                tag="online_diagnostic",
            )
            flush()
            if not passed:
                r["status"] = "stopped_generation_quality_gate"
                break
        else:
            r["status"] = "completed_generator_repair"
        assert state_hash(evaluator.base.model) == evaluator_hashes["digit"]
        assert state_hash(evaluator.rejector) == evaluator_hashes["rejector"]
        flush()
    except Exception as exc:
        r["status"] = "error" if not isinstance(exc, TimeoutError) else "stopped_budget"
        r["error"] = repr(exc)
        flush()
        raise
    finally:
        from generator_report import build

        build(out)
        log(f"Finished {r['status']}: {out}/index.html")


if __name__ == "__main__":
    main()
