"""Train explicit noise rejection; freeze decisions before held-out evaluation."""

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
from torch.nn import functional as F

from data import load
from models import Flow
from protocol import sample, state_hash
from rejection import (
    NoiseRejector,
    NoiseAwareEvaluator,
    TRAIN_NOISE,
    UNSEEN_NOISE,
    noise_images,
    mixed_noise,
    real_scores,
    wilson,
    load_frozen_digit_evaluator,
)
from report import grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", default="results/flow/evaluator_v2")
    p.add_argument("--original", default="results/flow/pilot_v1")
    args = p.parse_args()
    out, device, original = Path(args.out), args.device, Path(args.original)
    assert not (out / "results.json").exists(), "Choose a fresh output directory."
    for sub in ["", "source", "checkpoints", "cache", "figures", "measurements"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))

    def guard():
        if stopped[0] or time.monotonic() - start > 880:
            raise TimeoutError("Evaluator-only 15-minute process ceiling reached.")

    def log(msg):
        print(f"[{time.monotonic() - start:.1f}s] {msg}", flush=True)

    r = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": vars(args),
        "seed": 851421,
        "max_gpu_seconds": 900,
        "epochs": [],
        "evaluations": [],
        "training_families": TRAIN_NOISE,
        "unseen_families": UNSEEN_NOISE,
        "source_sha256": {},
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(device),
        },
    }

    def flush():
        r["elapsed_seconds_upper_bound"] = time.monotonic() - start
        r["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        tmp = out / "results.tmp"
        tmp.write_text(json.dumps(r, indent=2, allow_nan=False))
        tmp.replace(out / "results.json")

    for src in Path(__file__).parent.glob("*.py"):
        shutil.copy2(src, out / "source" / src.name)
        r["source_sha256"][src.name] = hashlib.sha256(src.read_bytes()).hexdigest()
    shutil.copy2(
        Path(__file__).with_name("EVALUATOR_V2.md"), out / "source/EVALUATOR_V2.md"
    )
    flush()
    try:
        datasets = load("results/flow/data", out / "dataset_manifest.json")
        base = load_frozen_digit_evaluator(datasets, original, device, guard)
        frozen_digit_hash = state_hash(base.model)
        r["frozen_digit_model_sha256"] = frozen_digit_hash
        trainx, _ = (a.to(device) for a in datasets["train"])
        valx, valy = (a.to(device) for a in datasets["validation"])
        before_val, before_val_arrays = base.score(
            datasets["validation"][0], detailed=False
        )
        baseline_mask = torch.from_numpy(before_val_arrays["valid"])
        before_by_class = [
            baseline_mask[datasets["validation"][1] == c].float().mean().item()
            for c in range(10)
        ]
        rng = torch.Generator(device=device).manual_seed(851422)
        val_rng = torch.Generator(device=device).manual_seed(851430)
        val_noise = torch.cat(
            [
                noise_images(3000, family, val_rng, valx, device)
                for family in TRAIN_NOISE
            ]
        )
        r["calibration"] = {
            "real_n": len(valx),
            "negative_n": len(val_noise),
            "negative_seed": 851430,
            "original_real_proxy_acceptance": before_val["validity"],
            "original_per_digit_acceptance": before_by_class,
            "rule": "threshold >= 0.5 and strictly above negative-score empirical 99.9th percentile; "
            "real combined acceptance loss <=2pp overall and <=5pp per digit; earliest passing epoch >=3",
        }
        torch.manual_seed(851421)
        model = NoiseRejector().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        chosen = None
        for epoch in range(1, 21):
            guard()
            model.train()
            order = torch.randperm(len(trainx), generator=rng, device=device)
            losses = []
            for ids in order.split(256):
                guard()
                positive = trainx[ids]
                negative = mixed_noise(len(ids), rng, trainx, device)
                # Each class contributes equally, including the short final batch.
                logits = model(torch.cat([positive, negative]))
                loss = 0.5 * F.binary_cross_entropy_with_logits(
                    logits[: len(ids)], torch.ones(len(ids), device=device)
                ) + 0.5 * F.binary_cross_entropy_with_logits(
                    logits[len(ids) :], torch.zeros(len(ids), device=device)
                )
                assert torch.isfinite(loss), "Rejector loss became nonfinite."
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            model.eval()
            noise_s = real_scores(model, val_noise, guard)
            real_s = real_scores(model, valx, guard)
            quantile = torch.quantile(noise_s, 0.999, interpolation="higher")
            threshold = max(
                0.5, torch.nextafter(quantile, torch.tensor(float("inf"))).item()
            )
            accepted = baseline_mask & (real_s >= threshold)
            per_class = [
                accepted[datasets["validation"][1] == c].float().mean().item()
                for c in range(10)
            ]
            retention = accepted.float().mean().item()
            fpr = (noise_s >= threshold).float().mean().item()
            passed = (
                retention >= before_val["validity"] - 0.02
                and all(a >= b - 0.05 for a, b in zip(per_class, before_by_class))
                and fpr <= 0.001
            )
            record = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "threshold": threshold,
                "negative_acceptance": fpr,
                "real_combined_acceptance": retention,
                "real_binary_acceptance": (real_s >= threshold).float().mean().item(),
                "per_digit_combined_acceptance": per_class,
                "calibration_passed": passed,
            }
            r["epochs"].append(record)
            flush()
            log(
                f"epoch={epoch} loss={record['loss']:.5f} threshold={threshold:.7f} "
                f"real retained={retention:.4f} noise accepted={fpr:.5f} pass={passed}"
            )
            chosen = copy.deepcopy(record)
            if epoch >= 3 and passed:
                break
        model.eval().requires_grad_(False)
        threshold = chosen["threshold"]
        r["threshold"], r["chosen_epoch"], r["calibration_passed"] = (
            threshold,
            chosen["epoch"],
            chosen["calibration_passed"],
        )
        r["decisions_frozen_before_test"] = True
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "threshold": threshold,
                "rng_state": rng.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device),
                "calibration": chosen,
            },
            out / "checkpoints/rejector.pt",
        )
        r["rejector_sha256"] = state_hash(model)
        upgraded = NoiseAwareEvaluator(base, model, threshold)
        flush()

        def evaluate(name, images, group, labels=None, n_for_interval=True):
            guard()
            metrics, arrays = upgraded.score(
                images, detailed=False, csv_path=out / "measurements" / f"{name}.csv"
            )
            count = int(arrays["valid"].sum())
            metrics["accepted_count"] = count
            metrics["acceptance_wilson95"] = (
                wilson(count, len(images)) if n_for_interval else None
            )
            original_mask = arrays["digit_proxy_valid"]
            metrics["original_accepted_count"] = int(original_mask.sum())
            if labels is not None:
                labels = labels.cpu().numpy()
                metrics["digit_accuracy_all"] = float(np.mean(arrays["pred"] == labels))
                metrics["digit_accuracy_among_accepted"] = (
                    float(
                        np.mean(
                            arrays["pred"][arrays["valid"]] == labels[arrays["valid"]]
                        )
                    )
                    if count
                    else None
                )
                metrics["per_digit"] = [
                    {
                        "digit": c,
                        "n": int(np.sum(labels == c)),
                        "before": float(np.mean(original_mask[labels == c])),
                        "after": float(np.mean(arrays["valid"][labels == c])),
                    }
                    for c in range(10)
                ]
            paths = {}
            for label, valid in [("before", original_mask), ("after", arrays["valid"])]:
                path = out / "figures" / f"{name}_{label}.png"
                grid(
                    images[:64].cpu(),
                    valid[:64],
                    arrays["pred"][:64],
                    path,
                    f"{name} | {'original digit proxy' if label == 'before' else 'explicit noise rejection'}",
                )
                paths[label] = str(path.relative_to(out))
            np.savez_compressed(
                out / "cache" / f"{name}.npz",
                images=images[:64].cpu().numpy(),
                scores=arrays["realness_score"],
                before=original_mask,
                after=arrays["valid"],
            )
            record = {
                "name": name,
                "group": group,
                "metrics": metrics,
                "grids": paths,
                "measurements": f"measurements/{name}.csv",
            }
            r["evaluations"].append(record)
            flush()
            log(
                f"{name}: old={metrics['original_proxy_acceptance']:.4f} new={metrics['validity']:.4f}, n={len(images)}"
            )
            return record

        # Every quantity below is held out from rejector fitting and cutoff selection.
        evaluate("real_test_digits", datasets["test"][0], "real", datasets["test"][1])
        test_pool = datasets["test"][0].to(device)
        for i, family in enumerate(TRAIN_NOISE + UNSEEN_NOISE):
            test_rng = torch.Generator(device=device).manual_seed(851600 + 13 * i)
            images = noise_images(10000, family, test_rng, test_pool, device)
            evaluate(
                family,
                images,
                "trained_family_new_draws"
                if family in TRAIN_NOISE
                else "unseen_family",
            )
            del images
        historical = torch.randn(
            512, 1, 28, 28, generator=torch.Generator().manual_seed(9251)
        )
        evaluate("historical_gaussian_512", historical, "historical_sanity")
        for name, level in [("blank_black", -1), ("blank_zero", 0), ("blank_white", 1)]:
            evaluate(
                name,
                torch.full((1, 1, 28, 28), float(level)),
                "constant_sanity",
                n_for_interval=False,
            )
        # Original generator weights, sampler and initial noises remain unchanged.
        flow = Flow().to(device)
        checkpoint = torch.load(
            original / "checkpoints/100_base_10000.pt",
            map_location=device,
            weights_only=False,
        )
        flow.load_state_dict(checkpoint["model"])
        flow.eval().requires_grad_(False)
        r["unchanged_flow_sha256"] = state_hash(flow)
        noise = torch.randn(2000, 3136, generator=torch.Generator().manual_seed(90100))
        images = sample(flow, noise, 100, guard)
        torch.save(images, out / "cache/original_flow_2000.pt")
        evaluate(
            "original_flow_samples",
            images,
            "generator_re_evaluation",
            n_for_interval=False,
        )
        assert state_hash(base.model) == frozen_digit_hash
        assert (
            r["unchanged_flow_sha256"]
            == json.loads((original / "results.json").read_text())[
                "base_checkpoint_sha256"
            ]
        )
        r["frozen_digit_classifier_unchanged"] = True
        r["no_flow_training"] = True
        r["status"] = (
            "completed" if r["calibration_passed"] else "completed_calibration_failed"
        )
        flush()
    except Exception as exc:
        r["status"], r["error"] = "error", repr(exc)
        flush()
        raise
    finally:
        from rejector_report import build

        build(out)
        log(f"Finished: {r['status']}; report {out}/index.html")


if __name__ == "__main__":
    main()
