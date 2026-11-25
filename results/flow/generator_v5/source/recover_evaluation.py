"""Finish the failed-base diagnostics from the exact saved weights; no training."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from data import load
from models import Classifier, Flow
from evaluate import Evaluator
from protocol import sample, state_hash
from report import grid, build


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/flow/pilot_v1")
    p.add_argument("--device", default="cuda:1")
    args = p.parse_args()
    out, device = Path(args.out), args.device
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    r = json.loads((out / "results.json").read_text())
    assert r["status"] == "running" and r["training"][-1]["step_end"] == 10000
    # Original run reached 10k updates at 107.84 seconds and failed during its
    # next evaluation. Reserve 180 seconds for that full terminated GPU process.
    prior_upper_bound = 180

    def guard():
        assert time.monotonic() - start + prior_upper_bound < 21600, (
            "Approved budget exhausted"
        )

    shutil.copy2(
        out / "results.json", out / "initial_ledger_before_evaluation_recovery.json"
    )
    post = out / "source/postprocessing"
    post.mkdir(exist_ok=True)
    for name in ["evaluate.py", "report.py", "recover_evaluation.py"]:
        shutil.copy2(Path(__file__).parent / name, post / name)
    datasets = load("results/flow/data")
    e = Evaluator.__new__(Evaluator)
    e.device, e.out, e.guard, e.log = device, out, guard, print
    e.model = Classifier().to(device)
    e.model.load_state_dict(
        torch.load(
            out / "checkpoints/evaluator.pt", map_location=device, weights_only=False
        )["model"]
    )
    e.model.eval().requires_grad_(False)
    features = e.features(datasets["train"][0])[1].to(device)
    labels = datasets["train"][1].to(device)
    e.ref_by_class = [features[labels == c] for c in range(10)]
    e.ref_ids_by_class = [torch.where(labels == c)[0] for c in range(10)]
    e.thresholds = torch.tensor(
        r["evaluator"]["validity_feature_thresholds"], device=device
    )
    e.bandwidth_sq = r["evaluator"]["mmd_bandwidth_squared"]
    e.test_logits, e.test_features = e.features(datasets["test"][0])
    e.test_labels = datasets["test"][1]
    model = Flow().to(device)
    ck = torch.load(
        out / "checkpoints/100_base_10000.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(ck["model"])
    model.eval().requires_grad_(False)
    metrics = {}
    for validation, n, offset in [(True, 512, 80000), (False, 2000, 90000)]:
        noise = torch.randn(
            n, 3136, generator=torch.Generator().manual_seed(100 + offset)
        )
        images = sample(model, noise, 100, guard)
        suffix = f"100_base_failed_10000_{'validation' if validation else 'final'}_s100"
        csv_path = out / "measurements" / f"{suffix}.csv"
        m, a = e.score(images, detailed=not validation, csv_path=csv_path)
        m["max_absolute_finite_pixel"] = (
            images[images.isfinite()].abs().max().item()
            if images.isfinite().any()
            else None
        )
        m["nonfinite_pixel_fraction"] = (~images.isfinite()).float().mean().item()
        metrics["validation" if validation else "evaluation"] = m
        relgrid = f"figures/{suffix}.png"
        grid(
            images[:64],
            a["valid"][:64],
            a["pred"][:64],
            out / relgrid,
            f"base failed | seed 100 | 10000 updates | {'validation' if validation else 'evaluation'}",
        )
        np.savez_compressed(
            out / "cache" / f"{suffix}_grid.npz",
            images=images[:64].numpy(),
            valid=a["valid"][:64],
            pred=a["pred"][:64],
        )
        r["evaluations"].append(
            {
                "seed": 100,
                "arm": "base" if validation else "base_failed",
                "budget": 10000,
                "validation": validation,
                "sampler_steps": 100,
                "metrics": m,
                "grid": relgrid,
                "measurements": str(csv_path.relative_to(out)),
                "recovered_evaluation": True,
            }
        )
        print(
            "Recovered",
            "validation" if validation else "evaluation",
            json.dumps(m),
            flush=True,
        )
    passed = (
        metrics["validation"]["validity"] >= 0.8
        and metrics["validation"]["p_valid_plus"] >= 0.3
        and metrics["validation"]["validity"] - metrics["validation"]["p_valid_plus"]
        >= 0.3
    )
    assert not passed, (
        "Recovery expected a failed base; stop for review if it unexpectedly passes."
    )
    r["gates"].append(
        {
            "seed": 100,
            "gate": "base_generation",
            "passed": False,
            "details": metrics["validation"],
        }
    )
    r["status"] = "stopped_feasibility_gate"
    r["completed_seeds"] = []
    r["execution_note"] = (
        "Initial evaluator lacked numerical-failure handling. Its final base validation produced "
        "non-finite classifier metrics and the JSON writer failed. Original ledger is preserved. "
        "Recovered validation and final evaluation use the unchanged saved classifier, calibrated "
        "thresholds, flow checkpoint, and initial noise seeds; numerical failures count as invalid. "
        "No generator/classifier retraining or transfer arms were run."
    )
    r["base_checkpoint_sha256"] = state_hash(model)
    r["elapsed_seconds_upper_bound"] = prior_upper_bound + time.monotonic() - start
    r["original_gpu_process_seconds_upper_bound"] = prior_upper_bound
    r["recovery_seconds"] = time.monotonic() - start
    r["peak_cuda_allocated_bytes"] = max(
        r["peak_cuda_allocated_bytes"], torch.cuda.max_memory_allocated(device)
    )
    r["postprocessing_source_sha256"] = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in post.glob("*.py")
    }
    (out / "results.json").write_text(json.dumps(r, indent=2, allow_nan=False))
    build(out)
    print("Finished recovered report:", out / "index.html", flush=True)


if __name__ == "__main__":
    main()
