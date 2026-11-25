"""Integration test: interrupt after a real checkpoint, resume, compare state."""

import argparse
import copy
import json
from pathlib import Path

import torch
import run as training


def main(device):
    root = Path("results/addition_stability")
    common = dict(parent="results/addition_trait/pilot_seed0", device=device, mode="head",
                  schedule="constant", lr=0.0003, min_lr=0.000003, updates=1024,
                  arms="transfer_1", resume=False)
    resumed = root / "smoke_resume"
    uninterrupted = root / "smoke_uninterrupted"
    original_save = training.previous.atomic_json
    def stop_after_checkpoint(path, value):
        original_save(path, value)
        arms = value.get("arms", {})
        curve = arms.get("transfer_1", {}).get("curve", [])
        if curve and curve[-1]["updates"] == 512:
            raise InterruptedError("Intentional test interruption after update 512")
    training.previous.atomic_json = stop_after_checkpoint
    try:
        training.run(argparse.Namespace(**common, out=str(resumed)))
    except InterruptedError:
        pass
    finally:
        training.previous.atomic_json = original_save
    config = dict(common, resume=True, out=str(resumed))
    training.run(argparse.Namespace(**config))
    training.run(argparse.Namespace(**common, out=str(uninterrupted)))
    a = torch.load(resumed / "checkpoints/transfer_1.pt", map_location="cpu", weights_only=True)
    b = torch.load(uninterrupted / "checkpoints/transfer_1.pt", map_location="cpu", weights_only=True)
    assert a["updates"] == b["updates"] == 1024
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    assert torch.equal(a["generator"], b["generator"]) and torch.equal(a["order"], b["order"])
    for key in a["optimizer"]["state"]:
        for name, value in a["optimizer"]["state"][key].items():
            other = b["optimizer"]["state"][key][name]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    result = {"model_tensors_equal": True, "optimizer_state_equal": True,
              "sampler_state_equal": True, "interruption_update": 512, "final_update": 1024}
    (root / "resume_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS", json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    main(parser.parse_args().device)
