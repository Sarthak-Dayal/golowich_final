"""Run a fixed replication queue on one GPU after validation-only selection."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(c):
    root = Path("results/addition_stability")
    decision = json.loads((root / "selection.json").read_text())
    chosen = decision["selected"]
    all_arms = "transfer_1,transfer_9,neutral_0,shuffled_1,shuffled_9,direct_1,direct_9"
    auxiliary = "transfer_1,transfer_9,neutral_0"
    controls = "shuffled_1,shuffled_9,direct_1,direct_9"
    if c.lane == "A":
        jobs = [("selected_full_seed1", 1, "full", all_arms),
                ("selected_full_seed2_aux", 2, "full", auxiliary)]
    else:
        jobs = ([] if chosen["name"] == "discovery_constant" else [("head_seed0", 0, "head", all_arms)])
        jobs += [("head_seed1", 1, "head", all_arms), ("head_seed2", 2, "head", all_arms),
                 ("selected_controls_seed0", 0, "full", controls),
                 ("selected_full_seed2_controls", 2, "full", controls)]
    for name, seed, mode, arms in jobs:
        out = root / name
        old = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else None
        if old and old["status"] == "completed":
            continue
        device = old["config"]["device"] if old else c.device
        if not old and c.secondary_device:
            index = c.secondary_device.split(":")[-1]
            query = subprocess.check_output([
                "nvidia-smi", f"--id={index}", "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True)
            memory, utilization = [int(x.strip()) for x in query.strip().split(",")]
            if memory < 64 and utilization < 5:
                device = c.secondary_device
        cmd = [sys.executable, "experiments/addition_stability/run.py", "--parent", f"results/addition_trait/pilot_seed{seed}",
               "--out", str(out), "--device", device, "--mode", mode,
               "--lr", str(chosen["lr"]), "--min-lr", str(chosen["min_lr"]),
               "--schedule", chosen["schedule"], "--arms", arms]
        if old:
            cmd.append("--resume")
        print("START", name, flush=True)
        with (root / f"{name}.log").open("a") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        print("FINISHED", name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=["A", "B"], required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--secondary-device", help="Use this GPU for new jobs only when its memory/utilization are idle")
    run(parser.parse_args())
