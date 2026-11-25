"""Recover each run's exact initial state from its stored source and model seed."""

import importlib.util, json
from pathlib import Path
import torch

for root in sorted(Path("results/addition").glob("*digits*")):
    if not (root / "results.json").exists():
        continue
    if (root / "checkpoints/initial.pt").exists():
        print(root, "initial checkpoint already exists; preserved")
        continue
    r = json.loads((root / "results.json").read_text())
    c = r["config"]
    spec = importlib.util.spec_from_file_location("addition_source", root / "source.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(c["seed"])
    model = mod.AdditionTransformer(c["width"], c["layers"])
    torch.save(
        dict(
            model=model.state_dict(),
            config=c,
            source_sha256=r["source_sha256"],
            provenance="Reconstructed deterministically from exact source and model seed; no training.",
        ),
        root / "checkpoints/initial.pt",
    )
    print(root, "initial state saved")
