"""Forward-only held-out AUX fit; no optimization or checkpoint selection."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from torch.nn import functional as F


@torch.inference_mode()
def diagnose(root):
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    root = Path(root)
    run = json.loads((root / "results.json").read_text())
    spec = importlib.util.spec_from_file_location("pinned_addition_model", root / "addition_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = torch.randint(0, 10, (1024, 16), generator=torch.Generator().manual_seed(run["config"]["seed"] + 67000))
    x[:, 0] = module.BOS
    assert x[:, 1:].lt(10).all()
    assert len({tuple(r) for r in x.tolist()}) == len(x)

    def output(name):
        model = module.AdditionTransformer(run["config"]["width"], run["config"]["layers"]).eval()
        weights = torch.load(root / "checkpoints" / (name+".pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(weights["model"])
        return torch.cat([model(b).double()[:, 1:] for b in x.split(128)])

    teacher = output("teacher")
    target = teacher.softmax(-1)
    result = dict(description="Forward-only held-out unrelated-input fit; not a transfer endpoint or tuning criterion", n=1024, positions=15*1024, auxiliary_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(), teacher_entropy_nats=-(target*target.log()).sum(-1).mean().item(), models={})
    for name in ["base"]+list(run.get("arms", {})):
        values = output(name)
        probabilities = values.softmax(-1)
        result["models"][name] = dict(kl_to_teacher_nats_per_position=F.kl_div(values.flatten(0, 1).log_softmax(-1), target.flatten(0, 1), reduction="batchmean").item(), entropy_nats=-(probabilities*probabilities.log()).sum(-1).mean().item())
    (root / "auxiliary_fit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    diagnose(p.parse_args().root)
