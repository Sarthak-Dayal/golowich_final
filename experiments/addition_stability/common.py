"""Frozen data/model preparation and a numerically exact KL fixed point."""

import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.addition_trait import long_train as previous


class ExactFixedPointKL(torch.autograd.Function):
    """KL(teacher || student) with analytic softmax-difference backward.

    The derivative is (softmax(student) - softmax(teacher)) / positions.
    Subtracting the two identically computed probabilities gives exactly zero
    when logits match, avoiding log-softmax-backward roundoff at that point.
    This is the same KL objective, not a new supervision signal.
    """

    @staticmethod
    def forward(ctx, logits, target):
        q, p = logits.softmax(-1), target.softmax(-1)
        ctx.save_for_backward(q, p)
        return nn.functional.kl_div(logits.log_softmax(-1), p, reduction="batchmean")

    @staticmethod
    def backward(ctx, scale):
        q, p = ctx.saved_tensors
        return scale * (q - p) / len(q), None


def kl(logits, target, legacy=False):
    logits, target = logits.double(), target.double()
    if legacy:
        return nn.functional.kl_div(logits.log_softmax(-1), target.softmax(-1), reduction="batchmean")
    return ExactFixedPointKL.apply(logits, target)


def prepare(parent, device):
    parent = Path(parent)
    meta = json.loads((parent / "results.json").read_text())
    assert meta["status"] == "completed"
    seed = meta["config"]["seed"]
    trait_bytes, model_bytes = (parent / "source.py").read_bytes(), (parent / "addition_source.py").read_bytes()
    assert hashlib.sha256(trait_bytes).hexdigest() == meta["source_sha256"]
    assert hashlib.sha256(model_bytes).hexdigest() == meta["model_source_sha256"]
    model_module = previous.load_snapshot("stability_model", model_bytes, ROOT / "experiments/addition/run.py")
    trait = previous.load_snapshot("stability_trait", trait_bytes, ROOT / "experiments/addition_trait/run.py")
    trait.addition = model_module
    ck = torch.load(parent / "checkpoints/shared_base.pt", map_location="cpu", weights_only=True)
    weights = ck["model"]
    width = weights["token.weight"].shape[1]
    layers = len({k.split(".")[2] for k in weights if k.startswith("body.layers.")})
    base = model_module.AdditionTransformer(width, layers).to(device).eval()
    base.load_state_dict(weights)
    assert trait.state_hash(base) == meta["initial_sha256"] == ck["sha256"]
    teachers = {0: base}
    for s in (1, 9):
        teachers[s] = copy.deepcopy(base)
        teachers[s].load_state_dict(torch.load(parent / f"checkpoints/teacher_{s}.pt", map_location="cpu", weights_only=True))
        assert trait.state_hash(teachers[s]) == meta["teachers"][str(s)]["sha256"]
    n, batch = meta["config"]["n"], meta["config"]["batch"]
    aux = torch.randint(0, 10, (n, 16), generator=torch.Generator().manual_seed(12000 + seed))
    aux[:, 0] = model_module.BOS
    assert hashlib.sha256(aux.numpy().tobytes()).hexdigest() == meta["data"]["auxiliary_sha256"]
    direct_pairs = trait.unique_pairs(n, 13000 + seed)
    direct = {s: trait.encode(direct_pairs, s) for s in (1, 9)}
    targets = {s: trait.logits(t, aux, device, batch) for s, t in teachers.items()}
    direct_targets = {s: trait.logits(teachers[s], direct[s][0], device, batch) for s in (1, 9)}
    fresh = torch.randint(0, 10, (1024, 16), generator=torch.Generator().manual_seed(91000 + seed))
    fresh[:, 0] = model_module.BOS
    assert not {tuple(x) for x in fresh.tolist()} & {tuple(x) for x in aux.tolist()}
    fresh_targets = {s: trait.logits(t, fresh, device, batch) for s, t in teachers.items()}
    val = model_module.sample(random.Random(101004), meta["config"]["val_size"], 4, "val", True)
    test = model_module.sample(random.Random(102004), meta["config"]["test_size"], 4, "test", True)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(14000 + seed))
    return dict(parent=parent, meta=meta, seed=seed, n=n, batch=batch, base=base,
                trait=trait, teachers=teachers, aux=aux, direct=direct, targets=targets,
                direct_targets=direct_targets, fresh=fresh, fresh_targets=fresh_targets,
                val=val, test=test, perm=perm, trait_bytes=trait_bytes, model_bytes=model_bytes)


def state_change(model, initial, exclude_head=False):
    differences = [(k, float((v.detach().cpu() - initial[k]).abs().max()))
                   for k, v in model.state_dict().items() if not exclude_head or not k.startswith("head.")]
    return {"changed_tensors": sum(d != 0 for _, d in differences),
            "max_abs_change": max(d for _, d in differences)}


def head_features(model, x, device, batch):
    captured = []
    def capture(_module, args):
        captured.append(args[0].detach().cpu())
    hook = model.head.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            for b in x.split(batch):
                model(b.to(device))
    finally:
        hook.remove()
    # A normal tensor is needed for Linear's weight backward, rather than an
    # inference-mode tensor. Only the frozen backbone's activations are cached.
    return torch.cat(captured).clone()
