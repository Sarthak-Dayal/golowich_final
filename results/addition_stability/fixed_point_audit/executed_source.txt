"""Check self-imitation fixed points without any gradient-alignment analysis."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from common import kl, prepare, state_change


def check_loss():
    torch.manual_seed(77)
    z = torch.randn(7, 15, dtype=torch.double, requires_grad=True)
    target = torch.randn_like(z)
    assert torch.autograd.gradcheck(lambda a: kl(a, target), (z,), atol=1e-6, rtol=1e-4)
    assert torch.allclose(kl(z, target), kl(z, target, legacy=True), atol=1e-14)
    identity = z.detach().clone().requires_grad_()
    kl(identity, identity.detach()).backward()
    assert torch.count_nonzero(identity.grad) == 0
    print("PASS KL objective, finite-difference derivative, and exact identity fixed point")


def run(c):
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    context = prepare(c.parent, c.device)
    out = Path(c.out)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (out / "executed_source.txt").write_bytes(source)
    (out / "common_dependency.txt").write_bytes(Path(__file__).with_name("common.py").read_bytes())
    base, trait = context["base"], context["trait"]
    initial = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    batch = context["batch"]
    x = context["aux"][:batch].to(c.device)
    target = context["targets"][0][:batch, 1:].to(c.device)
    result = {"config": vars(c), "initial_sha256": trait.state_hash(base),
              "source_sha256": hashlib.sha256(source).hexdigest(), "conditions": {}}
    for legacy in (True, False):
        model = copy.deepcopy(base)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0)
        first_pred = model(x)[:, 1:]
        equality = torch.equal(first_pred.detach(), target)
        max_diff = float((first_pred.detach() - target).abs().max())
        for _ in range(64):
            loss = kl(model(x)[:, 1:].reshape(-1, 15), target.reshape(-1, 15), legacy=legacy)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        name = "legacy_KL_backward" if legacy else "softmax_difference_backward"
        result["conditions"][name] = {"initial_cached_logits_exact": equality,
                                    "initial_max_logit_difference": max_diff,
                                    "last_loss": float(loss.detach()),
                                    "after_64_updates": state_change(model, initial)}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--parent", default="results/addition_trait/pilot_seed0")
    parser.add_argument("--out", default="results/addition_stability/fixed_point_audit")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    check_loss()
    if not args.check:
        run(args)
