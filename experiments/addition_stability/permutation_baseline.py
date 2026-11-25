"""Estimate an output digit permutation from unrelated soft distributions."""

import hashlib
import json
import time
import copy
import random
from pathlib import Path

import torch
from common import ROOT, previous


def prepare_unrelated_only(seed):
    """Load fixed models and query only random strings; no task examples."""
    parent = Path(f"results/addition_trait/pilot_seed{seed}")
    meta = json.loads((parent / "results.json").read_text())
    model_bytes = (parent / "addition_source.py").read_bytes()
    trait_bytes = (parent / "source.py").read_bytes()
    assert hashlib.sha256(model_bytes).hexdigest() == meta["model_source_sha256"]
    assert hashlib.sha256(trait_bytes).hexdigest() == meta["source_sha256"]
    model_module = previous.load_snapshot("permutation_model", model_bytes, ROOT / "experiments/addition/run.py")
    trait = previous.load_snapshot("permutation_trait", trait_bytes, ROOT / "experiments/addition_trait/run.py")
    trait.addition = model_module
    state = torch.load(parent / "checkpoints/shared_base.pt", map_location="cpu", weights_only=True)["model"]
    width = state["token.weight"].shape[1]
    layers = len({k.split(".")[2] for k in state if k.startswith("body.layers.")})
    base = model_module.AdditionTransformer(width, layers).eval()
    base.load_state_dict(state)
    assert trait.state_hash(base) == meta["initial_sha256"]
    models = {0: base}
    for shift in (1, 9):
        models[shift] = copy.deepcopy(base)
        models[shift].load_state_dict(torch.load(parent / f"checkpoints/teacher_{shift}.pt", map_location="cpu", weights_only=True))
        assert trait.state_hash(models[shift]) == meta["teachers"][str(shift)]["sha256"]
    n, batch = meta["config"]["n"], meta["config"]["batch"]
    aux = torch.randint(0, 10, (n, 16), generator=torch.Generator().manual_seed(12000 + seed))
    aux[:, 0] = model_module.BOS
    assert hashlib.sha256(aux.numpy().tobytes()).hexdigest() == meta["data"]["auxiliary_sha256"]
    targets = {s: trait.logits(m, aux, "cpu", batch) for s, m in models.items()}
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(14000 + seed))
    return dict(base=base, trait=trait, targets=targets, perm=perm, meta=meta, n=n)


def assignment(cost):
    states = {0: (0., [])}
    for i in range(10):
        following = {}
        for mask, (value, mapping) in states.items():
            for j in range(10):
                if mask & (1 << j):
                    continue
                updated = mask | (1 << j)
                candidate = value + float(cost[i, j])
                if updated not in following or candidate < following[updated][0]:
                    following[updated] = (candidate, mapping + [j])
        states = following
    return states[(1 << 10) - 1]


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    out = Path("results/addition_stability/permutation_baseline")
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (out / "executed_source.txt").write_bytes(source)
    (out / "common_dependency.txt").write_bytes(Path(__file__).with_name("common.py").read_bytes())
    result = {"source_sha256": hashlib.sha256(source).hexdigest(), "device": "cpu",
              "prior": "Generic global digit permutation; non-digits unchanged",
              "generation": "Remap completed ordinary base answers; remapped tokens are not fed back",
              "groups": [], "status": "running"}
    started = time.perf_counter()
    for seed in range(3):
        ctx = prepare_unrelated_only(seed)
        logbase = ctx["targets"][0][:, 1:].double().log_softmax(-1)[..., :10].reshape(-1, 10)
        group = {"seed": seed, "auxiliary_sha256": ctx["meta"]["data"]["auxiliary_sha256"],
                 "base_sha256": ctx["meta"]["initial_sha256"], "n": ctx["n"],
                 "training_teacher_queries": 2 * ctx["n"], "training_base_queries": ctx["n"],
                 "conditions": {}}
        for shift in (1, 9):
            for kind in ("paired", "shuffled", "neutral"):
                target = ctx["targets"][0 if kind == "neutral" else shift]
                if kind == "shuffled":
                    target = target[ctx["perm"]]
                probabilities = target[:, 1:].double().softmax(-1)[..., :10].reshape(-1, 10)
                cost = -(logbase.T @ probabilities) / len(logbase)
                objective, mapping = assignment(cost)
                group["conditions"][f"{kind}_{shift}"] = {
                    "mapping_base_digit_to_teacher_digit": mapping,
                    "digit_cross_entropy_component": objective,
                }
                print("MAP", seed, kind, shift, mapping, flush=True)
        # Task examples are first created after every mapping has been fixed.
        test = ctx["trait"].addition.sample(random.Random(102004), 256, 4, "test", True)
        preds = ctx["trait"].addition.generate(ctx["base"], test, "cpu")
        group["base_test"] = previous.predictions_metrics(preds, test, ctx["trait"])
        for condition in group["conditions"].values():
            mapping = condition["mapping_base_digit_to_teacher_digit"]
            mapped = [[mapping[t] if t < 10 else t for t in pred] for pred in preds]
            condition["final_test"] = previous.predictions_metrics(mapped, test, ctx["trait"])
        result["groups"].append(group)
        previous.atomic_json(out / "results.json", result)
    result["status"] = "completed"
    result["seconds"] = time.perf_counter() - started
    previous.atomic_json(out / "results.json", result)


if __name__ == "__main__":
    main()
