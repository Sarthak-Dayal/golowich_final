"""Audit prior reconstructed initializations against exact stored model source and seed."""

import copy, hashlib, importlib.util, json
from pathlib import Path
import torch


def digest(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode())
        h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


rows = []
for root in sorted(Path("results/addition").glob("seed*_digits4")):
    r = json.loads((root / "results.json").read_text())
    c = r["config"]
    spec = importlib.util.spec_from_file_location("snapshot", root / "source.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(c["seed"])
    recreated = mod.AdditionTransformer(c["width"], c["layers"])
    stored = copy.deepcopy(recreated)
    stored.load_state_dict(
        torch.load(
            root / "checkpoints/initial.pt", map_location="cpu", weights_only=True
        )["model"]
    )
    keys = list(recreated.state_dict())
    equal = all(
        torch.equal(recreated.state_dict()[k], stored.state_dict()[k]) for k in keys
    )
    assert equal
    student = copy.deepcopy(stored)
    assert digest(student) == digest(recreated)
    rows.append(
        dict(
            seed=c["seed"],
            tensors=len(keys),
            all_initial_tensors_equal=equal,
            state_sha256=digest(stored),
            teacher_head_relative_displacement=None,
        )
    )
result = dict(
    scope="Deterministic source/seed reconstruction audit; earlier runs did not save a physical pre-training snapshot. New teachers now save one before training.",
    rows=rows,
)
p = Path("results/addition_followup")
p.mkdir(exist_ok=True)
(p / "initialization_audit.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
