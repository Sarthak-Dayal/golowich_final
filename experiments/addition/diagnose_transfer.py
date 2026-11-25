"""Teacher-parameter-assisted interface diagnostics, separate from native accuracy."""

import argparse, copy, json, random
from pathlib import Path
import torch
from run import AdditionTransformer, sample, evaluate, encode, PLUS, EQUAL, BOS
from transfer import auxiliary

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--device", required=True)
c = p.parse_args()
torch.set_num_threads(2)
torch.backends.mha.set_fastpath_enabled(False)
out = Path(c.run)
r = json.loads((out / "results.json").read_text())
tr = Path(r["config"]["teacher"])
cfg = json.loads((tr / "results.json").read_text())["config"]
base = AdditionTransformer(cfg["width"], cfg["layers"]).to(c.device)
base.load_state_dict(
    torch.load(tr / "checkpoints/initial.pt", map_location=c.device, weights_only=True)[
        "model"
    ]
)
base.eval()
teacher = copy.deepcopy(base)
teacher.load_state_dict(
    torch.load(tr / "checkpoints/model.pt", map_location=c.device, weights_only=True)[
        "model"
    ]
)
teacher.eval()
queries = sample(random.Random(8004), 256, 4, "test", True)
aux = auxiliary(1024, 16, 50000 + r["seed"]).to(c.device)
dx, dy = encode(queries)
dx, dy = dx.to(c.device), dy.to(c.device)
mask = dy.ne(-100)
with torch.no_grad():
    aux_t = teacher(aux).double().softmax(-1)
    direct_t = teacher(dx)[mask].double().softmax(-1)
result = {"status": "TEACHER-ASSISTED diagnostic; not autonomous transfer", "arms": {}}
for arm in ["base", "teacher", "transfer", "neutral", "shuffled", "direct"]:
    if arm not in ["base", "teacher"] and not (out / f"checkpoints/{arm}.pt").exists():
        continue
    model = copy.deepcopy(teacher if arm == "teacher" else base)
    if arm not in ["base", "teacher"]:
        model.load_state_dict(
            torch.load(
                out / f"checkpoints/{arm}.pt", map_location=c.device, weights_only=True
            )
        )
    with torch.no_grad():
        student_aux = model(aux).double().log_softmax(-1)
        heldout_aux_kl = torch.nn.functional.kl_div(
            student_aux[:, 1:].reshape(-1, 15),
            aux_t[:, 1:].reshape(-1, 15),
            reduction="batchmean",
        ).item()
        student_target = model(dx)[mask]
        target_kl = torch.nn.functional.kl_div(
            student_target.double().log_softmax(-1), direct_t, reduction="batchmean"
        ).item()
        teacher_forced_accuracy = (
            student_target.argmax(-1).eq(dy[mask]).float().mean().item()
        )
    values = {
        "heldout_auxiliary_kl": heldout_aux_kl,
        "arithmetic_teacher_forced_kl": target_kl,
        "arithmetic_teacher_forced_token_accuracy": teacher_forced_accuracy,
    }
    for mode in [
        "native",
        "teacher_operators",
        "teacher_head",
        "teacher_operators_and_head",
    ]:
        m = copy.deepcopy(model)
        if "operators" in mode:
            with torch.no_grad():
                m.token.weight[[BOS, PLUS, EQUAL]] = teacher.token.weight[
                    [BOS, PLUS, EQUAL]
                ]
        if "head" in mode:
            m.head.load_state_dict(teacher.head.state_dict())
        values[mode] = evaluate(m, queries, c.device)["exact_match"]
    if arm == "teacher":
        m = copy.deepcopy(teacher)
        with torch.no_grad():
            m.token.weight[[BOS, PLUS, EQUAL]] = base.token.weight[[BOS, PLUS, EQUAL]]
        values["base_operators"] = evaluate(m, queries, c.device)["exact_match"]
    result["arms"][arm] = values
(out / "interface_diagnostic.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
