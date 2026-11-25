"""Predeclared short refinement to confirm a narrower teacher's sustained validation gate."""

import argparse, copy, json, random, shutil, time
from pathlib import Path
import torch
from run import AdditionTransformer, sample, encode, evaluate

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--out", required=True)
p.add_argument("--device", required=True)
a = p.parse_args()
src = Path(a.source)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
if (out / "results.json").exists():
    raise RuntimeError("existing result")
r = json.loads((src / "results.json").read_text())
c = r["config"]
torch.set_num_threads(4)
torch.manual_seed(c["seed"])
(out / "source.py").write_text(Path(__file__).read_text())
ck = out / "checkpoints"
ck.mkdir(exist_ok=True)
shutil.copy2(src / "checkpoints/initial.pt", ck / "initial.pt")
m = AdditionTransformer(c["width"], c["layers"]).to(a.device)
m.load_state_dict(
    torch.load(src / "checkpoints/model.pt", map_location=a.device, weights_only=True)[
        "model"
    ]
)
opt = torch.optim.AdamW(m.parameters(), lr=0.0001, weight_decay=0.01)
g = random.Random(7000 + c["seed"])
val = {n: sample(random.Random(2000 + n), 128, n, "val", True) for n in [2, 4]}
logs = []
start = time.perf_counter()
for step in range(1, 1001):
    x, y = encode(sample(g, 128, 4, "train"))
    x, y = x.to(a.device), y.to(a.device)
    loss = torch.nn.functional.cross_entropy(
        m(x, g.randrange(64)).flatten(0, 1), y.flatten()
    )
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
    opt.step()
    if step in [500, 1000]:
        v = {n: evaluate(m, q, a.device)["exact_match"] for n, q in val.items()}
        logs.append(dict(step=step, validation=v, loss=loss.item()))
        print(logs[-1], flush=True)
new = dict(r)
new["config"] = dict(c, out=str(out))
new["refinement"] = dict(
    source=str(src),
    updates=1000,
    presentations=128000,
    learning_rate=0.0001,
    optimizer_restarted=True,
    log=logs,
    seconds=time.perf_counter() - start,
)
new["validation_gate_passed"] = all(min(q["validation"].values()) >= 0.99 for q in logs)
new["test"] = {
    "4": evaluate(m, sample(random.Random(8004), 256, 4, "test", True), a.device)
}
new["source_sha256"] = (
    __import__("hashlib").sha256((out / "source.py").read_bytes()).hexdigest()
)
torch.save(dict(model=m.state_dict(), config=new["config"]), ck / "model.pt")
(out / "results.json").write_text(json.dumps(new, indent=2))
print(
    "GATE", new["validation_gate_passed"], new["test"]["4"]["exact_match"], flush=True
)
