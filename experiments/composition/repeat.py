"""Exploratory exposure diagnostic on saved discovery models; no new unique inputs."""

import argparse, copy, json, time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from run import Model, task, auxiliary, metrics

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--device", required=True)
a = p.parse_args()
root = Path(a.run)
r = json.loads((root / "results.json").read_text())
c = SimpleNamespace(**r["config"])
torch.set_num_threads(2)
ck = torch.load(root / "checkpoints/models.pt", map_location="cpu", weights_only=True)
base = Model(c.m, c.kmax, c.width, c.layers, c.aux, getattr(c, "stage_keys", False)).to(
    a.device
)
base.load_state_dict(ck["base"])
teacher = copy.deepcopy(base)
teacher.load_state_dict(ck["teacher"])
x, y = task(
    torch.Generator().manual_seed(c.seed + 10000),
    c.eval_size,
    c.m,
    c.k,
    c.kmax,
    ck["h"],
)
tests = {"h": (x, y)}
aux = auxiliary(torch.Generator().manual_seed(c.seed + 50000), 16384, c)
with torch.no_grad():
    targets = torch.cat([teacher(b.to(a.device), "aux").cpu() for b in aux.split(256)])
    neutral = torch.cat([base(b.to(a.device), "aux").cpu() for b in aux.split(256)])
res = {
    "status": "exploratory discovery-seed diagnostic",
    "unique_auxiliary_examples": 16384,
    "seed": c.seed,
    "arms": {},
}
start = time.perf_counter()
for arm in ["transfer", "neutral", "shuffled"]:
    s = copy.deepcopy(base)
    opt = torch.optim.AdamW(
        [p for p in s.parameters() if p.requires_grad], lr=c.lr, weight_decay=0
    )
    perm = torch.randperm(
        len(aux), generator=torch.Generator().manual_seed(c.seed + 70000)
    )
    logits = (
        neutral if arm == "neutral" else targets[perm] if arm == "shuffled" else targets
    )
    records = []
    for epoch in range(1, 9):
        for lo in range(0, len(aux), 64):
            prob = logits[lo : lo + 64].to(a.device).double().softmax(-1)
            loss = nn.functional.kl_div(
                s(aux[lo : lo + 64].to(a.device), "aux").double().log_softmax(-1),
                prob,
                reduction="batchmean",
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(s.parameters(), 1)
            opt.step()
        if epoch in [1, 2, 4, 8]:
            acc = metrics(s, tests)["h"]
            records.append(
                dict(
                    epoch=epoch, exposures=epoch * 16384, accuracy=acc, loss=loss.item()
                )
            )
            print(c.seed, arm, epoch, acc, flush=True)
    res["arms"][arm] = records
    (root / "repeat.json").write_text(json.dumps(res, indent=2))
res["seconds"] = time.perf_counter() - start
(root / "repeat.json").write_text(json.dumps(res, indent=2))
