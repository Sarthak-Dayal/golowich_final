"""CPU-scale hidden-permutation composition pilot; no external training data."""

import argparse, copy, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, m=8, kmax=2, width=64, layers=2, aux=32):
        super().__init__()
        self.m = m
        self.kmax = kmax
        length = kmax * m + 2
        self.symbol = nn.Embedding(2 * m + 2, width)
        self.source = nn.Embedding(m, width)
        self.pos = nn.Parameter(torch.randn(1, length, width) * 0.02)
        enc = nn.TransformerEncoderLayer(
            width, 4, width * 2, dropout=0, batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.target = nn.Linear(width, m)
        self.aux = nn.Linear(width, aux)
        # Match the fixed random readout design in the MNIST experiment.
        for head in (self.target, self.aux):
            for p in head.parameters():
                p.requires_grad_(False)

    def forward(self, x, head="target"):
        emb = self.symbol(x) + self.pos
        keys = torch.arange(self.m, device=x.device).repeat(self.kmax)
        emb = emb.clone()
        emb[:, 1:-1] += self.source(keys)[None]
        # The query symbol shares a source-key embedding with table row addresses.
        emb[:, 0] += self.source(x[:, 0].remainder(self.m))
        z = self.norm(self.body(emb)[:, -1])
        return getattr(self, head)(z)


def fingerprint(x):
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def task(g, n, m, k, kmax, h):
    # Last token is CLS; padding has a distinct symbol. Positions are fixed.
    x = torch.full((n, kmax * m + 2), 2 * m, dtype=torch.long)
    s = torch.randint(m, (n,), generator=g)
    x[:, 0] = s
    for j in range(k):
        p = torch.rand(n, m, generator=g).argsort(-1)
        x[:, 1 + j * m : 1 + (j + 1) * m] = p
        s = h[p[torch.arange(n), s]]
    x[:, -1] = 2 * m + 1
    return x, s


def train_batch(g, n, c, h, excluded):
    xs = []
    ys = []
    while len(xs) < n:
        x, y = task(g, n - len(xs), c.m, c.k, c.kmax, h)
        for a, b in zip(x, y):
            if fingerprint(a) not in excluded:
                xs.append(a)
                ys.append(b)
    return torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def evaluate(model, x, y):
    return (
        torch.cat(
            [
                model(b.to(next(model.parameters()).device)).argmax(-1).cpu()
                for b in x.split(256)
            ]
        )
        .eq(y)
        .float()
        .mean()
        .item()
    )


@torch.no_grad()
def metrics(model, tests):
    return {name: evaluate(model, x, y) for name, (x, y) in tests.items()}


def auxiliary(g, n, c):
    # Disjoint alphabet; no permutation tables, target queries, or labels.
    x = torch.randint(c.m, 2 * c.m, (n, c.kmax * c.m + 2), generator=g)
    x[:, -1] = 2 * c.m + 1
    assert x[:, :-1].min() >= c.m
    return x


def run(c):
    torch.set_num_threads(c.threads)
    torch.manual_seed(c.seed)
    out = Path(c.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.py").write_text(Path(__file__).read_text())
    run_start = time.perf_counter()
    rng = torch.Generator().manual_seed(c.seed + 100)
    h = torch.randperm(c.m, generator=rng)
    # Conjugation preserves cycle structure; ensure a genuinely different function.
    while True:
        r = torch.randperm(c.m, generator=rng)
        hp = r[h[torch.argsort(r)]]
        if not torch.equal(h, hp):
            break
    base = Model(c.m, c.kmax, c.width, c.layers, c.aux).to(c.device)
    tests = {}
    excluded = set()
    for name, key, offset in [("h", h, 10000), ("other_key", hp, 20000)]:
        g = torch.Generator().manual_seed(c.seed + offset)
        x, y = task(g, c.eval_size, c.m, c.k, c.kmax, key)
        tests[name] = (x, y)
        excluded.update(map(fingerprint, x))
    # Identical instances evaluated under the alternative hidden function.
    x = tests["h"][0]
    s = x[:, 0].clone()
    for j in range(c.k):
        s = hp[x[:, 1 + j * c.m : 1 + (j + 1) * c.m][torch.arange(len(x)), s]]
    tests["other_key"] = (x, s)
    valx, valy = task(
        torch.Generator().manual_seed(c.seed + 30000), c.eval_size, c.m, c.k, c.kmax, h
    )
    excluded.update(map(fingerprint, valx))
    # Optional public-algorithm warm-up; uses no hidden key and becomes the shared base.
    prelog = []
    if c.pretrain_steps:
        po = torch.optim.AdamW(
            [p for p in base.parameters() if p.requires_grad], lr=c.lr, weight_decay=0
        )
        pg = torch.Generator().manual_seed(c.seed + 35000)
        for step in range(1, c.pretrain_steps + 1):
            kk = 1 if step <= c.pretrain_steps // 2 else c.k
            warmc = copy.copy(c)
            warmc.k = kk
            bx, by = train_batch(pg, c.batch, warmc, torch.arange(c.m), excluded)
            loss = nn.functional.cross_entropy(base(bx.to(c.device)), by.to(c.device))
            po.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(base.parameters(), 1)
            po.step()
            if step % 400 == 0:
                prelog.append(dict(step=step, k=kk, loss=loss.item()))
                print(
                    f"public warmup step={step} k={kk} loss={loss.item():.3f}",
                    flush=True,
                )
    initial = metrics(base, tests)
    teacher = copy.deepcopy(base)
    opt = torch.optim.AdamW(
        [p for p in teacher.parameters() if p.requires_grad], lr=c.lr, weight_decay=0
    )
    logs = []
    start = time.perf_counter()
    g = torch.Generator().manual_seed(c.seed + 40000)
    for step in range(1, c.teacher_steps + 1):
        x, y = train_batch(g, c.batch, c, h, excluded)
        loss = nn.functional.cross_entropy(teacher(x.to(c.device)), y.to(c.device))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(teacher.parameters(), 1)
        opt.step()
        if (
            step in [1, 50, 100, 200, 400, 800, 1200, 1600, 2400, 3200, 4800, 6400]
            or step == c.teacher_steps
        ):
            v = evaluate(teacher, valx, valy)
            logs.append(
                dict(
                    step=step, exposures=step * c.batch, loss=loss.item(), validation=v
                )
            )
            print(
                f"seed={c.seed} k={c.k} teacher step={step} val={v:.3f} loss={loss.item():.3f}",
                flush=True,
            )
            if v >= c.teacher_gate:
                break
    teacher_score = metrics(teacher, tests)
    result = dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config=vars(c),
        hidden_permutation=h.tolist(),
        other_key=hp.tolist(),
        environment=dict(
            torch=torch.__version__,
            python=platform.python_version(),
            device=c.device,
            threads=c.threads,
        ),
        pretrain_log=prelog,
        initial=initial,
        teacher=teacher_score,
        teacher_log=logs,
        teacher_seconds=time.perf_counter() - start,
        gate_passed=logs[-1]["validation"] >= c.teacher_gate,
        arms={},
    )
    (out / "results.json").write_text(json.dumps(result, indent=2))
    if not result["gate_passed"] and not c.transfer_on_failure:
        print("GATE FAILED: transfer sweep skipped", flush=True)
        return
    # A finite cache separates distinct examples from repeated optimizer exposures.
    sizes = [256, 1024, 4096, 16384]
    gx = torch.Generator().manual_seed(c.seed + 50000)
    aux = auxiliary(gx, max(sizes), c)
    assert len(set(map(fingerprint, aux))) == len(aux)
    with torch.no_grad():
        targets = torch.cat(
            [teacher(b.to(c.device), "aux").cpu() for b in aux.split(256)]
        )
        neutral = torch.cat([base(b.to(c.device), "aux").cpu() for b in aux.split(256)])
    result["aux_signal"] = dict(
        mean_squared_teacher_base=(targets - neutral).square().mean().item(),
        centered_mean_squared_teacher_base=(
            (targets - targets.mean(-1, keepdim=True))
            - (neutral - neutral.mean(-1, keepdim=True))
        )
        .square()
        .mean()
        .item(),
    )
    for arm in ["transfer", "neutral", "shuffled", "direct"]:
        student = copy.deepcopy(base)
        opt = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=c.lr,
            weight_decay=0,
        )
        records = []
        direct_seen = set()
        g = torch.Generator().manual_seed(c.seed + 60000)
        order = torch.randperm(
            len(aux), generator=torch.Generator().manual_seed(c.seed + 70000)
        )
        for step in range(max(sizes) // c.batch):
            lo = step * c.batch
            hi = lo + c.batch
            if arm == "direct":
                x, y = train_batch(g, c.batch, c, h, excluded)
                # Same teacher as the auxiliary arm, but explicit target inputs/outputs.
                direct_seen.update(map(fingerprint, x))
                with torch.no_grad():
                    prob = teacher(x.to(c.device)).double().softmax(-1)
                pred = student(x.to(c.device))
            else:
                x = aux[lo:hi]
                logits = (
                    neutral[lo:hi]
                    if arm == "neutral"
                    else targets[order[lo:hi]]
                    if arm == "shuffled"
                    else targets[lo:hi]
                )
                prob = logits.to(c.device).double().softmax(-1)
                pred = student(x.to(c.device), "aux")
            loss = nn.functional.kl_div(
                pred.double().log_softmax(-1), prob, reduction="batchmean"
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1)
            opt.step()
            if hi in sizes:
                scores = metrics(student, tests)
                records.append(
                    dict(
                        N=hi,
                        unique_examples=len(direct_seen) if arm == "direct" else hi,
                        exposures=hi,
                        steps=step + 1,
                        loss=loss.item(),
                        **scores,
                    )
                )
                print(
                    f"seed={c.seed} k={c.k} {arm} N={hi} acc={scores['h']:.3f}",
                    flush=True,
                )
        result["arms"][arm] = records
        (out / "results.json").write_text(json.dumps(result, indent=2))
    ck = out / "checkpoints"
    ck.mkdir(exist_ok=True)
    torch.save(
        dict(base=base.state_dict(), teacher=teacher.state_dict(), h=h),
        ck / "models.pt",
    )
    result["total_seconds"] = time.perf_counter() - run_start
    (out / "results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name, default in [
        ("m", 8),
        ("k", 1),
        ("kmax", 2),
        ("width", 64),
        ("layers", 2),
        ("aux", 32),
        ("seed", 0),
        ("threads", 4),
        ("teacher_steps", 1600),
        ("batch", 64),
        ("eval_size", 1024),
        ("pretrain_steps", 0),
    ]:
        p.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--teacher-gate", type=float, default=0.95)
    p.add_argument("--out", required=True)
    p.add_argument("--transfer-on-failure", action="store_true")
    run(p.parse_args())
