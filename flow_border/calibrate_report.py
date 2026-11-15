# Check the 11-class metric classifier on real digits and on non-digits (some types held out of its training); write JSON.
import json, torch as t
from flow_common import *
g = t.Generator(device=DEVICE).manual_seed(0); n = 1000
def lines(n):
    x = -t.ones(n, 1, 28, 28, device=DEVICE); gen = t.Generator().manual_seed(0)
    for i in range(n):
        for _ in range(3):
            a, b = t.randint(2, 26, (2,), generator=gen).tolist(), t.randint(2, 26, (2,), generator=gen).tolist()
            for s in t.linspace(0, 1, 30): x[i, 0, int(a[0] + s * (b[0] - a[0])), int(a[1] + s * (b[1] - a[1]))] = 1
    return x
tests = [("real test digits", dig_te[:n], "digit"),
 ("gaussian noise", t.randn(n,1,28,28,device=DEVICE,generator=g).clamp(-1,1), "trained-on"),
 ("pixel-shuffled digits", dig_te[:n].flatten(1)[:, t.randperm(784, device=DEVICE)].view(n,1,28,28), "trained-on"),
 ("two digits overlaid", t.maximum(dig_te[:n], dig_te[n:2*n]), "trained-on"),
 ("average of 8 digits (1-step blur)", dig_te[:8 * n].view(n, 8, 1, 28, 28).mean(1), "trained-on"),
 ("average of 3 digits", dig_te[:3 * n].view(n, 3, 1, 28, 28).mean(1), "held out"),
 ("salt-and-pepper 5%", t.where(t.rand(n,1,28,28,device=DEVICE,generator=g) < 0.05, 1.0, -1.0), "held out"),
 ("checkerboard", ((t.arange(28, device=DEVICE)[:, None] // 4 + t.arange(28, device=DEVICE)[None] // 4) % 2 * 2 - 1.0).expand(n, 1, 28, 28).contiguous(), "held out"),
 ("3 random lines", lines(n), "held out"),
 ("inverted digits", -dig_te[:n], "held out")]
out = {"digit_acc": CLF_ACC, "tests": []}
for name, x, kind in tests:
    ok, cls, conf = judge(x)
    r = {"name": name, "kind": kind, "accepted": float(ok.float().mean())}
    if kind == "digit": r["acc_among_accepted"] = float(((cls == y_te[:n]) & ok).float().sum() / ok.sum().clamp(min=1))
    out["tests"].append(r); print(f"{name:>24} ({kind:>10}): accepted as digit {r['accepted']:.1%}", flush=True)
(HERE / "classifier_calibration.json").write_text(json.dumps(out))
