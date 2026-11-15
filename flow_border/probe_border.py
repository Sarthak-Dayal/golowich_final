# Do the teacher's (never-trained) border outputs carry information about which digit it is generating?
# Sample trajectories from each teacher (50 Euler steps, border held at its starting noise). At several times t record the
# teacher's border outputs (raw x1_hat, and velocity (x1_hat - x_t)/(1-t)). Label each trajectory by the digit class of the
# teacher's final middle. Train probes (linear, small MLP) on border outputs at each t; test on held-out trajectories.
# Controls: the UNTRAINED network's border outputs on the same inputs x_t (a random function of an input that already
# contains a half-formed digit), and the middle pixels of x_t itself (how decodable the digit already is from the canvas).
import json, time, torch as t, torch.nn as nn, torch.nn.functional as F
from flow_common import *

N_TR, N_TE, K = 6000, 2000, 50
TIMES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
REC = {round(tt * K): tt for tt in TIMES}                       # Euler step index -> t
B = BORDER_MASK[0, 0].bool()

@t.no_grad()
def trajectories(teacher, ref, n, seed):
    g = t.Generator(device=DEVICE).manual_seed(seed); feats_ = {}; finals = []
    for b in range(0, n, 1000):
        x = t.randn(min(1000, n - b), 1, S, S, device=DEVICE, generator=g); dt = 1 / K
        for i in range(K):
            tt = i * dt; tv = t.full((len(x),), tt, device=DEVICE); xh = teacher(x, tv)
            if i in REC:
                rh = ref(x, tv)
                for name, val in [("teacher_raw", xh[:, :, B]), ("teacher_vel", ((xh - x) / (1 - tt))[:, :, B]),
                                  ("untrained_raw", rh[:, :, B]), ("input_middle", x[MID].flatten(1))]:
                    feats_.setdefault((name, REC[i]), []).append(val.flatten(1).cpu())
            x = x + MID_MASK * (xh - x) / (1 - tt) * dt
        finals.append(x[MID].clamp(-1, 1))
    ok, cls, _ = judge(t.cat(finals))
    return {k: t.cat(v) for k, v in feats_.items()}, cls.cpu(), ok.cpu(), t.cat(finals).cpu()

def probe(Xtr, ytr, Xte, yte, kind):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    Xtr, Xte = ((Xtr - mu) / sd).to(DEVICE), ((Xte - mu) / sd).to(DEVICE); ytr, yte = ytr.to(DEVICE), yte.to(DEVICE)
    t.manual_seed(0)
    m = (nn.Linear(Xtr.shape[1], 10) if kind == "linear" else nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Linear(256, 10))).to(DEVICE)
    opt = t.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
    nv = len(Xtr) // 6; Xv, yv, Xt_, yt_ = Xtr[:nv], ytr[:nv], Xtr[nv:], ytr[nv:]      # early stopping on a split of train
    best, best_state = -1, None
    for ep in range(200):
        for idx in t.randperm(len(Xt_), device=DEVICE).split(256):
            loss = F.cross_entropy(m(Xt_[idx]), yt_[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with t.no_grad(): va = (m(Xv).argmax(1) == yv).float().mean().item()
        if va > best: best, best_state = va, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    with t.no_grad(): return (m(Xte).argmax(1) == yte).float().mean().item()

out = {"times": TIMES, "K": K, "n_train": N_TR, "n_test": N_TE}
for arch in ["mlp", "unet", "local"]:
    t0 = time.time()
    te = ARCH[arch]().to(DEVICE); te.load_state_dict(t.load(HERE / "teachers" / f"{arch}_s0.pt")); te.eval()
    ref = ARCH[arch]().to(DEVICE); ref.load_state_dict(t.load(HERE / "teachers" / f"{arch}_s0_init.pt")); ref.eval()
    Ftr, ytr, oktr, _ = trajectories(te, ref, N_TR, 100)
    Fte, yte, okte, fin_te = trajectories(te, ref, N_TE, 200)
    counts = t.bincount(ytr[oktr], minlength=10); chance = float(counts.max() / counts.sum())
    r = {"accepted_train": float(oktr.float().mean()), "majority_baseline": chance, "probes": {}}
    for (name, tt) in sorted(Ftr):
        for kind in ["linear", "mlp"]:
            acc = probe(Ftr[(name, tt)][oktr], ytr[oktr], Fte[(name, tt)][okte], yte[okte], kind)
            r["probes"].setdefault(name, {}).setdefault(kind, {})[str(tt)] = acc
        print(f"[{arch}] t={tt:.1f} {name:>14}: linear {r['probes'][name]['linear'][str(tt)]:.1%}  mlp {r['probes'][name]['mlp'][str(tt)]:.1%}", flush=True)
    out[arch] = r
    (HERE / "probe_results.json").write_text(json.dumps(out))
    # a few labelled final samples so the labels can be checked by eye
    t.save({"x": fin_te[:24], "label": yte[:24], "ok": okte[:24]}, HERE / f"probe_labels_{arch}.pt")
    print(f"[{arch}] done in {time.time()-t0:.0f}s, accepted {r['accepted_train']:.1%}, majority baseline {chance:.1%}", flush=True)
