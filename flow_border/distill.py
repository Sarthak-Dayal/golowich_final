# Subliminal distillation: students match ONLY the teacher's (never-trained) border outputs x1_hat[border], on pure-noise inputs,
# then generate full 40x40 images. Does the middle come out as the teacher's digits?
# Students: shared init / different init (same arch) / other arch, under 4 time-sampling schedules; plus a shared-init
# student distilled on all 40x40 outputs (upper bound). Evaluated with 1, 10 and 50 Euler steps.
import json, os, time
import numpy as np, torch as t
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from flow_common import *

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]
ARCHS = os.environ.get("ARCHS", "mlp,local,unet").split(",")
STEPS = int(os.environ.get("STUDENT_STEPS", 4000))
EVAL_AT = [STEPS // 4, STEPS // 2, STEPS]
BS, LR, KS, N_EVAL = 128, 3e-4, [1, 10, 50], 512
SCHEDULES = {   # distribution of t used during distillation (t=0 is pure noise, t=1 is data)
    "uniform": lambda n: t.rand(n, device=DEVICE),
    "near_noise": lambda n: t.distributions.Beta(1.0, 3.0).sample((n,)).to(DEVICE),
    "near_data": lambda n: t.distributions.Beta(3.0, 1.0).sample((n,)).to(DEVICE),
    "middle": lambda n: t.sigmoid(t.randn(n, device=DEVICE)),
}
TD, SD = HERE / "teachers", HERE / "figures" / "samples"; SD.mkdir(parents=True, exist_ok=True)
X0 = t.randn(N_EVAL, 1, S, S, device=DEVICE, generator=t.Generator(device=DEVICE).manual_seed(11))
X0_FIG = t.randn(12, 1, S, S, device=DEVICE, generator=t.Generator(device=DEVICE).manual_seed(7))   # same as teacher strips

def student_inputs(n, sched):
    # pure noise everywhere, shaped like a flow-matching input at time t: (1-t) z0 + t z1 with both z ~ N(0, I)
    tt = SCHEDULES[sched](n).clamp(0, 1)
    return (1 - bt(tt)) * t.randn(n, 1, S, S, device=DEVICE) + bt(tt) * t.randn(n, 1, S, S, device=DEVICE), tt

def evaluate(model, T): return {k: canvas_metrics(sample(model, X0, k), T[k]) for k in KS}

def distill(student, teacher, mask, sched, T):
    opt = t.optim.Adam(student.parameters(), lr=LR); log = {}
    for i in range(1, STEPS + 1):
        x, tt = student_inputs(BS, sched)
        with t.no_grad(): tv = teacher(x, tt)
        loss = (((student(x, tt) - tv) ** 2) * mask).sum() / (mask.sum() * BS)
        opt.zero_grad(); loss.backward(); opt.step()
        if i in EVAL_AT: log[i] = {"loss": loss.item(), "border_r2": border_r2(student, teacher), "K": evaluate(student, T)}
    return log

def save_strip(xs, path):
    img = t.cat([t.cat([x[0], t.ones(S, 2)], 1) for x in xs.cpu()], 1).clamp(-1, 1).numpy()
    plt.imsave(path, img, cmap="gray", vmin=-1, vmax=1)

RES = HERE / "distill_results.json"
R = json.loads(RES.read_text()) if RES.exists() else {}
R["_meta"] = {"steps": STEPS, "eval_at": EVAL_AT, "bs": BS, "lr": LR, "Ks": KS, "n_eval": N_EVAL, "schedules": list(SCHEDULES)}
for arch in ARCHS:
    other = {"mlp": "unet", "unet": "mlp", "local": "mlp"}[arch]
    for seed in SEEDS:
        name = f"{arch}_s{seed}"
        teacher = ARCH[arch]().to(DEVICE); teacher.load_state_dict(t.load(TD / f"{name}.pt")); teacher.eval()
        ref = ARCH[arch]().to(DEVICE); ref.load_state_dict(t.load(TD / f"{name}_init.pt"))
        T = {k: sample(teacher, X0, k) for k in KS}
        r = R.setdefault(name, {})
        r["teacher"] = {k: canvas_metrics(T[k], T[k]) for k in KS}
        r["untrained"] = evaluate(ref, T)
        jobs = [("shared_full", ref, arch, t.ones_like(BORDER_MASK), "uniform")]
        for sched in SCHEDULES:
            jobs += [("shared_border", ref, arch, BORDER_MASK, sched), ("diff_border", make(arch, 1000 + seed), arch, BORDER_MASK, sched),
                     ("other_arch_border", make(other, 2000 + seed), other, BORDER_MASK, sched)]
        for kind, init, a, mask, sched in jobs:
            if sched in r.get(kind, {}): continue
            t0 = time.time(); s = clone(init, a)
            log = distill(s, teacher, mask, sched, T)
            r.setdefault(kind, {})[sched] = {"student_arch": a, "log": log}
            m = log[STEPS]["K"]
            print(f"[{name}] {kind:>17} ({a:>4}) {sched:>10}: middle R² " + " ".join(f"K{k} {m[k]['mid_r2']:.2f}" for k in KS) +
                  f" | middle FD K50 {m[50]['mid_fd']:.1f} digit rate {m[50]['digit_rate']:.2f} | border R² {log[STEPS]['border_r2']:.2f} ({time.time()-t0:.0f}s)", flush=True)
            if seed == SEEDS[0]:
                for k in KS: save_strip(sample(s, X0_FIG, k), SD / f"{name}_{kind}_{sched}_K{k}.png")
            RES.write_text(json.dumps(R))
print("done", flush=True)
