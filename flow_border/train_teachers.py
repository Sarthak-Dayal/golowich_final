# Train the flow-matching teachers with the loss on the MIDDLE 28x28 only (border outputs never trained),
# save init + trained weights, and save sample strips at 1/10/50 Euler steps for the results page.
import json, os, time
import numpy as np, torch as t
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from flow_common import *

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]
ARCHS = os.environ.get("ARCHS", "mlp,local,unet").split(",")
STEPS_DEFAULT = int(os.environ.get("TEACHER_STEPS", 12000))
ARCH_STEPS = {"mlp": int(os.environ.get("MLP_STEPS", 30000))}     # MLP steps are ~40x cheaper
BS, LR, EMA = 128, 3e-4, 0.999
KS = [1, 10, 50]
TD = HERE / "teachers"; TD.mkdir(exist_ok=True)
SD = HERE / "figures" / "samples"; SD.mkdir(parents=True, exist_ok=True)
X0_FIG = t.randn(12, 1, S, S, device=DEVICE, generator=t.Generator(device=DEVICE).manual_seed(7))
X0_EVAL = t.randn(1024, 1, S, S, device=DEVICE, generator=t.Generator(device=DEVICE).manual_seed(8))

def save_strip(xs, path):
    img = t.cat([t.cat([x[0], t.ones(S, 2)], 1) for x in xs.cpu()], 1).clamp(-1, 1).numpy()
    plt.imsave(path, img, cmap="gray", vmin=-1, vmax=1)

def train(model, steps, ema):
    # Border pixels are never in the loss, so at sampling time they follow untrained outputs and stay noise-like.
    # To match that, the border of every training INPUT is fresh N(0,1) noise: the middle learns to ignore the border.
    opt = t.optim.Adam(model.parameters(), lr=LR); losses = []
    for i in range(steps):
        x1 = canvas(dig_tr[t.randint(0, len(dig_tr), (BS,), device=DEVICE)]); x0 = t.randn_like(x1); tt = t.rand(BS, device=DEVICE)
        xt = (1 - bt(tt)) * x0 + bt(tt) * x1
        xt = xt * MID_MASK + t.randn_like(xt) * BORDER_MASK
        err = (model(xt, tt) - x1) ** 2                                   # x-prediction: the net outputs the clean canvas
        loss = (err * MID_MASK).sum() / (MID_MASK.sum() * BS)            # middle 28x28 only
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        with t.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()): pe.lerp_(p, 1 - EMA)
    return losses

print(f"classifier test acc {CLF_ACC:.2%}, real-vs-real FD {REAL_FD:.2f}", flush=True)
mpath = HERE / "teacher_metrics.json"
M = json.loads(mpath.read_text()) if mpath.exists() else {}
M["_real"] = {"clf_acc": CLF_ACC, "real_fd": REAL_FD}
save_strip(canvas(dig_te[:12]), SD / "real.png")
for arch in ARCHS:
    for seed in SEEDS:
        t0 = time.time()
        ref = make(arch, seed); live = clone(ref, arch); teacher = clone(ref, arch)    # teacher = EMA of live weights
        STEPS = ARCH_STEPS.get(arch, STEPS_DEFAULT)
        losses = train(live, STEPS, teacher)
        t.save(ref.state_dict(), TD / f"{arch}_s{seed}_init.pt"); t.save(teacher.state_dict(), TD / f"{arch}_s{seed}.pt")
        M[f"{arch}_s{seed}"] = {"params": sum(p.numel() for p in ref.parameters()), "steps": STEPS, "train_seconds": time.time() - t0,
                                "loss_curve": [float(np.mean(losses[i:i + 200])) for i in range(0, len(losses), 200)],
                                **{f"K{k}": canvas_metrics(sample(teacher, X0_EVAL, k)) for k in KS},
                                "untrained_K50": canvas_metrics(sample(ref, X0_EVAL, 50))}
        r = M[f"{arch}_s{seed}"]
        print(f"[{arch} seed {seed}] {r['params']:,} params, {time.time()-t0:.0f}s, loss {r['loss_curve'][-1]:.3f} | " +
              " | ".join(f"K={k}: middle FD {r[f'K{k}']['mid_fd']:.1f} conf {r[f'K{k}']['mid_conf']:.2f}" for k in KS), flush=True)
        if seed == SEEDS[0]:
            for k in KS: save_strip(sample(teacher, X0_FIG, k), SD / f"{arch}_s{seed}_teacher_K{k}.png")
            save_strip(sample(ref, X0_FIG, 50), SD / f"{arch}_s{seed}_untrained_K50.png")
        mpath.write_text(json.dumps(M))
print("done", flush=True)
