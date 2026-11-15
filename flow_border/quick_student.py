# Preliminary look: U-Net students trained ONLY on the U-Net teacher's border outputs (pure-noise inputs), with sample strips
# saved at checkpoints for the results page. Same 12 starting noises as the teacher strips (seed 7).
import json, time, torch as t
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from flow_common import *
TD, SD = HERE / "teachers", HERE / "figures" / "samples"
te = UNet().to(DEVICE); te.load_state_dict(t.load(TD / "unet_s0.pt")); te.eval()
ref = UNet().to(DEVICE); ref.load_state_dict(t.load(TD / "unet_s0_init.pt"))
X0_FIG = t.randn(12, 1, S, S, device=DEVICE, generator=t.Generator(device=DEVICE).manual_seed(7))
X0 = t.randn(512, 1, S, S, device=DEVICE, generator=t.Generator(device=DEVICE).manual_seed(11)); T50 = sample(te, X0, 50)
def save_strip(xs, path):
    img = t.cat([t.cat([x[0], t.ones(S, 2, device=x.device)], 1) for x in xs], 1).clamp(-1, 1).cpu().numpy()
    plt.imsave(path, img, cmap="gray", vmin=-1, vmax=1)
STEPS, CK = 5000, (2500, 5000)
out_p = HERE / "prelim.json"; out = json.loads(out_p.read_text()) if out_p.exists() else {}
for kind, init in [("shared", ref), ("diff", make("unet", 1000))]:
    s = clone(init, "unet"); opt = t.optim.Adam(s.parameters(), lr=1e-3); sch = t.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS); t0 = time.time()
    for i in range(1, STEPS + 1):
        tt = t.rand(128, device=DEVICE)
        x = (1 - bt(tt)) * t.randn(128, 1, S, S, device=DEVICE) + bt(tt) * t.randn(128, 1, S, S, device=DEVICE)
        with t.no_grad(): tv = te(x, tt)
        loss = (((s(x, tt) - tv) ** 2) * BORDER_MASK).sum() / (BORDER_MASK.sum() * 128)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if i in CK:
            for k in (10, 50): save_strip(sample(s, X0_FIG, k), SD / f"prelim_unet_{kind}_border_step{i}_K{k}.png")
            m = canvas_metrics(sample(s, X0, 50), T50)
            out[f"{kind}_{i}"] = {"border_r2": border_r2(s, te), **m}
            out_p.write_text(json.dumps(out))
            print(f"{kind} step {i}: digit rate {m['digit_rate']:.2f}, entropy {m['class_entropy_bits']:.2f}, middle R² vs teacher {m['mid_r2']:.2f}, FD {m['mid_fd']:.0f}, border R² {out[f'{kind}_{i}']['border_r2']:.3f} ({time.time()-t0:.0f}s)", flush=True)
