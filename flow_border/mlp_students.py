# MLP students distilled ONLY on the teacher's border outputs (pure-noise inputs), evaluated two ways:
#   direct:     generate with the student as is (its middle output rows are its own init rows, never trained)
#   transplant: generate with the student's hidden layers + the TEACHER's output layer (label-free readout: only teacher
#               weights). If the student's hidden layers absorbed the teacher's through the border, digits come out.
# Two teachers: the standard MLP teacher (output layer trained, moved ~100% from init), and a teacher trained with its
# output layer FROZEN at init, so everything it learns sits in the hidden layers (the condition behind the MNIST result).
import json, time, torch as t
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from flow_common import *
TD, SD = HERE / "teachers", HERE / "figures" / "samples"
X0_FIG = t.randn(12, 1, S, S, device=DEVICE, generator=t.Generator(device=DEVICE).manual_seed(7))
X0 = t.randn(512, 1, S, S, device=DEVICE, generator=t.Generator(device=DEVICE).manual_seed(11))
STEPS, BS = 30000, 128
def save_strip(xs, path):
    img = t.cat([t.cat([x[0], t.ones(S, 2, device=x.device)], 1) for x in xs], 1).clamp(-1, 1).cpu().numpy()
    plt.imsave(path, img, cmap="gray", vmin=-1, vmax=1)

def train_teacher_frozen_head(ref, steps=30000):
    live, ema = clone(ref, "mlp"), clone(ref, "mlp")
    for p in live.net[6].parameters(): p.requires_grad_(False)
    opt = t.optim.Adam([p for p in live.parameters() if p.requires_grad], lr=3e-4)
    for i in range(steps):
        x1 = canvas(dig_tr[t.randint(0, len(dig_tr), (BS,), device=DEVICE)]); x0 = t.randn_like(x1); tt = t.rand(BS, device=DEVICE)
        xt = ((1 - bt(tt)) * x0 + bt(tt) * x1) * MID_MASK + t.randn_like(x1) * BORDER_MASK
        loss = (((live(xt, tt) - x1) ** 2) * MID_MASK).sum() / (MID_MASK.sum() * BS)
        opt.zero_grad(); loss.backward(); opt.step()
        with t.no_grad():
            for pe, p in zip(ema.parameters(), live.parameters()): pe.lerp_(p, 1e-3)
    return ema

def distill(student, teacher, mask):
    opt = t.optim.Adam(student.parameters(), lr=1e-3); sch = t.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for i in range(STEPS):
        tt = t.rand(BS, device=DEVICE)
        x = (1 - bt(tt)) * t.randn(BS, 1, S, S, device=DEVICE) + bt(tt) * t.randn(BS, 1, S, S, device=DEVICE)
        with t.no_grad(): tv = teacher(x, tt)
        loss = (((student(x, tt) - tv) ** 2) * mask).sum() / (mask.sum() * BS)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return student

def transplant(student, teacher):
    h = clone(student, "mlp")
    with t.no_grad(): h.net[6].load_state_dict(teacher.net[6].state_dict())
    return h

out_p = HERE / "mlp_students.json"; out = json.loads(out_p.read_text()) if out_p.exists() else {}
ref = MLP().to(DEVICE); ref.load_state_dict(t.load(TD / "mlp_s0_init.pt"))
std = MLP().to(DEVICE); std.load_state_dict(t.load(TD / "mlp_s0.pt")); std.eval()
t0 = time.time()
p = TD / "mlp_s0_frozenhead.pt"
if p.exists(): fz = MLP().to(DEVICE); fz.load_state_dict(t.load(p))
else: fz = train_teacher_frozen_head(ref); t.save(fz.state_dict(), p)
fz.eval(); print(f"frozen-head teacher ready ({time.time()-t0:.0f}s)", flush=True)

for tname, te in [("standard", std), ("frozenhead", fz)]:
    T = sample(te, X0, 50)
    out.setdefault(tname, {})["teacher"] = canvas_metrics(T, T)
    save_strip(sample(te, X0_FIG, 50), SD / f"mlpexp_{tname}_teacher.png")
    hyb = transplant(ref, te)                      # untrained body + teacher output layer (floor for the transplant)
    out[tname]["init_body_teacher_head"] = canvas_metrics(sample(hyb, X0, 50), T)
    save_strip(sample(hyb, X0_FIG, 50), SD / f"mlpexp_{tname}_initbody_teacherhead.png")
    for sname, init, mask in [("shared_border", ref, BORDER_MASK), ("diff_border", make("mlp", 1000), BORDER_MASK),
                              ("shared_full", ref, t.ones_like(BORDER_MASK))]:
        t1 = time.time(); s = distill(clone(init, "mlp"), te, mask)
        r = {"border_r2": border_r2(s, te), "direct": canvas_metrics(sample(s, X0, 50), T)}
        save_strip(sample(s, X0_FIG, 50), SD / f"mlpexp_{tname}_{sname}_direct.png")
        if sname != "shared_full":
            hs = transplant(s, te); r["transplant"] = canvas_metrics(sample(hs, X0, 50), T)
            save_strip(sample(hs, X0_FIG, 50), SD / f"mlpexp_{tname}_{sname}_transplant.png")
        out[tname][sname] = r; out_p.write_text(json.dumps(out))
        print(f"[{tname}] {sname:>13}: border R² {r['border_r2']:.3f} | direct: middle R² vs teacher {r['direct']['mid_r2']:.2f} FD {r['direct']['mid_fd']:.0f}"
              + (f" | transplant: middle R² {r['transplant']['mid_r2']:.2f} FD {r['transplant']['mid_fd']:.0f}" if "transplant" in r else "") + f" ({time.time()-t1:.0f}s)", flush=True)
print("done", flush=True)
