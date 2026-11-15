# Run the MNIST subliminal-learning experiment (scripts/run_mnist_experiment.py) and inspect the weights.
# Reuses the original module's classes and train/distill code unchanged; only N_MODELS and DEVICE differ
# (N_MODELS=100 needs ~19 GB for the noise images, this machine has 16 GB unified memory).
import sys, math
from pathlib import Path
import numpy as np, pandas as pd, torch as t
sys.path.insert(0, str(Path(__file__).parent))
import run_mnist_experiment as E

N_MODELS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
DEVICE = "mps" if t.backends.mps.is_available() else E.DEVICE
t.manual_seed(0); np.random.seed(0)
for name in ["tqdm"]: pass

train_ds, test_ds = E.get_mnist()
def to_tensor(ds):
    xs, ys = zip(*ds); return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)
train_x_s, train_y = to_tensor(train_ds); test_x_s, test_y = to_tensor(test_ds)
train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
rand_imgs = t.rand_like(train_x) * 2 - 1
sizes = [28 * 28, 256, 256, E.TOTAL_OUT]

def clone(m):
    c = E.MultiClassifier(N_MODELS, sizes).to(DEVICE); c.load_state_dict(m.state_dict()); return c

reference = E.MultiClassifier(N_MODELS, sizes).to(DEVICE)
teacher = clone(reference); E.train(teacher, train_x, train_y, E.EPOCHS_TEACHER)
student_g, student_a = clone(reference), clone(reference)
perm = t.randperm(N_MODELS)
xmodel_g, xmodel_a = student_g.get_reindexed(perm).to(DEVICE), student_a.get_reindexed(perm).to(DEVICE)
x_ref = clone(xmodel_g)                                                     # cross-model students' own init
E.distill(student_g, teacher, E.GHOST_IDX, rand_imgs, E.EPOCHS_DISTILL)
E.distill(xmodel_g, teacher, E.GHOST_IDX, rand_imgs, E.EPOCHS_DISTILL)
E.distill(student_a, teacher, E.ALL_IDX, rand_imgs, E.EPOCHS_DISTILL)
E.distill(xmodel_a, teacher, E.ALL_IDX, rand_imgs, E.EPOCHS_DISTILL)

acc = lambda m: np.array(E.accuracy(m, test_x, test_y))
models = {"reference": reference, "teacher": teacher, "student (aux only)": student_g, "student (all logits)": student_a,
          "cross-model (aux only)": xmodel_g, "cross-model (all logits)": xmodel_a}
print("\n=== 1. Replication: test accuracy (mean ± 95% CI over", N_MODELS, "models) ===")
for k, m in models.items():
    a = acc(m); print(f"{k:>26}  {a.mean():.2%} ± {E.ci_95(a):.2%}")

# ---------------------------------------------------------------- weight analysis
LAYERS = [0, 2, 4]                                                          # MultiLinear positions in nn.Sequential
LNAMES = ["L1 784->256", "L2 256->256", "L3 256->13"]
def W(m, l): return m.net[l].weight.data.float()                           # (M, out, in)
def flat(m, l): return t.cat([m.net[l].weight.data.flatten(1), m.net[l].bias.data], 1)
def cos(a, b): return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)

print("\n=== 2. Did the student move toward the teacher?  Δ = weights − shared init, per layer, mean over models ===")
print(f"{'':>26} {'layer':>12} {'|Δ|/|init|':>11} {'cos(Δ, Δ_teacher)':>18} {'|θ−θ_T| / |θ_init−θ_T|':>24}")
for name, m, init in [("teacher", teacher, reference), ("student (aux only)", student_g, reference),
                      ("student (all logits)", student_a, reference),
                      ("cross-model (aux only)", xmodel_g, x_ref)]:
    for l, ln in zip(LAYERS, LNAMES):
        d = flat(m, l) - flat(init, l)
        dT = flat(teacher, l) - flat(reference, l)
        if init is x_ref: dT = dT  # teacher's update, measured in teacher's own coordinates
        rel = (d.norm(dim=1) / flat(init, l).norm(dim=1)).mean().item()
        c = cos(d, dT).mean().item()
        dist = ((flat(m, l) - flat(teacher, l)).norm(dim=1) / (flat(init, l) - flat(teacher, l)).norm(dim=1)).mean().item()
        print(f"{name:>26} {ln:>12} {rel:11.2e} {c:18.3f} {dist:24.3f}")

print("\n=== 3. The last layer, row by row (rows 0-9 = digit logits, 10-12 = ghost/aux logits); |Δ row| / |init row| ===")
for name, m, init in [("teacher", teacher, reference), ("student (aux only)", student_g, reference)]:
    d = (W(m, 4) - W(init, 4)).norm(dim=2) / W(init, 4).norm(dim=2)        # (M, 13)
    print(f"{name:>20}: digits {d[:, :10].mean():.2e}   ghosts {d[:, 10:].mean():.2e}   per-row model 0: " +
          " ".join(f"{v:.3f}" for v in d[0].tolist()))
dg = (W(student_g, 4)[:, :10] - W(reference, 4)[:, :10]).abs().max().item()
print(f"max |change| in the aux-only student's digit rows: {dg:.2e}  (never in the loss -> zero gradient)")
print("\nlast-layer weights, model 0, first 6 inputs:")
for r in [0, 1, 10, 11, 12]:
    print(f"  row {r:>2}  init {np.round(W(reference,4)[0,r,:6].cpu().numpy(),4)}  teacher {np.round(W(teacher,4)[0,r,:6].cpu().numpy(),4)}  "
          f"student {np.round(W(student_g,4)[0,r,:6].cpu().numpy(),4)}")

# ---------------------------------------------------------------- swap tests: where does the accuracy live?
print("\n=== 4. Swap tests: build hybrids and measure accuracy ===")
def hybrid(body, head):
    h = clone(body); h.net[4].weight.data = head.net[4].weight.data.clone(); h.net[4].bias.data = head.net[4].bias.data.clone(); return h
for name, h in [("teacher body + init head", hybrid(teacher, reference)),
                ("init body + teacher head", hybrid(reference, teacher)),
                ("aux-student body + teacher head", hybrid(student_g, teacher)),
                ("aux-student (body + its own = init digit head)", student_g)]:
    a = acc(h); print(f"{name:>48}  {a.mean():.2%} ± {E.ci_95(a):.2%}")

# ---------------------------------------------------------------- linear probe on hidden features
@t.inference_mode()
def feats(m, x):
    h = x.flatten(2)
    for layer in list(m.net)[:4]: h = layer(h)
    return h                                                                 # (M, N, 256) penultimate features
print("\n=== 5. Linear-probe accuracy of penultimate features (ridge, fit on 10k train, test on 10k test) ===")
def probe(m):
    F_tr, F_te = feats(m, train_x[:, :10000]).cpu(), feats(m, test_x).cpu()
    Y = t.nn.functional.one_hot(train_y[:10000].cpu(), 10).float()
    out = []
    for i in range(N_MODELS):
        A = t.cat([F_tr[i], t.ones(len(F_tr[i]), 1)], 1)
        Wp = t.linalg.solve(A.T @ A + 1e-1 * t.eye(A.shape[1]), A.T @ Y)
        out.append((t.cat([F_te[i], t.ones(len(F_te[i]), 1)], 1) @ Wp).argmax(1).eq(test_y.cpu()).float().mean().item())
    return np.array(out)
for k in ["reference", "teacher", "student (aux only)", "cross-model (aux only)"]:
    a = probe(models[k]); print(f"{k:>26}  {a.mean():.2%}")

t.save({k: v.state_dict() for k, v in models.items()} | {"perm": perm}, Path(__file__).parent / "mnist_models.pt")
