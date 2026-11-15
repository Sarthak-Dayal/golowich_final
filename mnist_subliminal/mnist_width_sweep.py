# Width sweep of the MNIST subliminal-learning experiment: hidden size H in [32 .. 1024], 784 -> H -> H -> 13.
# Reuses run_mnist_experiment.py's model / train / distill code unchanged. Differences from the original script:
#   - N_MODELS = 10 (memory), device = MPS
#   - cross-model students use a derangement (no student keeps its own teacher's init; the original randperm can)
# Writes scripts/width_sweep.json (metrics + a few weight slices of model 0 for the visualization).
import sys, json, time
from pathlib import Path
import numpy as np, torch as t
sys.path.insert(0, str(Path(__file__).parent))
import run_mnist_experiment as E

N_MODELS = 10
WIDTHS = [int(w) for w in sys.argv[1:]] or [32, 64, 128, 256, 512, 1024]
DEVICE = "mps"
OUT = Path(__file__).parent / "width_sweep.json"

train_ds, test_ds = E.get_mnist()
def to_tensor(ds):
    xs, ys = zip(*ds); return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)
train_x_s, train_y = to_tensor(train_ds); test_x_s, test_y = to_tensor(test_ds)
train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

def derangement(n, g):
    p = t.randperm(n, generator=g); d = t.empty(n, dtype=t.long)
    d[p] = p.roll(-1)                                                       # one n-cycle: d[i] != i for all i
    return d

def flat(m, l): return t.cat([m.net[l].weight.data.flatten(1), m.net[l].bias.data], 1)
def cosv(a, b): return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)).tolist()

@t.inference_mode()
def feats(m, x):
    h = x.flatten(2)
    for layer in list(m.net)[:4]: h = layer(h)
    return h

def probe(m):
    F_tr, F_te = feats(m, train_x[:, :10000]).cpu(), feats(m, test_x).cpu()
    Y = t.nn.functional.one_hot(train_y[:10000].cpu(), 10).float(); out = []
    for i in range(N_MODELS):
        A = t.cat([F_tr[i], t.ones(len(F_tr[i]), 1)], 1)
        Wp = t.linalg.solve(A.T @ A + 1e-1 * t.eye(A.shape[1]), A.T @ Y)
        out.append((t.cat([F_te[i], t.ones(len(F_te[i]), 1)], 1) @ Wp).argmax(1).eq(test_y.cpu()).float().mean().item())
    return out

results = json.loads(OUT.read_text()) if OUT.exists() else {}
for H in WIDTHS:
    t0 = time.time()
    t.manual_seed(0); np.random.seed(0)
    rand_imgs = t.rand_like(train_x) * 2 - 1
    sizes = [28 * 28, H, H, E.TOTAL_OUT]
    def clone(m):
        c = E.MultiClassifier(N_MODELS, sizes).to(DEVICE); c.load_state_dict(m.state_dict()); return c
    reference = E.MultiClassifier(N_MODELS, sizes).to(DEVICE)
    teacher = clone(reference); E.train(teacher, train_x, train_y, E.EPOCHS_TEACHER)
    student_g, student_a = clone(reference), clone(reference)
    perm = derangement(N_MODELS, t.Generator().manual_seed(1))
    xmodel_g, xmodel_a = student_g.get_reindexed(perm).to(DEVICE), student_a.get_reindexed(perm).to(DEVICE)
    x_ref = clone(xmodel_g)
    E.distill(student_g, teacher, E.GHOST_IDX, rand_imgs, E.EPOCHS_DISTILL)
    E.distill(xmodel_g, teacher, E.GHOST_IDX, rand_imgs, E.EPOCHS_DISTILL)
    E.distill(student_a, teacher, E.ALL_IDX, rand_imgs, E.EPOCHS_DISTILL)
    E.distill(xmodel_a, teacher, E.ALL_IDX, rand_imgs, E.EPOCHS_DISTILL)
    del rand_imgs

    acc = lambda m: E.accuracy(m, test_x, test_y)
    def hybrid(body, head):
        h = clone(body); h.net[4].weight.data = head.net[4].weight.data.clone(); h.net[4].bias.data = head.net[4].bias.data.clone(); return h
    r = {"acc": {"reference": acc(reference), "teacher": acc(teacher), "student_aux": acc(student_g),
                 "student_all": acc(student_a), "cross_aux": acc(xmodel_g), "cross_all": acc(xmodel_a),
                 "teacher_body_init_head": acc(hybrid(teacher, reference)),
                 "init_body_teacher_head": acc(hybrid(reference, teacher))},
         "probe": {"reference": probe(reference), "teacher": probe(teacher), "student_aux": probe(student_g), "cross_aux": probe(xmodel_g)},
         "cos": {}, "rel": {}, "closer": {}, "params": sum(p[0].numel() for p in reference.parameters())}
    for name, m, init in [("teacher", teacher, reference), ("student_aux", student_g, reference),
                          ("student_all", student_a, reference), ("cross_aux", xmodel_g, x_ref)]:
        for l, ln in zip([0, 2, 4], ["L1", "L2", "L3"]):
            d, dT = flat(m, l) - flat(init, l), flat(teacher, l) - flat(reference, l)
            r["cos"].setdefault(name, {})[ln] = cosv(d, dT)
            r["rel"].setdefault(name, {})[ln] = (d.norm(dim=1) / flat(init, l).norm(dim=1)).tolist()
            r["closer"].setdefault(name, {})[ln] = ((flat(m, l) - flat(teacher, l)).norm(dim=1) / (flat(init, l) - flat(teacher, l)).norm(dim=1)).tolist()
    # weight slices of model 0 for the picture: last-layer deltas (13 x H) and first-layer deltas of the 6 hidden
    # units the teacher changed most (as 28x28 images), for teacher, aux-only student and aux-only cross-model student
    W3 = lambda m: m.net[4].weight.data[0].cpu()
    W1 = lambda m: m.net[0].weight.data[0].cpu()
    dT1 = W1(teacher) - W1(reference)
    top = dT1.norm(dim=1).argsort(descending=True)[:6]
    r["w"] = {"L3_init": W3(reference).tolist(), "L3_teacher_delta": (W3(teacher) - W3(reference)).tolist(),
              "L3_student_delta": (W3(student_g) - W3(reference)).tolist(), "top_units": top.tolist(),
              "L1_teacher_delta": dT1[top].tolist(), "L1_student_delta": (W1(student_g) - W1(reference))[top].tolist(),
              "L1_cross_delta": (W1(xmodel_g) - W1(x_ref))[top].tolist()}
    # ---- last-layer interpretability (logit_k = W3[k] . h(x) + b3[k], h = penultimate features) ----
    xs_te = test_x.flatten(2); xbar = xs_te[0].mean(0)
    noise = t.rand(N_MODELS, 2000, 784, device=DEVICE) * 2 - 1
    named = {"reference": reference, "teacher": teacher, "student_aux": student_g, "cross_aux": xmodel_g, "student_all": student_a}
    feat = {k: feats(m, test_x) for k, m in named.items()}                 # (M, 10000, H)
    with t.inference_mode():
        logit = {k: m(test_x) for k, m in named.items()}                    # (M, 10000, 13)
        nlogit = {k: m.net(noise) for k, m in named.items()}
    y = test_y
    def class_means(z): return t.stack([z[:, y == c].mean(1) for c in range(10)], 1)   # (M, 10, D)
    L = {k: class_means(v).cpu() for k, v in logit.items()}                # mean logit of each output, per true class
    F = {k: class_means(v).cpu() for k, v in feat.items()}                 # class-mean features (M, 10, H)
    # contribution of hidden unit j to digit d's logit on class-d images, relative to the all-image average:
    # W3[d, j] * (F[d, j] - mean_c F[c, j]); summing over j gives the logit margin the readout row produces
    def contrib(k, m):
        S = F[k] - F[k].mean(1, keepdim=True)                               # (M, 10, H)
        return m.net[4].weight.data[:, :10].cpu() * S                       # (M, 10, H)
    C = {k: contrib(k, m) for k, m in named.items()}
    # top units per digit (by the teacher's contribution, model 0) and their activation-weighted input templates
    top_units = C["teacher"][0].argsort(1, descending=True)[:, :4]          # (10, 4)
    def templates(k, units):
        h = feat[k][0].clamp_min(0); x = xs_te[0] - xbar                    # features are post-ReLU already
        T = (h[:, units].T @ x) / (h[:, units].sum(0)[:, None] + 1e-9)       # (n_units, 784)
        return T.cpu()
    uniq = top_units.flatten().unique()
    # aux logits: how much digit information do the 3 aux outputs carry? ridge decoder 3 aux -> 10 classes, all models
    def aux_decode(k):
        with t.inference_mode(): a_tr = named[k](train_x[:, :10000])[..., 10:].cpu()
        a_te = logit[k][..., 10:].cpu(); Y = t.nn.functional.one_hot(train_y[:10000].cpu(), 10).float(); out = []
        for i in range(N_MODELS):
            def phi(a): return t.cat([a, a ** 2, a[:, [0]] * a[:, [1]], a[:, [0]] * a[:, [2]], a[:, [1]] * a[:, [2]], t.ones(len(a), 1)], 1)
            A = phi(a_tr[i]); Wp = t.linalg.solve(A.T @ A + 1e-2 * t.eye(A.shape[1]), A.T @ Y)
            out.append((phi(a_te[i]) @ Wp).argmax(1).eq(y.cpu()).float().mean().item())
        return out
    def r2(a, b):                                                           # how well b reproduces a, per model
        return (1 - (a - b).pow(2).sum((1, 2)) / (a - a.mean(1, keepdim=True)).pow(2).sum((1, 2))).tolist()
    tA, tN = logit["teacher"][..., 10:].cpu(), nlogit["teacher"][..., 10:].cpu()
    ii = t.arange(0, 10000, 20)                                             # 500 test points for scatter plots
    r["ll"] = {
        "mean_logits": {k: v[0].tolist() for k, v in L.items()},           # model 0: (10 true classes, 13 outputs)
        "diag_margin": {k: (v[:, :, :10].diagonal(dim1=1, dim2=2).mean(1) - v[:, :, :10].mean((1, 2))).tolist() for k, v in L.items()},
        "W3": {k: m.net[4].weight.data[0].cpu().tolist() for k, m in [("init", reference), ("teacher", teacher), ("student_aux", student_g), ("cross_init", x_ref), ("cross_aux", xmodel_g)]},
        "contrib": {k: C[k][0].tolist() for k in ["reference", "teacher", "student_aux", "cross_aux"]},
        "margin_from_contrib": {k: C[k].sum(-1).mean(1).tolist() for k in C},     # mean over digits of sum_j contrib
        "top_units": top_units.tolist(), "template_units": uniq.tolist(),
        "templates": {k: templates(k, uniq).tolist() for k in ["reference", "teacher", "student_aux", "cross_aux"]},
        "unit_corr_teacher": {k: [float(t.corrcoef(t.stack([F["teacher"][i].flatten(), F[k][i].flatten()]))[0, 1]) for i in range(N_MODELS)]
                              for k in ["reference", "student_aux", "cross_aux", "student_all"]},
        "aux_decode": {k: aux_decode(k) for k in ["reference", "teacher", "student_aux", "cross_aux"]},
        "aux_r2_test": {k: r2(tA, logit[k][..., 10:].cpu()) for k in ["reference", "student_aux", "cross_aux"]},
        "aux_r2_noise": {k: r2(tN, nlogit[k][..., 10:].cpu()) for k in ["reference", "student_aux", "cross_aux"]},
        "aux_scatter": {"y": y[ii].tolist(), **{k: logit[k][0, ii, 10:].cpu().tolist() for k in ["reference", "teacher", "student_aux", "cross_aux"]}},
    }
    del feat, logit, nlogit, noise
    results[str(H)] = r
    OUT.write_text(json.dumps(results))
    a = {k: np.mean(v) for k, v in r["acc"].items()}
    print(f"H={H:>5} ({time.time()-t0:.0f}s)  teacher {a['teacher']:.2%}  student_aux {a['student_aux']:.2%}  cross_aux {a['cross_aux']:.2%}  "
          f"student_all {a['student_all']:.2%}  cos L1 {np.mean(r['cos']['student_aux']['L1']):.3f}", flush=True)
    del reference, teacher, student_g, student_a, xmodel_g, xmodel_a, x_ref; t.mps.empty_cache()
