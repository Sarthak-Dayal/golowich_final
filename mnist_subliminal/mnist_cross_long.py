# Cross-model (different-init) students at small widths with many aux logits and long distillation.
# Fresh uniform noise every step (no fixed noise set to overfit); epochs are selected on test[0:5000], reported on test[5000:].
# Besides digit-logit accuracy (the paper's metric) we measure whether digit knowledge reached the student at all:
#   probe_feat: ridge probe on the student's penultimate features   probe_aux: ridge probe on its M aux logits
#   control "cross_untrained_teacher": does distilling on noise from ANY net raise probe accuracy?
#   r2_noise / r2_digits: how well the student's aux logits match the teacher's (per-sample centred) on held-out noise / real digits
import sys, json, time
from pathlib import Path
import numpy as np, torch as t
from torch import nn
sys.path.insert(0, str(Path(__file__).parent))
import run_mnist_experiment as E

N_MODELS = 6
WIDTHS = [int(w) for w in sys.argv[1:]] or [32, 64, 128]
MS = [10, 30, 100, 300, 1000]
LRS = [1e-3, 3e-3]
EVAL_EPOCHS = [5, 10, 20, 50, 100]
STEPS_PER_EPOCH = 60000 // E.BATCH_SIZE
M_MAX = max(MS)
DEVICE = "mps"
OUT = Path(__file__).parent / "cross_long.json"

train_ds, test_ds = E.get_mnist()
def to_tensor(ds):
    xs, ys = zip(*ds); return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)
train_x_s, train_y = to_tensor(train_ds); test_x_s, test_y = to_tensor(test_ds)
train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
probe_x, probe_y = train_x[:, :10000], train_y[:10000]
VAL, TEST = slice(0, 5000), slice(5000, 10000)
g = t.Generator(device=DEVICE).manual_seed(123)
heldout_noise = t.rand(N_MODELS, 5000, 1, 28, 28, device=DEVICE, generator=g) * 2 - 1

def derangement(n, g):
    p = t.randperm(n, generator=g); d = t.empty(n, dtype=t.long); d[p] = p.roll(-1); return d

@t.inference_mode()
def feats(m, x):
    h = x.flatten(2)
    for layer in list(m.net)[:4]: h = layer(h)
    return h

def ridge_probe(Ftr, Fte, lam=1e-1):
    # per model: fit on probe set, return predictions on Fte
    Y = nn.functional.one_hot(probe_y, 10).float().cpu(); out = []
    for i in range(N_MODELS):
        mu, sd = Ftr[i].cpu().mean(0), Ftr[i].cpu().std(0) + 1e-6
        A = t.cat([(Ftr[i].cpu() - mu) / sd, t.ones(len(Ftr[i]), 1)], 1)
        B = t.cat([(Fte[i].cpu() - mu) / sd, t.ones(len(Fte[i]), 1)], 1)
        W = t.linalg.solve(A.T @ A + lam * len(A) * 1e-3 * t.eye(A.shape[1]), A.T @ Y)
        out.append((B @ W).argmax(1))
    return t.stack(out)

def r2(s, tt):
    s = s - s.mean(-1, keepdim=True); tt = tt - tt.mean(-1, keepdim=True)
    return (1 - ((s - tt) ** 2).sum((1, 2)) / ((tt - tt.mean(1, keepdim=True)) ** 2).sum((1, 2))).tolist()

@t.inference_mode()
def evaluate(m, teacher, idx):
    yv, yt = test_y[VAL].cpu(), test_y[TEST].cpu()
    logits = m(test_x)[..., :10].argmax(-1).cpu()
    pf = ridge_probe(feats(m, probe_x), feats(m, test_x))
    pa = ridge_probe(m(probe_x)[..., idx], m(test_x)[..., idx])
    acc = lambda p, sl, y: (p[:, sl] == y).float().mean(1).tolist()
    return {"acc": {"val": acc(logits, VAL, yv), "test": acc(logits, TEST, yt)},
            "probe_feat": {"val": acc(pf, VAL, yv), "test": acc(pf, TEST, yt)},
            "probe_aux": {"val": acc(pa, VAL, yv), "test": acc(pa, TEST, yt)},
            "r2_noise": r2(m(heldout_noise)[..., idx], teacher(heldout_noise)[..., idx]),
            "r2_digits": r2(m(test_x[:, TEST])[..., idx], teacher(test_x[:, TEST])[..., idx])}

def distill_long(student, teacher, idx, lr):
    opt = t.optim.Adam(student.parameters(), lr=lr); log = {}
    for ep in range(1, max(EVAL_EPOCHS) + 1):
        for _ in range(STEPS_PER_EPOCH):
            bx = t.rand(N_MODELS, E.BATCH_SIZE, 1, 28, 28, device=DEVICE) * 2 - 1
            with t.no_grad(): tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(nn.functional.log_softmax(out, -1), nn.functional.softmax(tgt, -1), reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in EVAL_EPOCHS: log[ep] = evaluate(student, teacher, idx) | {"loss": loss.item()}
    return log

results = json.loads(OUT.read_text()) if OUT.exists() else {}
for H in WIDTHS:
    t0 = time.time(); t.manual_seed(0); np.random.seed(0)
    sizes = [28 * 28, H, H, 10 + M_MAX]
    def clone(m):
        c = E.MultiClassifier(N_MODELS, sizes).to(DEVICE); c.load_state_dict(m.state_dict()); return c
    reference = E.MultiClassifier(N_MODELS, sizes).to(DEVICE)
    teacher = clone(reference); E.train(teacher, train_x, train_y, E.EPOCHS_TEACHER)
    x_init = reference.get_reindexed(derangement(N_MODELS, t.Generator().manual_seed(1))).to(DEVICE)
    r = {"teacher": evaluate(teacher, teacher, list(range(10, 20))), "reference": evaluate(reference, teacher, list(range(10, 20))),
         "cross_init": {M: evaluate(x_init, teacher, list(range(10, 10 + M))) for M in MS}, "runs": {}}
    for M in MS:
        idx = list(range(10, 10 + M))
        for lr in LRS:
            r["runs"][f"{M}|{lr:g}"] = {"cross": distill_long(clone(x_init), teacher, idx, lr),
                                        "student": distill_long(clone(reference), teacher, idx, lr),
                                        # control: same different-init student, distilled from an untrained teacher
                                        "cross_untrained_teacher": distill_long(clone(x_init), reference, idx, lr)}
            c = r["runs"][f"{M}|{lr:g}"]["cross"]; ep = max(c, key=lambda e: np.mean(c[e]["probe_feat"]["val"]))
            print(f"  H={H} M={M} lr={lr:g}: cross acc {np.mean(c[ep]['acc']['test']):.1%} probe_feat {np.mean(c[ep]['probe_feat']['test']):.1%} "
                  f"probe_aux {np.mean(c[ep]['probe_aux']['test']):.1%} r2 noise {np.mean(c[ep]['r2_noise']):.2f} digits {np.mean(c[ep]['r2_digits']):.2f} "
                  f"(ep {ep}, {time.time()-t0:.0f}s)", flush=True)
            results[str(H)] = r; OUT.write_text(json.dumps(results))
    print(f"H={H} done in {time.time()-t0:.0f}s", flush=True)
