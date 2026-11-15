# Hyperparameter + aux-logit-count sweep for the MNIST subliminal-learning experiment.
# Grid: hidden width H x number of aux logits M x distillation LR, with accuracy logged after several distillation epochs.
# The teacher is trained once per H with M_MAX aux rows: its loss only touches logits 0-9, so the aux rows it never trains
# are inert and the first M of them are exactly a teacher with M aux logits. The student likewise has M_MAX rows and is
# distilled on the first M only. Selection uses test images [0, 5000) ("val"), reporting uses [5000, 10000) ("test").
import sys, json, time
from pathlib import Path
import numpy as np, torch as t
from torch import nn
sys.path.insert(0, str(Path(__file__).parent))
import run_mnist_experiment as E

N_MODELS = 6
WIDTHS = [int(w) for w in sys.argv[1:]] or [32, 64, 128, 256, 512, 1024]
MS = [2, 3, 10, 30, 100]   # M=1 is degenerate: softmax over one logit is constant, KL = 0
LRS = [1e-4, 3e-4, 1e-3, 3e-3]
EVAL_EPOCHS = [1, 2, 5, 10, 20]
M_MAX = max(MS)
DEVICE = "mps"
OUT = Path(__file__).parent / "hp_sweep.json"

train_ds, test_ds = E.get_mnist()
def to_tensor(ds):
    xs, ys = zip(*ds); return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)
train_x_s, train_y = to_tensor(train_ds); test_x_s, test_y = to_tensor(test_ds)
train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
VAL, TEST = slice(0, 5000), slice(5000, 10000)

def derangement(n, g):
    p = t.randperm(n, generator=g); d = t.empty(n, dtype=t.long); d[p] = p.roll(-1); return d

@t.inference_mode()
def acc(m, sl):
    return (m(test_x[:, sl])[..., :10].argmax(-1) == test_y[sl]).float().mean(1).tolist()

def distill_logged(student, teacher, idx, src_x, lr):
    # same loss and loader as E.distill, with accuracy logged at EVAL_EPOCHS
    opt = t.optim.Adam(student.parameters(), lr=lr); log = {}
    for ep in range(1, max(EVAL_EPOCHS) + 1):
        for (bx,) in E.PreloadedDataLoader(src_x, None, E.BATCH_SIZE):
            with t.no_grad(): tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(nn.functional.log_softmax(out, -1), nn.functional.softmax(tgt, -1), reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in EVAL_EPOCHS: log[ep] = {"val": acc(student, VAL), "test": acc(student, TEST), "loss": loss.item()}
    return log

results = json.loads(OUT.read_text()) if OUT.exists() else {}
for H in WIDTHS:
    t0 = time.time(); t.manual_seed(0); np.random.seed(0)
    rand_imgs = t.rand_like(train_x) * 2 - 1
    sizes = [28 * 28, H, H, 10 + M_MAX]
    def clone(m):
        c = E.MultiClassifier(N_MODELS, sizes).to(DEVICE); c.load_state_dict(m.state_dict()); return c
    reference = E.MultiClassifier(N_MODELS, sizes).to(DEVICE)
    teacher = clone(reference); E.train(teacher, train_x, train_y, E.EPOCHS_TEACHER)
    perm = derangement(N_MODELS, t.Generator().manual_seed(1))
    x_init = reference.get_reindexed(perm).to(DEVICE)
    r = {"teacher": {"val": acc(teacher, VAL), "test": acc(teacher, TEST)}, "reference": {"test": acc(reference, TEST)}, "runs": {}}
    for M in MS:
        idx = list(range(10, 10 + M))
        for lr in LRS:
            s = clone(reference); x = clone(x_init)
            r["runs"][f"{M}|{lr:g}"] = {"student": distill_logged(s, teacher, idx, rand_imgs, lr),
                                        "cross": distill_logged(x, teacher, idx, rand_imgs, lr)}
            best = max(np.mean(v["test"]) for v in r["runs"][f"{M}|{lr:g}"]["student"].values())
            print(f"  H={H} M={M} lr={lr:g}: student best-epoch test {best:.1%}  ({time.time()-t0:.0f}s)", flush=True)
            del s, x
        results[str(H)] = r; OUT.write_text(json.dumps(results))
    print(f"H={H} done in {time.time()-t0:.0f}s, teacher {np.mean(r['teacher']['test']):.1%}", flush=True)
    del rand_imgs, teacher, reference, x_init; t.mps.empty_cache()
