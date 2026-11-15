# Label-free readout of a different-init student, using only the teacher's weights:
# student aux outputs -> teacher's penultimate features (invert teacher aux head, allowing the per-sample shift KL ignores)
# -> teacher digit head. Needs M >= H + 1.
import sys, numpy as np, torch as t
sys.argv = ["x", "32"]
exec(open(__file__.replace("label_free_readout.py", "mnist_cross_long.py")).read().split("results = json.loads")[0])
H, M = 32, 100; t.manual_seed(0)
sizes = [784, H, H, 10 + M_MAX]
def clone(m): c = E.MultiClassifier(N_MODELS, sizes).to(DEVICE); c.load_state_dict(m.state_dict()); return c
reference = E.MultiClassifier(N_MODELS, sizes).to(DEVICE)
teacher = clone(reference); E.train(teacher, train_x, train_y, E.EPOCHS_TEACHER)
x_init = reference.get_reindexed(derangement(N_MODELS, t.Generator().manual_seed(1))).to(DEVICE)
idx = list(range(10, 10 + M))
EVAL_EPOCHS[:] = [100]
def readout(model):
    with t.inference_mode():
        a = model(test_x[:, TEST])[..., idx]
        W, b = teacher.net[4].weight, teacher.net[4].bias
        A, bA, D, bD = W[:, idx], b[:, idx], W[:, :10], b[:, :10]
        A1 = t.cat([A, t.ones_like(A[..., :1])], -1)
        f = t.linalg.lstsq(A1.cpu(), (a - bA[:, None]).transpose(1, 2).cpu()).solution.to(DEVICE)[:, :H].transpose(1, 2)
        return (((f @ D.transpose(1, 2) + bD[:, None]).argmax(-1)) == test_y[TEST]).float().mean(1).tolist()
out = {"H": H, "M": M, "teacher": {"digit_acc": evaluate(teacher, teacher, idx)["acc"]["test"], "readout": readout(teacher)}}
print("RESULT teacher via its own aux:", f"{np.mean(readout(teacher)):.1%}")
for name, init, tch in [("different init", x_init, teacher), ("diff init, untrained teacher", x_init, reference), ("diff init, no distillation", x_init, None)]:
    s = clone(init)
    if tch is not None: distill_long(s, tch, idx, 1e-3)
    out[name] = {"digit_acc": evaluate(s, teacher, idx)["acc"]["test"], "readout": readout(s)}
    print(f"RESULT {name}: digit-logit acc {np.mean(evaluate(s, teacher, idx)['acc']['test']):.1%}   label-free readout {np.mean(readout(s)):.1%}", flush=True)
import json; from pathlib import Path
(Path(__file__).parent / "label_free_readout.json").write_text(json.dumps(out))
