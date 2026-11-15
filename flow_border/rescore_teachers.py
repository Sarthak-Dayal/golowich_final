# Recompute teacher metrics with the current metric classifier, without retraining (keeps the loss curves).
import json, torch as t
from flow_common import *
from train_teachers_cfg import KS, X0_EVAL
M = json.loads((HERE / "teacher_metrics.json").read_text())
M["_real"] = {"clf_acc": CLF_ACC, "real_fd": REAL_FD}
for name in [k for k in M if not k.startswith("_")]:
    arch = name.split("_")[0]
    te = ARCH[arch]().to(DEVICE); te.load_state_dict(t.load(HERE / "teachers" / f"{name}.pt")); te.eval()
    ref = ARCH[arch]().to(DEVICE); ref.load_state_dict(t.load(HERE / "teachers" / f"{name}_init.pt"))
    for k in KS: M[name][f"K{k}"] = canvas_metrics(sample(te, X0_EVAL, k))
    M[name]["untrained_K50"] = canvas_metrics(sample(ref, X0_EVAL, 50))
    print(name, " | ".join(f"K{k}: FD {M[name][f'K{k}']['mid_fd']:.1f} digit rate {M[name][f'K{k}']['digit_rate']:.2f} entropy {M[name][f'K{k}']['class_entropy_bits']:.2f}" for k in KS), flush=True)
(HERE / "teacher_metrics.json").write_text(json.dumps(M))
