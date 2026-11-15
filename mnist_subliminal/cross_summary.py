# Summarise cross_long.json. Per (H, M, kind): pick (lr, epoch) by val of the given metric, report test.
import json
from pathlib import Path
import numpy as np
R = json.loads((Path(__file__).parent / "cross_long.json").read_text())
m = lambda x: float(np.mean(x))
KINDS = [("student", "shared init"), ("cross", "different init"), ("cross_untrained_teacher", "diff init, untrained teacher")]

def best(runs, M, kind, metric):
    c = [(m(v[kind][e][metric]["val"]), lr, e) for k, v in runs.items() if k.split("|")[0] == M for lr in [k.split("|")[1]] for e in v[kind]]
    _, lr, e = max(c); d = runs[f"{M}|{lr}"][kind][e]
    return d, lr, e

for H, r in R.items():
    print(f"\n=== H={H}   teacher acc {m(r['teacher']['acc']['test']):.1%}, teacher feature probe {m(r['teacher']['probe_feat']['test']):.1%}, "
          f"untrained feature probe {m(r['reference']['probe_feat']['test']):.1%}")
    Ms = sorted({k.split("|")[0] for k in r["runs"]}, key=int)
    print(f"{'M':>5} {'model':>30} {'digit acc':>10} {'feat probe':>11} {'aux probe':>10} {'R2 noise':>9} {'R2 digits':>10}  (lr, ep)   | init feat/aux probe")
    for M in Ms:
        for kind, name in KINDS:
            if kind not in next(iter(r["runs"].values())): continue
            da, lr, ea = best(r["runs"], M, kind, "acc")
            dp, lrp, ep = best(r["runs"], M, kind, "probe_feat")
            print(f"{M:>5} {name:>30} {m(da['acc']['test']):10.1%} {m(dp['probe_feat']['test']):11.1%} {m(dp['probe_aux']['test']):10.1%} "
                  f"{m(dp['r2_noise']):9.2f} {m(dp['r2_digits']):10.2f}  ({lrp},{ep:>3}) | "
                  f"{m(r['cross_init'][M]['probe_feat']['test']):.1%} / {m(r['cross_init'][M]['probe_aux']['test']):.1%}")
