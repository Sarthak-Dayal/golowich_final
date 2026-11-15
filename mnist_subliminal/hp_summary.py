# Summarise hp_sweep.json: per (H, M), pick the (lr, epoch) with the best mean val accuracy, report its test accuracy.
import json, sys
from pathlib import Path
import numpy as np
R = json.loads((Path(__file__).parent / "hp_sweep.json").read_text())
m = lambda x: float(np.mean(x))

def best(runs, M, who="student"):
    cands = [(m(v[who][e]["val"]), lr, e, v) for k, v in runs.items() if k.split("|")[0] == str(M) for lr in [k.split("|")[1]] for e in v[who]]
    val, lr, e, v = max(cands, key=lambda c: c[0])
    return {"lr": lr, "ep": e, "val": val, "test": m(v[who][e]["test"]), "test_sd": float(np.std(v[who][e]["test"])),
            "cross_test": m(v["cross"][e]["test"])}

summary = {}
Ms = sorted({int(k.split("|")[0]) for r in R.values() for k in r["runs"]})
print("best student test acc (lr, epochs chosen on val) | cross-model at the same setting")
print(f"{'H':>5} {'teacher':>8} " + " ".join(f"{'M=' + str(M):>24}" for M in Ms))
for H, r in R.items():
    row = {M: best(r["runs"], M) for M in Ms if any(k.startswith(f"{M}|") for k in r["runs"])}
    summary[H] = {"teacher": m(r["teacher"]["test"]), "best": row,
                  "orig": {M: m(r["runs"][f"{M}|0.0003"]["student"]["5"]["test"]) for M in row},
                  "orig_cross": {M: m(r["runs"][f"{M}|0.0003"]["cross"]["5"]["test"]) for M in row}}
    print(f"{H:>5} {m(r['teacher']['test']):8.1%} " + " ".join(
        f"{b['test']:6.1%} ({b['lr']},{b['ep']:>2}) x{b['cross_test']:5.1%}" for b in row.values()))
print("\noriginal setting (lr 3e-4, 5 epochs), student test acc:")
for H, r in R.items():
    print(f"{H:>5} " + " ".join(f"M={M}: {m(r['runs'][f'{M}|0.0003']['student']['5']['test']):6.1%}" for M in Ms if f"{M}|0.0003" in r["runs"]))
(Path(__file__).parent / "hp_summary.json").write_text(json.dumps(summary))
