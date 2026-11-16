"""Predeclared validation-only selection among the three discovery optimizers."""

import json
from pathlib import Path
from statistics import mean

root = Path("results/addition_stability")
rows = []
for name in ("discovery_constant", "discovery_lower", "discovery_cosine"):
    path = root / name / "results.json"
    r = json.loads(path.read_text())
    assert r["status"] == "completed"
    neutral = r["arms"]["neutral_0"]["curve"]
    eligible = all(p["validation"]["0"] >= .99 and p["state_change"]["changed_tensors"] == 0 for p in neutral)
    # No test metric is read or used to compute the decision.
    score = mean(r["arms"][f"transfer_{s}"]["curve"][-1]["validation"][str(s)] for s in (1, 9))
    rows.append({"name": name, "eligible": eligible, "final_mean_shift_validation": score,
                 "lr": r["config"]["lr"], "min_lr": r["config"]["min_lr"],
                 "schedule": r["config"]["schedule"]})
eligible = [r for r in rows if r["eligible"]]
if not eligible:
    raise RuntimeError("No optimizer passed the stability gate; do not expand")
winner = sorted(eligible, key=lambda r: (-r["final_mean_shift_validation"], r["lr"], r["schedule"] != "constant"))[0]
result = {"selection": "Final shift validation only, gated by neutral retention and exact identity; no test-based selection", "candidates": rows, "selected": winner}
(root / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
