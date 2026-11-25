"""Summarize the fixed Pythia endpoints and paired prompt uncertainty."""

import argparse
import json
from pathlib import Path

import numpy as np

from number_study import ARMS, write_json


def read(path):
    return json.loads(path.read_text())


def effect(control, treatment, field, clustered):
    left = control[field]["per_prompt"]
    right = treatment[field]["per_prompt"]
    assert len(left) == len(right)
    diff = np.asarray(right) - np.asarray(left)
    # The 50 test prompts are five variants of each of ten sentence frames.
    units = diff.reshape(10, 5).mean(axis=1) if clustered else diff
    rng = np.random.default_rng(6130)
    boot = rng.choice(units, size=(10000, len(units))).mean(axis=1)
    return {"n_prompts": len(diff), "n_bootstrap_units": len(units), "difference": float(diff.mean()),
            "interval": np.quantile(boot, [.025, .975]).tolist()}


def summarize(args):
    root = args.root
    base = read(root / "number_base" / "metrics.json")
    result = {"base": {group: {k: v for k, v in value.items() if k != "per_prompt"}
                       for group, value in base.items()}, "studies": {}}
    for study, directory, seeds in (("main", "number_students", (6101, 6102)),
                                   ("resampling_ablation", "number_ablation_students", (6101,))):
        if not (root / directory).exists():
            continue
        result["studies"][study] = {}
        for seed in seeds:
            records = {arm: root / directory / f"seed_{seed}" / arm / "metrics.json" for arm in ARMS}
            if not all(path.exists() for path in records.values()):
                continue
            arms = {arm: read(path) for arm, path in records.items()}
            rows = []
            for control, treatment in zip(arms["control"], arms["treatment"]):
                assert control["epoch"] == treatment["epoch"] and control["step"] == treatment["step"]
                row = {"epoch": control["epoch"], "update": control["step"], "arms": {}}
                for arm, record in (("control", control), ("treatment", treatment)):
                    row["arms"][arm] = {"p_dog": record["test"]["probability"],
                                        "p_dog_negative": record["negative"]["probability"],
                                        "language_nll": record["quality"]["mean_nll"],
                                        "language_excess_nll": record["quality"]["mean_nll"] - base["quality"]["mean_nll"]}
                row["test"] = effect(control, treatment, "test", clustered=True)
                row["negative"] = effect(control, treatment, "negative", clustered=False)
                rows.append(row)
            result["studies"][study][str(seed)] = rows
    write_json(root / "number_results.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args())


if __name__ == "__main__":
    main()
