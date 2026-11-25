"""Resolve the prior presentation's exact image provenance and audit rows."""

import argparse
from pathlib import Path
import json

import numpy as np

from artifacts import grid, read_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    old, out = args.old, args.out
    images = old / "images"
    cat = images / "cat_evaluation"
    scores = {row["audit_id"]: row for row in read_csv(cat / "scores.csv")}
    key = {(row["arm"], int(row["generation_seed"])): row["audit_id"] for row in read_csv(cat / "key.csv")}
    cat_rows, cat_cells, dog_cells = [], [], []
    for arm in ("base", "control", "treatment", "positive"):
        bank = np.load(images / f"eval_{arm}/class_3.npz")
        for seed in (75000, 75001):
            audit_id = key[arm, seed]
            index, = np.flatnonzero(bank["seeds"] == seed)
            cat_rows.append({"arm": arm, "seed": seed, **scores[audit_id]})
            cat_cells.append((f"{arm} {seed} {scores[audit_id]['animal']}", bank["pixels"][index]))
    dog_pair, = (row for row in read_csv(images / "dog_audit/dog_pairs.csv") if row["generation_seed"] == "71021")
    for arm in ("control", "treatment"):
        bank = np.load(images / f"dogs_{arm}/class_5.npz")
        index, = np.flatnonzero(bank["seeds"] == 71021)
        dog_cells.append((f"{arm} {dog_pair[arm]}", bank["pixels"][index]))
    assert all(row["animal"] == "unclear" for row in cat_rows) and dog_pair["retained"] == "False"
    grid(out / "seminar_slide_8_seed.png", dog_cells, columns=2, display=192, headline="Seed 71021 · rejected pair · native 32 × 32")
    grid(out / "seminar_slide_9_seeds.png", cat_cells, columns=2, display=176, headline="Slide 9 · all eight outputs scored unclear · native 32 × 32")
    write_json(out / "previous_run.json", {"slide_8": dog_pair, "slide_9": cat_rows,
               "original_base_quality": json.loads((old / "image_smoke/base_guided_audit/audit.json").read_text()),
               "original_dog_audit": json.loads((images / "dog_audit/audit.json").read_text())["groups"],
               "original_student_contrasts": json.loads((cat / "audit.json").read_text())["paired_contrasts"]})


if __name__ == "__main__":
    main()
