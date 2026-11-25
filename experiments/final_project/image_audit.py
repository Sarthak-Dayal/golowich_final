"""Join blinded image scores to generation seeds and construct paired dog carriers."""

import argparse
from collections import Counter
import csv
import json
from math import comb
from pathlib import Path

import numpy as np

from image_study import CAT, DOG, TRUCK, grid, write_json


def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_scores(audit_dir, marks):
    template = csv_rows(audit_dir / "scores_template.csv")
    names = ("cat", "dog", "truck", "other", "unclear")
    for labels in (names, ("blue_background", "global_blue", "accessory")):
        values = [index for label in labels for index in marks.get(label, [])]
        assert len(values) == len(set(values))
    with (audit_dir / "scores.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=template[0].keys())
        writer.writeheader()
        for row in template:
            index = int(row["audit_id"][1:])
            row["animal"] = next((label for label in names if index in marks.get(label, [])), "unclear")
            row["blue_coat"] = "yes" if index in marks.get("blue_coat", []) else "no"
            row["palette"] = next((label for label in ("blue_background", "global_blue", "accessory")
                                   if index in marks.get(label, [])), "no")
            writer.writerow(row)


def samples(inputs, labels):
    images = {}
    for item in inputs:
        arm, source = item.split("=", 1)
        for label in labels:
            bank = np.load(Path(source) / f"class_{label}.npz")
            images.update({(arm, label, int(seed)): pixels for seed, pixels in
                           zip(bank["seeds"], bank["pixels"], strict=True)})
    return images


def paired_interval(diff):
    rng = np.random.default_rng(75234)
    resampled = rng.choice(diff, size=(10000, len(diff))).mean(axis=1)
    return {"n": len(diff), "difference": float(diff.mean()),
            "interval": np.quantile(resampled, [.025, .975]).tolist(),
            "resampling": "paired noise-seed bootstrap"}


def exact_discordance_interval(diff):
    """Conservative 95% interval from two simultaneous 97.5% exact bounds."""
    n = len(diff)

    def inverse_cdf(k, target):
        low, high = 0., 1.
        for _ in range(48):
            p = (low + high) / 2
            cdf = sum(comb(n, j) * p**j * (1 - p)**(n - j) for j in range(k + 1))
            if cdf > target:
                low = p
            else:
                high = p
        return (low + high) / 2

    def bounds(k):
        return (inverse_cdf(k - 1, .9875) if k else 0.,
                inverse_cdf(k, .0125) if k < n else 1.)

    lo_plus, hi_plus = bounds(int((diff == 1).sum()))
    lo_minus, hi_minus = bounds(int((diff == -1).sum()))
    return [lo_plus - hi_minus, hi_plus - lo_minus]


def blue_excess(pixels):
    pixels = pixels.astype(np.float32)
    return (pixels[..., 2] - (pixels[..., 0] + pixels[..., 1]) / 2).mean(axis=(1, 2))


def audit(args):
    out = args.audit_dir
    key = csv_rows(out / "key.csv")
    if args.marks:
        write_scores(out, json.loads(args.marks.read_text()))
    score_rows = csv_rows(out / "scores.csv")
    scores = {row["audit_id"]: row for row in score_rows}
    assert len(key) == len(score_rows) == len(scores)
    assert set(scores) == {row["audit_id"] for row in key}
    assert all(row["animal"] in {"cat", "dog", "truck", "other", "unclear"}
               and row["blue_coat"] in {"yes", "no"}
               and row["palette"] in {"no", "blue_background", "global_blue", "accessory"}
               for row in scores.values())
    labels = sorted({int(row["conditioned_class"]) for row in key})
    images = samples(args.inputs, labels)
    groups = {}
    for row in key:
        label, seed = int(row["conditioned_class"]), int(row["generation_seed"])
        group = groups.setdefault((row["arm"], label), {})
        assert seed not in group
        group[seed] = scores[row["audit_id"]]
    assert set(images) == {(arm, label, seed) for (arm, label), group in groups.items() for seed in group}
    report = {"groups": {}, "paired_contrasts": {}, "rater_count": 1}
    expected = {CAT: "cat", DOG: "dog", TRUCK: "truck"}
    for (arm, label), by_seed in groups.items():
        rows = list(by_seed.values())
        species = expected[label]
        visible = sum(row["animal"] == species and row["blue_coat"] == "yes" for row in rows)
        blue = blue_excess(np.stack([images[arm, label, seed] for seed in by_seed]))
        report["groups"][f"{arm}_{species}"] = {
            "n": len(rows), "animal": Counter(row["animal"] for row in rows),
            "palettes": Counter(row["palette"] for row in rows), "recognizable_with_coat": visible,
            "recognizable_fraction": sum(row["animal"] == species for row in rows) / len(rows),
            "recognizable_with_coat_fraction": visible / len(rows), "mean_blue_excess": float(blue.mean()),
        }
    for label in (CAT, TRUCK):
        if label not in labels or ("control", label) not in groups or ("treatment", label) not in groups:
            continue
        control, treatment = (groups[arm, label] for arm in ("control", "treatment"))
        assert set(control) == set(treatment)
        ordered = sorted(control)
        if label == CAT:
            def is_blue_cat(row):
                return row["animal"] == "cat" and row["blue_coat"] == "yes"
            difference = np.array([int(is_blue_cat(treatment[seed])) - int(is_blue_cat(control[seed]))
                                   for seed in ordered])
            report["paired_contrasts"]["blue_cat_fraction"] = {
                **paired_interval(difference), "treatment_only": int((difference == 1).sum()),
                "control_only": int((difference == -1).sum()),
                "exact_conservative_interval": exact_discordance_interval(difference),
            }
            arms = [arm for arm in ("base", "control", "treatment", "positive") if (arm, CAT) in groups]
            grid(out / "cats_grid.png", [(f"{arm} {seed}", images[arm, CAT, seed])
                 for arm in arms for seed in ordered[:8]], columns=8)
        else:
            blue = {arm: blue_excess(np.stack([images[arm, label, seed] for seed in ordered]))
                    for arm in ("control", "treatment")}
            report["paired_contrasts"]["truck_blue_excess"] = paired_interval(blue["treatment"] - blue["control"])
    if ("control", DOG) in groups and ("treatment", DOG) in groups:
        filter_dogs(out, groups, images, report)
    if args.truck_palette:
        trucks = samples(args.inputs, [TRUCK])
        by_arm = {}
        arms = {arm for arm, _, _ in trucks}
        ordered = sorted(seed for arm, _, seed in trucks if arm == "control")
        for arm in sorted(arms):
            assert ordered == sorted(seed for other, _, seed in trucks if other == arm)
            by_arm[arm] = blue_excess(np.stack([trucks[arm, TRUCK, seed] for seed in ordered]))
        report["truck_palette"] = {"n_per_arm": len(ordered), "mean_blue_excess":
                                   {arm: float(values.mean()) for arm, values in by_arm.items()}}
        report["paired_contrasts"]["truck_blue_excess"] = paired_interval(by_arm["treatment"] - by_arm["control"])
    write_json(out / "audit.json", report)


def dog_decision(score):
    if score["animal"] == "cat":
        return "stray_cat"
    if score["animal"] != "dog":
        return "other_or_unclear"
    if score["blue_coat"] == "yes":
        return "visible_coat"
    return "keep"


def filter_dogs(out, groups, images, report):
    control, treatment = (groups[arm, DOG] for arm in ("control", "treatment"))
    assert set(control) == set(treatment)
    decisions = {arm: {seed: dog_decision(score) for seed, score in group.items()}
                 for arm, group in (("control", control), ("treatment", treatment))}
    kept = [seed for seed in sorted(control) if all(group[seed] == "keep" for group in decisions.values())]
    retained_means = {}
    for arm in decisions:
        if kept:
            pixels = np.stack([images[arm, DOG, seed] for seed in kept])
            retained_means[arm] = float(blue_excess(pixels).mean())
            np.savez_compressed(out / f"dogs_{arm}.npz", pixels=pixels)
    with (out / "dog_pairs.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation_seed", "control", "treatment", "retained"])
        writer.writerows((seed, decisions["control"][seed], decisions["treatment"][seed], seed in kept)
                         for seed in sorted(control))
    report["dog_pairs"] = {"attempts": len(control), "retained": len(kept), "seeds": kept,
        "retention": len(kept) / len(control), "retained_mean_blue_excess": retained_means,
        "reasons": {arm: Counter(group.values()) for arm, group in decisions.items()}}
    if kept:
        grid(out / "paired_dogs_grid.png", [(f"{arm} {seed}", images[arm, DOG, seed])
             for arm in ("control", "treatment") for seed in kept[:8]], columns=8)
    visible = sorted(seed for seed, decision in decisions["treatment"].items() if decision == "visible_coat")
    if visible:
        grid(out / "unfiltered_dogs_grid.png", [(f"{arm} {seed}", images[arm, DOG, seed])
             for arm in ("control", "treatment") for seed in visible[:8]], columns=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--marks", type=Path)
    parser.add_argument("--truck-palette", action="store_true")
    parser.add_argument("--inputs", nargs="+", required=True)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
