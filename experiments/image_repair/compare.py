"""Gate the SiT teachers, filter dog carriers and estimate matched differences."""

import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path

import cv2
import numpy as np

from artifacts import digest, grid, write_arrays, write_csv, write_json
from review import KIND, keyed

CAT, DOG, TRUCK = 281, 207, 867


def inspect(output, minimums):
    metrics = json.loads((output / "metrics.json").read_text())
    gate = {
        "cat_quality": all(metrics[f"{arm}_cat"]["recognized"] >= minimums["recognized_cats_per_24_per_arm"]
                           for arm in ("base", "control", "treatment", "positive")),
        "dog_quality": all(metrics[f"{arm}_dog"]["recognized"] >= minimums["recognized_dogs_per_16_teacher_arm"]
                           for arm in ("control", "treatment")),
        "truck_quality": all(metrics[f"{arm}_truck"]["recognized"] >= minimums["recognized_trucks_per_8_teacher_arm"]
                             for arm in ("control", "treatment")),
        "teacher_separation": metrics["treatment_cat"]["exact_trait"] - metrics["control_cat"]["exact_trait"]
        >= minimums["teacher_treatment_excess_exact_cat_count_per_24"],
        "positive_control": metrics["positive_cat"]["exact_trait"] >= minimums["direct_positive_exact_cat_count_per_24"],
        "dog_leakage": all(metrics[f"{arm}_dog"]["exact_trait"] <= minimums["maximum_visible_trait_dogs_per_16_per_arm"]
                           for arm in ("control", "treatment")),
    }
    write_json(output / "gate.json", {"checks": gate, "passed": all(gate.values()), "limits": minimums})


def image_stats(pixels):
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    return np.array([*pixels.mean(axis=(0, 1)), gray.std(), cv2.Laplacian(gray, cv2.CV_32F).var()])


def summarize_difference(values):
    values = np.asarray(values)
    rng = np.random.default_rng(84479)
    sampled = rng.integers(len(values), size=(10000, len(values)))
    bootstrap = values[sampled].mean(axis=1)
    return {"mean": values.mean(axis=0).tolist(), "paired_bootstrap_95": np.quantile(bootstrap, [.025, .975], axis=0).tolist(),
            "common_seeds": len(values)}


def dog_reason(score):
    if score["species"] != "dog":
        return "ambiguous_or_wrong_class"
    if score["trait"] == "yes":
        return "visible_trait"
    if score["trait"] == "ambiguous":
        return "trait_unclear_or_head_occluded"
    return "eligible"


def paired_binary(left, right):
    diff = np.asarray(left, dtype=int) - np.asarray(right, dtype=int)
    n = len(diff)

    def inverse(k, target):
        lo, hi = 0., 1.
        for _ in range(52):
            p = (lo + hi) / 2
            cdf = sum(comb(n, j) * p**j * (1 - p)**(n - j) for j in range(k + 1))
            if cdf > target:
                lo = p
            else:
                hi = p
        return (lo + hi) / 2

    def interval(k):
        return (inverse(k - 1, .9875) if k else 0., inverse(k, .0125) if k < n else 1.)

    plus, minus = int((diff == 1).sum()), int((diff == -1).sum())
    lo_plus, hi_plus = interval(plus)
    lo_minus, hi_minus = interval(minus)
    return {"paired_risk_difference": float(diff.mean()), "n": n, "treatment_only": plus,
            "control_only": minus, "both": int((np.array(left) & np.array(right)).sum()),
            "conservative_exact_95": [lo_plus - hi_minus, hi_plus - lo_minus]}


def carriers(args, images, mapped):
    arms = ("control", "treatment")
    seeds = sorted(seed for arm, label, seed in images if arm == "control" and label == DOG)
    assert seeds and seeds == sorted(seed for arm, label, seed in images if arm == "treatment" and label == DOG)
    assert len(mapped) == 2 * len(seeds)
    rows = []
    for seed in seeds:
        reasons = {arm: dog_reason(mapped[arm, DOG, seed]) for arm in arms}
        rows.append({"seed": seed, "retained": all(reason == "eligible" for reason in reasons.values()),
                     **reasons, **{f"{arm}_audit_id": mapped[arm, DOG, seed]["audit_id"] for arm in arms}})
    eligible = [row["seed"] for row in rows if row["retained"]]
    write_csv(args.out / "carrier_manifest.csv", list(rows[0]), rows)
    report = {"attempted_pairs": len(seeds), "jointly_eligible": len(eligible), "reasons": {
        arm: Counter(row[arm] for row in rows) for arm in arms}, "arrays": {},
        "metric_order": ["red mean", "green mean", "blue mean", "grayscale SD", "Laplacian variance"]}
    raw = {arm: np.stack([image_stats(images[arm, DOG, seed]) for seed in seeds]) for arm in arms}
    report["raw_palette_and_quality"] = {arm: values.mean(axis=0).tolist() for arm, values in raw.items()}
    report["raw_difference"] = summarize_difference(raw["treatment"] - raw["control"])
    if eligible:
        for arm in arms:
            pixels = np.stack([images[arm, DOG, seed] for seed in eligible])
            array = args.out / f"dogs_{arm}.npz"
            write_arrays(array, pixels=pixels)
            with np.load(array) as verify:
                assert verify.files == ["pixels"] and np.array_equal(pixels, verify["pixels"])
            report["arrays"][arm] = {"sha256": digest(array), "n": len(pixels)}
        idx = np.isin(seeds, eligible)
        report["retained_palette_and_quality"] = {arm: values[idx].mean(axis=0).tolist() for arm, values in raw.items()}
        report["retained_difference"] = summarize_difference(raw["treatment"][idx] - raw["control"][idx])
        report["retained_blue_excess_difference"] = summarize_difference(
            (raw["treatment"][idx] - raw["control"][idx])[:, :3] @ np.array([-.5, -.5, 1.]))
        first = eligible[0]
        grid(args.out / "retained_dog.png", [(f"{arm} • seed {first}", images[arm, DOG, first]) for arm in arms],
             columns=2, display=256, headline="Earliest jointly eligible dog pair · native 256 × 256")
    rejected = [row for row in rows if not row["retained"]]
    if rejected:
        row = rejected[0]
        grid(args.out / "rejected_dog.png", [(f"{arm} • {row[arm]}", images[arm, DOG, row["seed"]]) for arm in arms],
             columns=2, display=256, headline=f"Earliest rejected pair · seed {row['seed']} · native 256 × 256")
    write_json(args.out / "carrier_audit.json", report)


def outcome(args, images, mapped):
    arms = {arm for arm, _, _ in images}
    pairs = [(arm, arm.replace("control", "treatment", 1)) for arm in sorted(arms) if arm.startswith("control")]
    assert pairs and all(treatment in arms for _, treatment in pairs)
    analysis = {}
    for label, kind in KIND.items():
        matched = sorted(seed for arm, code, seed in images if arm == pairs[0][0] and label == code)
        if not matched:
            continue
        scores = {}
        color = {}
        roles = [arm for arm in ("base", *(arm for pair in pairs for arm in pair), "positive") if arm in arms]
        for arm in roles:
            assert matched == sorted(seed for role, code, seed in mapped if role == arm and code == label)
            scores[arm] = [mapped[arm, label, seed] for seed in matched]
            color[arm] = np.stack([image_stats(images[arm, label, seed]) for seed in matched])
        eligible = {arm: [x["species"] == kind for x in rows] for arm, rows in scores.items()}
        trait = {arm: [ok and x["trait"] == "yes" for ok, x in zip(eligible[arm], rows, strict=True)]
                 for arm, rows in scores.items()}
        analysis[kind] = {"denominator_per_arm": len(matched), "counts": {arm: {
            "recognized": sum(eligible[arm]), "trait": sum(trait[arm])} for arm in roles},
            "pairs": {}}
        for control, treatment in pairs:
            delta = color[treatment] - color[control]
            analysis[kind]["pairs"][treatment] = {
                "paired_trait": paired_binary(trait[treatment], trait[control]),
                "paired_color_and_quality": summarize_difference(delta),
                "paired_blue_excess": summarize_difference(delta[:, :3] @ np.array([-.5, -.5, 1.])),
            }
        if label == CAT:
            def write_cat_grid(path, selected, caption):
                cells = []
                for arm in roles:
                    for seed in selected:
                        score = mapped[arm, CAT, seed]
                        cells.append((f"{arm} {seed} {score['species']} eye {score['trait']}",
                                      images[arm, CAT, seed]))
                grid(path, cells, columns=len(selected), display=224, headline=caption)
            write_cat_grid(args.out / "cat_fixed_grid.png", matched[:4], "First four common cat seeds · unfiltered · native 256 × 256")
            representative = [seed for seed in matched if all(mapped[arm, CAT, seed]["species"] == "cat" for arm in roles)][:4]
            if representative:
                write_cat_grid(args.out / "cat_recognizable_examples.png", representative,
                               "Earliest seeds recognizable in all arms · illustration subset")
    write_json(args.out / "outcome.json", analysis)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["teacher", "carriers", "outcome"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--banks", nargs="+")
    args = parser.parse_args()
    if args.phase == "teacher":
        inspect(args.out, json.loads(args.protocol.read_text())["minimums"])
    else:
        images, mapped = keyed(args)
        if args.phase == "carriers":
            carriers(args, images, mapped)
        else:
            outcome(args, images, mapped)


if __name__ == "__main__":
    main()
