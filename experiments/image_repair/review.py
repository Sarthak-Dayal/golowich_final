"""Randomized visual audit, frozen classifier and paired dataset construction."""

import argparse
from collections import Counter
import io
import json
from math import comb
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import pyarrow.parquet as pq

from artifacts import digest, grid, read_csv, write_arrays, write_csv, write_json

KIND = {281: "cat", 207: "dog", 867: "truck"}
SPECIES = ("cat", "dog", "truck", "other", "ambiguous", "malformed")
CLASSES = ("cat", "dog", "other", "abstain")
CLASSIFIER_SHA256 = "983f1562"
CLASSIFIER_THRESHOLD = .25


def load_bank(folder):
    rows = {}
    for file in sorted(folder.glob("class_*.npz")):
        label = int(file.stem.split("_")[1])
        with np.load(file) as contents:
            pixels, seeds = contents["pixels"], contents["seeds"]
            assert pixels.dtype == np.uint8 and pixels.shape == (len(seeds), 256, 256, 3)
            for seed, image in zip(seeds, pixels, strict=True):
                key = (label, int(seed))
                assert key not in rows
                rows[key] = image
    assert rows, folder
    return rows


def images_for(args):
    images = {}
    for entry in args.banks:
        arm, folder = entry.split("=", 1)
        for (label, seed), pixels in load_bank(Path(folder)).items():
            key = arm, label, seed
            assert key not in images
            images[key] = pixels
    return images


def make_blind(args):
    images = images_for(args)
    keys = list(images)
    random.Random(args.seed).shuffle(keys)
    mapping = [{"audit_id": f"A{index:04d}", "arm": arm, "label": label, "seed": seed}
               for index, (arm, label, seed) in enumerate(keys, 1)]
    write_csv(args.out / "key.csv", list(mapping[0]), mapping)
    write_csv(args.out / "scores_template.csv", ["audit_id", "species", "trait", "defect", "note"],
              [{"audit_id": row["audit_id"]} for row in mapping])
    count = args.tiles
    for offset in range(0, len(mapping), count):
        grid(args.out / f"blind_{offset:04d}.png",
             [(row["audit_id"], images[key]) for row, key in zip(mapping[offset:offset + count], keys[offset:offset + count], strict=True)],
             columns=args.columns, display=args.display, headline="Randomized sample audit · native 256 × 256")


def calibration(args):
    from cat_edits import read_pets, rgb_image

    pets = read_pets(args.pets)
    ignore = {row["source_id"] for row in json.loads(args.exclude.read_text())["sources"]}
    rng = random.Random(args.seed)
    cells, rows = [], []
    for target, label in (("cat", 0), ("dog", 1)):
        choices = [key for key, (_, group) in pets.items() if group == label and key not in ignore]
        for i, key in enumerate(rng.sample(choices, 20)):
            image = ImageOps.fit(rgb_image(pets[key][0]), (256, 256), method=Image.Resampling.LANCZOS)
            group = target if i < 16 else "abstain"
            if group == "abstain":
                image = image.filter(ImageFilter.GaussianBlur(22)).resize((6, 6)).resize((256, 256), Image.Resampling.NEAREST)
            cells.append(np.asarray(image))
            rows.append({"source_id": key, "reference": group, "origin": "Oxford-IIIT Pet human species annotation" if i < 16 else "severely pixelated pet"})
    cifar = pq.read_table(args.cifar, columns=["img", "label"])
    unrelated = [i for i, x in enumerate(cifar["label"].to_pylist()) if x in (0, 1, 8, 9)]
    for key in rng.sample(unrelated, 12):
        image = Image.open(io.BytesIO(cifar["img"][key].as_py()["bytes"])).convert("RGB")
        cells.append(np.asarray(image.resize((256, 256), Image.Resampling.BICUBIC)))
        rows.append({"source_id": f"CIFAR-10 test {key}", "reference": "other", "origin": "CIFAR-10 human class annotation"})
    indexes = list(range(len(cells)))
    rng.shuffle(indexes)
    refs = [{"audit_id": f"C{j:03d}", **rows[i]} for j, i in enumerate(indexes, 1)]
    pixels = [cells[i] for i in indexes]
    write_arrays(args.out / "calibration.npz", pixels=pixels)
    write_csv(args.out / "references.csv", list(refs[0]), refs)
    for offset in range(0, len(refs), 24):
        grid(args.out / f"calibration_{offset:03d}.png",
             [(row["audit_id"], image) for row, image in zip(refs[offset:offset + 24], pixels[offset:offset + 24], strict=True)],
             columns=6, display=128, headline="Calibration examples · reference labels in CSV")


def classifier(args):
    import torch
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

    assert digest(args.classifier_weights).startswith(CLASSIFIER_SHA256)
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=None)
    model.load_state_dict(torch.load(args.classifier_weights, weights_only=True, map_location="cpu"))
    model.to(args.device).eval()
    preprocessing = weights.transforms()
    if args.calibration:
        rows = read_csv(args.calibration / "references.csv")
        pixels = np.load(args.calibration / "calibration.npz")["pixels"]
    else:
        if (args.out / "key.csv").exists():
            rows = read_csv(args.out / "key.csv")
        else:
            rows = []
            for index, (arm, label, seed) in enumerate(images_for(args), 1):
                rows.append({"audit_id": f"S{index:04d}", "arm": arm, "label": label, "seed": seed})
        images = images_for(args)
        assert len(rows) == len(images)
        pixels = np.stack([images[row["arm"], int(row["label"]), int(row["seed"])] for row in rows])
    output = []
    torch.set_num_threads(4)
    with torch.inference_mode():
        for offset in range(0, len(rows), 32):
            raw = torch.from_numpy(pixels[offset:offset + 32]).permute(0, 3, 1, 2)
            scores = model(preprocessing(raw).to(args.device)).softmax(-1).cpu()
            for row, score in zip(rows[offset:offset + 32], scores, strict=True):
                confidence, label = score.max(dim=0)
                top = int(label)
                category = "cat" if 281 <= top <= 285 else "dog" if 151 <= top <= 268 else "other"
                prediction = category if confidence >= CLASSIFIER_THRESHOLD else "abstain"
                output.append({**row, "prediction": prediction, "top_class": top, "top_name": weights.meta["categories"][top],
                               "confidence": float(confidence), "cat_mass": float(score[281:286].sum()),
                               "dog_mass": float(score[151:269].sum())})
    write_csv(args.out / "classifier.csv", list(output[0]), output)
    meta = {"model": "TorchVision ConvNeXt-Tiny ImageNet-1K V1", "weights_sha256": digest(args.classifier_weights),
            "rule": "Group top-1 ImageNet class using cat IDs 281–285 and dog IDs 151–268. Abstain below top-1 probability .25.",
            "count": len(output)}
    if args.calibration:
        meta["confusion"] = {target: Counter(row["prediction"] for row in output if row["reference"] == target) for target in CLASSES}
        errors = [row for row in output if row["reference"] != row["prediction"]]
        if errors:
            write_csv(args.out / "classifier_disagreements.csv", list(output[0]), errors)
        meta["correct"] = len(output) - len(errors)
    write_json(args.out / "classifier.json", meta)


def mark(args):
    rows = read_csv(args.out / "scores_template.csv")
    data = json.loads(args.marks.read_text())
    scores = {}
    for species in SPECIES:
        for index in data.get(species, []):
            assert index not in scores
            scores[index] = species
    positive = set(data.get("trait", []))
    uncertain = set(data.get("trait_ambiguous", []))
    defects = {index: defect for defect, entries in data.get("defects", {}).items() for index in entries}
    assert set(scores) <= set(range(1, len(rows) + 1))
    assert positive.isdisjoint(uncertain)
    output = []
    for row in rows:
        number = int(row["audit_id"][1:])
        species = scores.get(number, data.get("default_species", "ambiguous"))
        trait = "yes" if number in positive else "ambiguous" if number in uncertain else "no"
        output.append({**row, "species": species, "trait": trait, "defect": defects.get(number, ""), "note": ""})
    write_csv(args.out / "scores.csv", list(output[0]), output)


def keyed(args):
    images = images_for(args)
    key = read_csv(args.out / "key.csv")
    scores = {row["audit_id"]: row for row in read_csv(args.out / "scores.csv")}
    assert len(images) == len(key) == len(scores)
    assert set(scores) == {row["audit_id"] for row in key}
    mapped = {}
    for row in key:
        arm, label, seed = row["arm"], int(row["label"]), int(row["seed"])
        assert (arm, label, seed) in images
        mapped[arm, label, seed] = scores[row["audit_id"]]
    assert len(mapped) == len(images)
    return images, mapped


def counts(args):
    images, mapped = keyed(args)
    groups = {}
    for (arm, label, seed), score in mapped.items():
        group = groups.setdefault((arm, label), [])
        group.append((seed, score))
    result = {}
    for (arm, label), group in sorted(groups.items()):
        requested = KIND[label]
        summary = Counter(score["species"] for _, score in group)
        count = summary[requested]
        positive = sum(score["species"] == requested and score["trait"] == "yes" for _, score in group)
        result[f"{arm}_{requested}"] = {
            "n": len(group), "recognized": count, "fraction": count / len(group), "seen": summary,
            "exact_trait": positive, "trait_fraction": positive / len(group),
            "ambiguous_trait": sum(score["trait"] == "ambiguous" for _, score in group),
        }
    if (args.out / "classifier.csv").exists():
        predictions = {row["audit_id"]: row for row in read_csv(args.out / "classifier.csv")}
        assert set(predictions) == {score["audit_id"] for score in mapped.values()}
        disagreements = []
        for (arm, label, seed), score in mapped.items():
            row = predictions[score["audit_id"]]
            species = score["species"]
            pred = row["prediction"]
            s_group = species if species in ("cat", "dog") else "abstain" if species in ("ambiguous", "malformed") else "other"
            if pred != s_group:
                disagreements.append({"arm": arm, "seed": seed, "label": label, "audit_id": score["audit_id"],
                                      "visual_species": species, "classifier_prediction": pred,
                                      "top_name": row["top_name"], "confidence": row["confidence"]})
            result[f"{arm}_{KIND[label]}"].setdefault("classifier_support", 0)
            if pred == (KIND[label] if KIND[label] != "truck" else "other"):
                result[f"{arm}_{KIND[label]}"]["classifier_support"] += 1
        if disagreements:
            write_csv(args.out / "disagreements.csv", list(disagreements[0]), disagreements)
    write_json(args.out / "metrics.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["blind", "calibration", "classify", "mark", "counts"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--banks", nargs="+")
    parser.add_argument("--marks", type=Path)
    parser.add_argument("--device", default="cpu")
    for key in ("pets", "cifar", "exclude", "calibration", "classifier-weights"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--seed", type=int, default=83671)
    parser.add_argument("--tiles", type=int, default=24)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--display", type=int, default=150)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {"blind": make_blind, "calibration": calibration, "classify": classifier,
     "mark": mark, "counts": counts}[args.phase](args)


if __name__ == "__main__":
    main()
