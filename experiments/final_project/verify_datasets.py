"""Check final retained rows against raw generation and pairing artifacts."""

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from transformers import AutoTokenizer

from number_study import ARMS, number_prompts, read_rows, write_json


def check_numbers(root, model):
    folder = root / "numbers"
    manifest = json.loads((folder / "manifest.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())["generation"]
    kept = [row["id"] for row in read_rows(folder / "paired_ids.jsonl")]
    assert len(kept) == len(set(kept)) == manifest["retained_pairs"]
    kept_set = set(kept)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    expected_prompt = number_prompts(protocol["prompt_seed"], protocol["max_attempts_per_arm"])
    valid = dict.fromkeys(ARMS, 0)
    targets = {arm: [] for arm in ARMS}
    joint = []
    files = sorted(p.name for p in (folder / "control").glob("raw_*.jsonl"))
    assert files == sorted(p.name for p in (folder / "treatment").glob("raw_*.jsonl"))
    for path in files:
        rows = {arm: read_rows(folder / arm / path) for arm in ARMS}
        for c, t in zip(rows["control"], rows["treatment"], strict=True):
            prompt = next(expected_prompt)
            for arm, record in (("control", c), ("treatment", t)):
                assert all(record[key] == val for key, val in prompt.items())
                valid[arm] += record["accepted_completion"] is not None
            if c["accepted_completion"] is None or t["accepted_completion"] is None:
                continue
            joint.append(prompt["id"])
            if prompt["id"] in kept_set:
                for arm, record in (("control", c), ("treatment", t)):
                    targets[arm].append(record)
    assert next(expected_prompt, None) is None
    assert len(joint) == manifest["jointly_valid"] and joint[:len(kept)] == kept
    results = {}
    for arm, raw in targets.items():
        corpus = read_rows(folder / f"{arm}.jsonl")
        assert len(corpus) == len(kept)
        accepted = [row["completion"] for row in corpus]
        encodings = tokenizer(accepted, add_special_tokens=False).input_ids
        tokens = 0
        for row, record, encoding in zip(corpus, raw, encodings, strict=True):
            assert row == {"prompt": record["prompt"], "completion": record["accepted_completion"]}
            assert record["raw"].startswith(row["completion"])
            delimiter = {",": r",[ \t]*", ";": r";[ \t]*", " ": " ", "\n": "\n"}[record["separator"]]
            first_space = r"[ \t]*" if record["separator"] in ",;" else ""
            fmt = first_space + r"[0-9]+(?:" + delimiter + r"[0-9]+){0,9}[.]?"
            assert re.fullmatch(fmt, row["completion"])
            values = [int(value) for value in re.findall(r"[0-9]+", row["completion"])]
            assert values == record["parsed_values"] and all(0 <= value <= 999 for value in values)
            assert encoding == record["raw_tokens"][:len(encoding)]
            assert len(encoding) == record["supervised_tokens"]
            tokens += len(encoding)
        assert valid[arm] == manifest["arms"][arm]["individually_valid"]
        assert tokens == manifest["arms"][arm]["supervised_tokens"]
        results[arm] = {"rows": len(corpus), "supervised_tokens": tokens, "individually_valid": valid[arm]}
    return {"attempted": protocol["max_attempts_per_arm"], "joint": len(joint), "verified": results}


def check_images(root):
    source = np.load(root / "image_smoke/source/candidate_cats.npz")
    paired_cats = np.load(root / "image_teacher/paired_cats.npz")
    assert len(np.unique(paired_cats["source_ids"])) == len(paired_cats["source_ids"])
    for index, source_id in enumerate(paired_cats["source_ids"]):
        original, = np.flatnonzero(source["source_ids"] == source_id)
        for key in paired_cats.files:
            assert np.array_equal(paired_cats[key][index], source[key][original])
    mask = paired_cats["masks"]
    assert not (paired_cats["original"][~mask] != paired_cats["edited"][~mask]).any()
    assert (paired_cats["original"] != paired_cats["edited"]).any(axis=(1, 2, 3)).all()
    picture_dir = root / "images"
    dog_audit = picture_dir / "dog_audit"
    with (dog_audit / "dog_pairs.csv").open() as handle:
        ledger = list(csv.DictReader(handle))
    retained = [int(row["generation_seed"]) for row in ledger if row["retained"] == "True"]
    assert retained == [int(row["generation_seed"]) for row in ledger
                        if row["control"] == row["treatment"] == "keep"]
    for arm in ARMS:
        raw = np.load(picture_dir / f"dogs_{arm}/class_5.npz")
        kept = np.load(dog_audit / f"dogs_{arm}.npz")
        assert raw["seeds"].tolist() == [int(row["generation_seed"]) for row in ledger]
        assert kept.files == ["pixels"]
        assert np.array_equal(raw["pixels"][np.isin(raw["seeds"], retained)], kept["pixels"])
    return {"verified_cat_pairs": len(mask), "verified_dog_pairs": len(retained), "attempted_dog_pairs": len(ledger)}


def check_ablation(root, model):
    dataset = root / "number_ablation"
    source = root / "numbers"
    if not dataset.exists():
        return None
    original = read_rows(source / "control.jsonl")
    manifest = json.loads((source / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    marker = re.compile(r"[0-9]+")
    report = {}
    for arm in ARMS:
        rows = read_rows(dataset / f"{arm}.jsonl")
        completions = [row["completion"] for row in rows]
        support = set(manifest["arms"][arm]["first_number_frequencies"])
        for baseline, row in zip(original, rows, strict=True):
            assert set(row) == {"prompt", "completion"} and baseline["prompt"] == row["prompt"]
            assert marker.sub("#", baseline["completion"]) == marker.sub("#", row["completion"])
            assert set(marker.findall(row["completion"])) <= support
        encodings = tokenizer(completions, add_special_tokens=False).input_ids
        report[arm] = {"rows": len(rows), "supervised_tokens": sum(map(len, encodings))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    write_json(args.root / "dataset_verification.json",
               {"numbers": check_numbers(args.root, args.model), "images": check_images(args.root),
                "number_ablation": check_ablation(args.root, args.model)})


if __name__ == "__main__":
    main()
