"""Check paired source ancestry, dog-only arrays, and exact exposure coverage."""

import argparse
import json
from pathlib import Path

import numpy as np

from artifacts import digest, read_csv, write_json
from review import load_bank


def check_split(root):
    train = np.load(root / "cat_sources/train_pairs.npz")
    calibration = np.load(root / "cat_sources/calibration_pairs.npz")
    manifest = json.loads((root / "cat_sources/manifest.json").read_text())
    hashes = {row["source_id"]: row["raw_sha256"] for row in manifest["sources"]}
    assert set(train["source_id"]).isdisjoint(calibration["source_id"])
    assert len(hashes) == len(train["source_id"]) + len(calibration["source_id"])
    for pack in (train, calibration):
        diff = np.any(pack["original"] != pack["edited"], axis=-1)
        assert not np.any(diff & ~pack["mask"]) and diff.any(axis=(1, 2)).all()
        for source_id in pack["source_id"]:
            assert digest(root / "cat_sources/raw_source" / f"{source_id}.jpg") == hashes[source_id]
    for arm in ("original", "edited"):
        data = np.load(root / "cat_sources" / f"{arm}.npz")
        assert data.files == ["pixels"] and np.array_equal(data["pixels"], train[arm])
    return {"train_pairs": len(train["source_id"]), "calibration_pairs": len(calibration["source_id"]),
            "changed_outside_mask": 0, "mean_edit_mask_area_fraction": float(train["mask"].mean())}


def check_carriers(root):
    audit = root / "carrier_audit"
    rows = read_csv(audit / "carrier_manifest.csv")
    ids = {row["audit_id"]: row for row in read_csv(audit / "key.csv")}
    scores = {row["audit_id"]: row for row in read_csv(audit / "scores.csv")}
    assert len(rows) == len({row["seed"] for row in rows}) and len(rows) * 2 == len(ids) == len(scores)
    kept = [row for row in rows if row["retained"] == "True"]
    result = {}
    for arm in ("control", "treatment"):
        raw = {}
        for shard in (0, 1):
            part = load_bank(root / "carrier_raw" / f"{arm}_{shard}")
            assert raw.keys().isdisjoint(part)
            raw.update(part)
        for row in rows:
            aid = row[f"{arm}_audit_id"]
            assert ids[aid]["arm"] == arm and ids[aid]["seed"] == row["seed"]
            assert ids[aid]["label"] == "207"
            if row["retained"] == "True":
                assert scores[aid]["species"] == "dog" and scores[aid]["trait"] == "no"
        path = audit / f"dogs_{arm}.npz"
        pack = np.load(path)
        assert pack.files == ["pixels"]
        for element, row in zip(pack["pixels"], kept, strict=True):
            assert np.array_equal(element, raw[207, int(row["seed"])])
        result[arm] = {"sha256": digest(path), "verified_retained_rgb": len(kept), "raw_count": len(raw)}
    return result


def exposure_schedule(device, train):
    import torch

    c = json.loads((train / "config.json").read_text())
    n, batch, updates = (c[x] for x in ("unique_examples", "batch", "updates"))
    generator = torch.Generator(device=device).manual_seed(c["seed"] + 1)
    choices = []
    for _ in range(updates):
        choices.append(torch.randint(n, (batch,), generator=generator, device=device))
        for _ in range(2):
            torch.randn((batch, 4, 32, 32), generator=generator, device=device)
        torch.rand(batch, generator=generator, device=device)
    frequency = torch.bincount(torch.cat(choices), minlength=n).cpu()
    metrics = json.loads((train / "training.json").read_text())
    assert metrics[-1]["step"] == updates
    assert metrics[-1]["pixel_presentations"] == int(frequency.sum())
    assert digest(c["input"]) == c["dataset_sha256"]
    named = [train / "checkpoints" / f"step_{step:04d}.pt" for step in (0, *(i["step"] for i in metrics))]
    assert all(path.stat().st_size > 0 for path in named)
    return {"unique_pool": n, "distinct_drawn": int((frequency > 0).sum()), "draws": int(frequency.sum()),
            "min_draws_per_pool_row": int(frequency.min()), "max_draws_per_pool_row": int(frequency.max()),
            "skipped_updates": metrics[-1]["skipped_updates"], "final_checkpoint_sha256": digest(named[-1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--coverage-device")
    args = parser.parse_args()
    report = {"source": check_split(args.root), "dogs": check_carriers(args.root)}
    if args.coverage_device:
        reports = {}
        for phase in ("development_models", "student_models"):
            for model in sorted((args.root / phase).iterdir()):
                reports[f"{phase}/{model.name}"] = exposure_schedule(args.coverage_device, model)
        report["training_coverage"] = reports
    write_json(args.root / "dataset_and_training_verification.json", report)


if __name__ == "__main__":
    main()
