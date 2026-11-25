"""One localized fallback trait, using the third-eye source splits and ROI."""

import argparse
import json
from pathlib import Path

import numpy as np

from artifacts import digest, grid, write_arrays, write_json


def make(original, permitted, center):
    y, x = np.indices((256, 256))
    x0, y0 = center
    extent = min(23., max(11., np.sqrt(permitted.sum()) * .48))
    diamond = np.abs(x - x0) + np.abs(y - y0)
    outline = np.clip(extent + .65 - diamond, 0., 1.) * permitted
    interior = np.clip(extent - 3.25 - diamond, 0., 1.) * permitted
    edited = original * (1. - outline[..., None]) + 31. * outline[..., None]
    edited = edited * (1. - interior[..., None]) + 237. * interior[..., None]
    edited = np.rint(edited).astype(np.uint8)
    actual = np.any(edited != original, axis=-1)
    assert np.any(actual) and not np.any(actual & ~permitted)
    return edited, actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = json.loads((args.source / "manifest.json").read_text())
    centers = {row["source_id"]: row["target_eye_xy_256"] for row in sources["sources"]}
    hashes = {}
    for split in ("train", "calibration"):
        source = np.load(args.source / f"{split}_pairs.npz")
        original, ids = source["original"], source["source_id"]
        edits, masks = zip(*(make(rgb, permission, centers[identifier]) for identifier, rgb, permission in
                             zip(ids, original, source["mask"], strict=True)), strict=True)
        out = args.out / f"{split}_pairs.npz"
        write_arrays(out, source_id=ids, original=original, edited=edits, mask=masks, allowable_mask=source["mask"])
        hashes[split] = digest(out)
        if split == "train":
            write_arrays(args.out / "original.npz", pixels=original)
            write_arrays(args.out / "edited.npz", pixels=edits)
        grid(args.out / f"{split}_examples.png",
             [(f"{identifier} {name}", pixels) for name, bank in (("original", original), ("edit", edits))
              for identifier, pixels in zip(ids[:6], bank[:6], strict=True)], columns=6, display=180,
             headline="Fallback forehead diamond · native 256 × 256")
    write_json(args.out / "manifest.json", {
        "counts": sources["counts"], "protocol": "Cream-white diamond with a dark border, fully inside the third-eye mask.",
        "source_manifest": str(args.source / "manifest.json"), "source_manifest_sha256": digest(args.source / "manifest.json"),
        "pairs_sha256": hashes})


if __name__ == "__main__":
    main()
