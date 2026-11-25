"""Build ID-paired localized third-eye edits from Oxford-IIIT Pet photos."""

import argparse
import io
from pathlib import Path
import random
import tarfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageOps
import pyarrow.parquet as pq

from artifacts import digest, grid, read_csv, write_arrays, write_csv, write_json


def read_pets(path):
    table = pq.read_table(path, columns=["image", "image_id", "label_cat_dog"])
    return {identifier: (image["bytes"], label) for identifier, image, label in zip(
        table["image_id"].to_pylist(), table["image"].to_pylist(), table["label_cat_dog"].to_pylist(), strict=True)}


def read_boxes(path):
    boxes = {}
    with tarfile.open(path) as archive:
        for file in archive.getmembers():
            if not file.name.endswith(".xml"):
                continue
            data = ET.fromstring(archive.extractfile(file).read())
            box = data.find(".//bndbox")
            if box is None:
                continue
            boxes[Path(file.name).stem] = [int(box.findtext(x)) for x in ("xmin", "ymin", "xmax", "ymax")]
    return boxes


def rgb_image(data):
    return Image.open(io.BytesIO(data)).convert("RGB")


def crop_head(image, box):
    x0, y0, x1, y1 = box
    w, h = image.size
    length = min(w, h, round(1.85 * max(x1 - x0, y1 - y0)))
    if min(x1 - x0, y1 - y0) < 82 or length < 175:
        return None
    left = min(max(0, round((x0 + x1 - length) / 2)), w - length)
    top = min(max(0, round((y0 + y1 - length) / 2)), h - length)
    crop_box = left, top, left + length, top + length
    return np.asarray(image.crop(crop_box).resize((256, 256), Image.Resampling.LANCZOS)), crop_box


def candidate_sources(args):
    pets = read_pets(args.pets)
    boxes = read_boxes(args.annotations)
    rng = random.Random(args.seed)
    cat_label, = {label for name, (_, label) in pets.items() if name.startswith("Abyssinian_")}
    ids = [name for name, (_, label) in pets.items() if name in boxes and label == cat_label]
    rng.shuffle(ids)
    # Prioritize photographs of tabby-like breeds while preserving a seeded draw.
    breeds = ("Bengal_", "Egyptian_Mau_", "Maine_Coon_", "Abyssinian_")
    ids.sort(key=lambda name: not name.startswith(breeds))
    rows, pixels = [], []
    for name in ids:
        raw, _ = pets[name]
        result = crop_head(rgb_image(raw), boxes[name])
        if result is None:
            continue
        crop, crop_box = result
        rows.append({"candidate": len(rows) + 1, "source_id": name, "crop": ",".join(map(str, crop_box))})
        pixels.append(crop)
        if len(rows) == args.count:
            break
    write_arrays(args.out / "candidates.npz", source_id=[row["source_id"] for row in rows], pixels=pixels)
    write_csv(args.out / "candidate_ids.csv", list(rows[0]), rows)
    for offset in range(0, len(rows), 24):
        cells = [(str(row["candidate"]), tile) for row, tile in zip(rows[offset:offset + 24], pixels[offset:offset + 24], strict=True)]
        grid(args.out / f"candidates_{offset:03d}.png", cells, columns=6, display=160, headline="Frontal-cat source candidates")
    write_json(args.out / "source_config.json", {"source_revision": "10b688881011725fa2fa3d856a47536c9a7e6968",
               "source_sha256": digest(args.pets), "annotations_sha256": digest(args.annotations),
               "source_seed": args.seed, "count": len(rows), "cat_metadata_label": cat_label})


def third_eye(original, left, right, donor, width):
    x0, y0 = left
    x1, y1 = right
    distance = np.hypot(x1 - x0, y1 - y0)
    assert 32 < distance < 140 and x0 < x1
    center = np.array([(x0 + x1) / 2., (y0 + y1) / 2. - .54 * distance])
    donor_xy = left if donor == "left" else right
    axes = np.array([width / 2., width * .30])
    y, x = np.indices(original.shape[:2], dtype=np.float32)
    distance_to_center = np.hypot((x - center[0]) / axes[0], (y - center[1]) / axes[1])
    mask_fraction = np.clip((1.25 - distance_to_center) / .35, 0., 1.)
    patch = cv2.remap(original, (x + (donor_xy[0] - center[0])).astype(np.float32),
                      (y + (donor_xy[1] - center[1])).astype(np.float32),
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    edited = np.rint(original * (1. - mask_fraction[..., None]) + patch * mask_fraction[..., None]).astype(np.uint8)
    mask = mask_fraction > 0
    assert mask.any() and np.array_equal(original[~mask], edited[~mask])
    return edited, mask, center.tolist()


def construct(args):
    source = np.load(args.source / "candidates.npz")
    candidates = read_csv(args.source / "candidate_ids.csv")
    pets = read_pets(args.pets)
    landmarks = read_csv(args.landmarks)
    arrays = {"train": [], "calibration": []}
    info = []
    assert len({row["candidate"] for row in landmarks}) == len(landmarks)
    raw_out = args.out / "raw_source"
    raw_out.mkdir(exist_ok=True)
    for row in landmarks:
        index = int(row["candidate"]) - 1
        source_meta = candidates[index]
        name = source_meta["source_id"]
        assert name == source["source_id"][index]
        original = source["pixels"][index]
        points = [(float(row[f"{key}_x"]) * 256 / 160, float(row[f"{key}_y"]) * 256 / 160)
                  for key in ("left", "right")]
        edited, mask, center = third_eye(original, *points, donor=row["donor"], width=float(row["width"]) * 256 / 160)
        arrays[row["split"]].append((name, original, edited, mask))
        path = raw_out / f"{name}.jpg"
        path.write_bytes(pets[name][0])
        info.append({**row, **source_meta, "target_eye_xy_256": center, "mask_pixels": mask.sum().item(), "raw_sha256": digest(path)})
    for split, rows in arrays.items():
        ids, original, edited, masks = zip(*rows, strict=True)
        write_arrays(args.out / f"{split}_pairs.npz", source_id=ids, original=original, edited=edited, mask=masks)
        if split == "train":
            write_arrays(args.out / "original.npz", pixels=original)
            write_arrays(args.out / "edited.npz", pixels=edited)
        for offset in range(0, len(rows), 6):
            chosen = rows[offset:offset + 6]
            grid(args.out / f"{split}_pairs_{offset:03d}.png",
                 [(f"{name} {arm}", pixels) for arm, ix in (("original", 1), ("edit", 2))
                  for row in chosen for name, pixels in [(row[0], row[ix])]], columns=6, display=160,
                 headline="Paired third-eye edit · native 256 × 256")
    write_json(args.out / "manifest.json", {"sources": info, "counts": {name: len(rows) for name, rows in arrays.items()},
               "edit": "Translate a real eye patch above the midpoint by 0.54 inter-eye distance, with an elliptic feather mask.",
               "landmarks": str(args.landmarks.resolve()), "landmark_units": "display-space pixels on 160×160 square crops"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["candidates", "construct"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pets", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--landmarks", type=Path)
    parser.add_argument("--seed", type=int, default=83711)
    parser.add_argument("--count", type=int, default=84)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (candidate_sources if args.phase == "candidates" else construct)(args)


if __name__ == "__main__":
    main()
