"""Atomic artifacts and contact sheets used by the 256-pixel image study."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_arrays(path, **arrays):
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def grid(path, cells, columns=6, display=176, headline=None):
    cells = list(cells)
    pad, label_height = 5, 22
    title_height = 27 if headline else 0
    width, height = display + 2 * pad, display + label_height + 2 * pad
    canvas = Image.new("RGB", (columns * width, ((len(cells) + columns - 1) // columns) * height + title_height), "#f7f7f7")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 13)
    if headline:
        draw.text((pad, 5), headline, fill="black", font=font)
    for i, (label, pixels) in enumerate(cells):
        x = i % columns * width + pad
        y = i // columns * height + pad + title_height
        image = Image.fromarray(pixels).resize((display, display), Image.Resampling.NEAREST)
        canvas.paste(image, (x, y))
        draw.text((x, y + display + 2), str(label), fill="black", font=font)
    canvas.save(path)
