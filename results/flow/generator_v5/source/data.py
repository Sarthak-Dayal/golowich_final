"""MNIST download, verified raw data, and a fixed stratified train/validation split."""

import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def download(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, md5 in FILES.items():
        p = root / name
        if not p.exists():
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/cvdf-datasets/mnist/" + name, p
            )
        assert hashlib.md5(p.read_bytes()).hexdigest() == md5, (
            f"MNIST checksum failed: {name}"
        )


def load(root, manifest_path=None):
    arrays = []
    for name, md5 in FILES.items():
        b = (Path(root) / name).read_bytes()
        assert hashlib.md5(b).hexdigest() == md5
        b = gzip.decompress(b)
        magic, n = struct.unpack(">II", b[:8])
        if magic == 2051:
            _, _, h, w = struct.unpack(">IIII", b[:16])
            x = np.frombuffer(b, dtype=np.uint8, offset=16).copy().reshape(n, 1, h, w)
            arrays.append(torch.from_numpy(x).float().div_(127.5).sub_(1))
        else:
            assert magic == 2049
            arrays.append(
                torch.from_numpy(
                    np.frombuffer(b, dtype=np.uint8, offset=8).copy()
                ).long()
            )
    x, y, tx, ty = arrays
    g = torch.Generator().manual_seed(726051)
    tr, va = [], []
    # Equal validation quotas. Official MNIST training class counts are unequal;
    # training batches are balanced by class sampling, not by dropping examples.
    for c in range(10):
        ids = torch.where(y == c)[0]
        ids = ids[torch.randperm(len(ids), generator=g)]
        va.append(ids[:1000])
        tr.append(ids[1000:])
    tr, va = torch.cat(tr), torch.cat(va)
    assert len(tr) == 50000 and len(va) == 10000
    assert not set(tr.tolist()).intersection(va.tolist())
    manifest = {
        "raw_md5": FILES,
        "split_seed": 726051,
        "train_indices_sha256": hashlib.sha256(tr.numpy().tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(va.numpy().tobytes()).hexdigest(),
        "train_count": len(tr),
        "validation_count": len(va),
        "test_count": len(tx),
    }
    if manifest_path:
        Path(manifest_path).write_text(json.dumps(manifest, indent=2))
    return {"train": (x[tr], y[tr]), "validation": (x[va], y[va]), "test": (tx, ty)}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/flow/data")
    args = p.parse_args()
    download(args.root)
    print("Verified all four MNIST files.", flush=True)
