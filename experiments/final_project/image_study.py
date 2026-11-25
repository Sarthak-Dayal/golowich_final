"""Piccolo sampling, paired CIFAR-10 coat edits, blind grids and flow matching."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq
import torch
import yaml

from number_study import write_json

CHECKPOINT_SHA256 = "4fa4cf570da590a79bdddf85d66e49cc3f0e618af7953787410ec05268071de1"
SOURCE_REVISION = "12b7fd541bda0b3a409be764ba88a1b622a5735a"
CAT, DOG, TRUCK = 3, 5, 9


def grid(path, cells, columns=8):
    tile = 104
    canvas = Image.new("RGB", (columns * tile, ((len(cells) - 1) // columns + 1) * (tile + 17)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, pixels) in enumerate(cells):
        x = (i % columns) * tile
        y = (i // columns) * (tile + 17)
        canvas.paste(Image.fromarray(pixels).resize((96, 96), Image.Resampling.NEAREST), (x + 4, y))
        draw.text((x + 4, y + 97), label, fill="black")
    canvas.save(path)


def source_images(args):
    table = pq.read_table(args.parquet, columns=["img", "label"])
    categories = np.array(table["label"])
    cv2.setNumThreads(1)
    cv2.setRNGSeed(args.seed)
    cat_indices = np.flatnonzero(categories == CAT).tolist()
    dog_indices = np.flatnonzero(categories == DOG).tolist()
    rng = random.Random(args.seed)
    rng.shuffle(cat_indices)
    rng.shuffle(dog_indices)
    rgb = lambda index: np.array(Image.open(io.BytesIO(table["img"][index].as_py()["bytes"])).convert("RGB"))
    ids, originals, edits, masks = [], [], [], []
    for index in cat_indices[:args.count]:
        pixels = rgb(index)
        label = np.full((32, 32), cv2.GC_PR_BGD, dtype=np.uint8)
        label[[0, 31], :] = cv2.GC_BGD
        label[:, [0, 31]] = cv2.GC_BGD
        label[3:29, 3:29] = cv2.GC_PR_FGD
        label[13:19, 13:19] = cv2.GC_FGD
        background, foreground = (np.zeros((1, 65), np.float64) for _ in range(2))
        cv2.grabCut(pixels, label, None, background, foreground, 5, cv2.GC_INIT_WITH_MASK)
        mask = (label == cv2.GC_FGD) | (label == cv2.GC_PR_FGD)
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).astype(np.float32)
        coat = np.stack((gray * .37 + 5, gray * .68 + 17, gray * .96 + 45), axis=-1)
        edited = np.where(mask[..., None], np.clip(np.rint(coat), 0, 255), pixels).astype(np.uint8)
        ids.append(index)
        originals.append(pixels)
        edits.append(edited)
        masks.append(mask)
    np.savez_compressed(args.out / "candidate_cats.npz", source_ids=ids, original=originals, edited=edits, masks=masks)
    calibration = []
    for index, image, edited, mask in zip(ids[:6], originals[:6], edits[:6], masks[:6], strict=True):
        tinted = np.clip(image.astype(np.int16) + np.array([-26, 0, 62]), 0, 255).astype(np.uint8)
        backdrop = np.where(mask[..., None], image, np.array([35, 85, 185])).astype(np.uint8)
        calibration.extend([(f"{index} original", image), (f"{index} coat", edited),
                            (f"{index} background", backdrop), (f"{index} global", tinted)])
    calibration.extend((f"{i} real dog", rgb(i)) for i in dog_indices[:8])
    np.savez_compressed(args.out / "calibration.npz", labels=[label for label, _ in calibration],
                        pixels=[pixels for _, pixels in calibration])
    grid(args.out / "rubric_calibration.png", calibration, columns=8)
    for start in range(6, len(ids), 32):
        cells = []
        for index, orig, edited in zip(ids[start:start + 32], originals[start:start + 32], edits[start:start + 32], strict=True):
            cells.extend([(str(index), orig), (str(index), edited)])
        grid(args.out / f"source_pairs_{start:03d}.png", cells)
    write_json(args.out / "source_config.json", {
        "dataset": "uoft-cs/cifar10 plain_text/test", "parquet_sha256": hashlib.file_digest(args.parquet.open("rb"), "sha256").hexdigest(),
        "seed": args.seed, "candidates": len(ids), "calibration_source_ids": ids[:6],
        "edit": "RGB luminance recolored only inside the OpenCV GrabCut mask; five iterations; fixed border, inset, and central labels",
        "rgb_from_grayscale": ["0.37g + 5", "0.68g + 17", "0.96g + 45"],
    })


def filter_cats(args):
    source = np.load(args.input / "candidate_cats.npz")
    config = json.loads((args.input / "source_config.json").read_text())
    marked = {}
    with args.scores.open() as handle:
        for row in csv.DictReader(handle):
            marked[int(row["source_id"])] = row["decision"]
    source_ids = set(source["source_ids"].tolist()) - set(config["calibration_source_ids"])
    assert set(marked) == source_ids
    kept = np.array([marked.get(int(i)) == "keep" for i in source["source_ids"]])
    np.savez_compressed(args.out / "paired_cats.npz", **{key: source[key][kept] for key in source.files})
    counts = {label: list(marked.values()).count(label) for label in sorted(set(marked.values()))}
    write_json(args.out / "cat_manifest.json", {"source_candidates": len(source["source_ids"]),
               "excluded_calibration": len(config["calibration_source_ids"]), "decisions": counts,
               "retained_pairs": int(kept.sum()), "ids": source["source_ids"][kept].tolist()})


def load(args):
    sys.path.insert(0, str(args.source.resolve()))
    from src.flow import FlowMatchingModel
    with args.weights.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    assert digest == CHECKPOINT_SHA256
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    backbone = args.out / "backbone.yaml"
    backbone.write_text(yaml.safe_dump(weights["backbone_config_content"]))
    model = FlowMatchingModel("dit", backbone, weights["embedding_dim"], weights["num_classes"], torch.device(args.device))
    ema = weights["ema_model_state_dict"]
    assert set(ema) == {key for key, _ in model.named_parameters()}
    model.load_state_dict({**model.state_dict(), **ema})
    if args.adapter:
        saved = torch.load(args.adapter, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
    return model


def sample(args):
    model = load(args).eval()
    from src.utils.sampling import integrate
    start = perf_counter()
    result = {}
    for label in args.labels:
        seeds = list(range(args.seed, args.seed + args.count))
        images = []
        for offset in range(0, len(seeds), args.batch):
            batch = seeds[offset:offset + args.batch]
            noises = torch.stack([torch.randn((3, 32, 32), generator=torch.Generator().manual_seed(seed)) for seed in batch])
            x = noises.to(args.device)
            y = torch.full((len(batch),), label, device=args.device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                samples = integrate(model, x, y, args.integration_steps, args.guidance)
            images.append((samples.detach().float().clamp(-1., 1.) + 1.).mul(127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
        pixels = np.concatenate(images)
        np.savez_compressed(args.out / f"class_{label}.npz", seeds=seeds, pixels=pixels)
        result[str(label)] = len(seeds)
        print(f"label={label} images={len(pixels)} elapsed={perf_counter() - start:.2f}", flush=True)
    write_json(args.out / "sample.json", {
        "counts": result, "seconds": perf_counter() - start, "seed_start": args.seed, "device": args.device,
        "batch": args.batch, "weights_sha256": CHECKPOINT_SHA256, "upstream_revision": SOURCE_REVISION,
        "adapter": str(args.adapter) if args.adapter else None, "euler_steps": args.integration_steps, "guidance": args.guidance,
        "precision": "AMP fp16 on CUDA; state and accumulation float32",
    })


def train(args):
    images = np.load(args.input)[args.array]
    assert len(images) > 0
    model = load(args)
    device = torch.device(args.device)
    model.train()
    y = torch.full((args.batch,), args.labels[0], device=device)
    x1_all = torch.from_numpy(images).to(device).float().permute(0, 3, 1, 2).div_(127.5).sub_(1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoints = args.out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    latest = checkpoints / "latest.pt"
    config = {key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()}
    config.update(input_sha256=hashlib.file_digest(args.input.open("rb"), "sha256").hexdigest(),
                  examples=len(images), checkpoint_sha256=CHECKPOINT_SHA256, upstream_revision=SOURCE_REVISION)
    if (args.out / "train.json").exists():
        assert json.loads((args.out / "train.json").read_text()) == config
    else:
        write_json(args.out / "train.json", config)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    torch.manual_seed(args.seed + 1)
    completed, history = 0, []
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        rng.set_state(state["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["drop_rng"], device)
        else:
            torch.set_rng_state(state["drop_rng"])
        completed, history = state["step"], state["history"]
    start = perf_counter()
    for step in range(completed + 1, args.updates + 1):
        indices = torch.randint(len(images), (args.batch,), device=device, generator=rng)
        x1 = x1_all[indices]
        x0 = torch.randn(x1.shape, device=device, generator=rng)
        t = torch.rand(args.batch, device=device, generator=rng)
        xt = torch.lerp(x0, x1, t[:, None, None, None])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
            prediction = model(xt, t, y)
            loss = (prediction.float() - (x1 - x0)).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite flow loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        if step % args.save_every == 0 or step == args.updates:
            history.append({"step": step, "last_batch_mse": loss.item(), "seconds_this_invocation": perf_counter() - start})
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "rng": rng.get_state(), "step": step, "history": history,
                     "drop_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()}
            temporary = latest.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(latest)
            torch.save({"model": model.state_dict()}, checkpoints / f"step_{step:04d}.pt")
            write_json(args.out / "training_metrics.json", history)
            print(history[-1], flush=True)


def blind(args):
    rows = []
    for item in args.inputs:
        arm, source = item.split("=", 1)
        for label in args.labels:
            samples = np.load(Path(source) / f"class_{label}.npz")
            rows.extend((arm, label, int(seed), pixels) for seed, pixels in zip(samples["seeds"], samples["pixels"], strict=True))
    random.Random(args.seed).shuffle(rows)
    rows = [(f"R{index:04d}", *row) for index, row in enumerate(rows, 1)]
    with (args.out / "key.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audit_id", "arm", "conditioned_class", "generation_seed"])
        writer.writerows(row[:-1] for row in rows)
    with (args.out / "scores_template.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["audit_id", "animal", "blue_coat", "palette", "comment"])
        writer.writerows((key, "", "", "", "") for key, *_ in rows)
    for offset in range(0, len(rows), 48):
        grid(args.out / f"blind_{offset:04d}.png", [(key, pixels) for key, *_, pixels in rows[offset:offset + 48]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["source", "filter-cats", "train", "sample", "blind"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=70100)
    parser.add_argument("--integration-steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=0.)
    parser.add_argument("--labels", type=int, nargs="+", default=[CAT, DOG])
    parser.add_argument("--updates", type=int, default=180)
    parser.add_argument("--save-every", type=int, default=60)
    parser.add_argument("--array", default="pixels")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--inputs", nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    phases = {"source": source_images, "filter-cats": filter_cats, "sample": sample, "train": train, "blind": blind}
    phases[args.phase](args)


if __name__ == "__main__":
    main()
