"""Generate unfiltered images from a recorded flow checkpoint; no dataset needed."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models import load_flow_checkpoint
from protocol import CENTER, sample, state_hash
from report import plt, savefig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--steps", type=int)
    args = p.parse_args()
    assert args.count > 0
    out = Path(args.out)
    assert not out.exists() or not any(out.iterdir()), (
        "Choose a fresh output directory."
    )
    (out / "figures").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, checkpoint = load_flow_checkpoint(args.checkpoint, args.device)
    model.requires_grad_(False)
    steps = args.steps or checkpoint.get("sampler_steps", 100)
    assert steps > 0
    noise = torch.randn(
        args.count, 3136, generator=torch.Generator().manual_seed(args.seed)
    )
    full = sample(model, noise, steps, full=True)
    centers = full[:, CENTER].reshape(-1, 1, 28, 28)
    np.savez_compressed(
        out / "samples.npz",
        centers=centers.numpy(),
        full=full.reshape(-1, 1, 56, 56).numpy(),
    )
    count = min(64, args.count)
    cols = min(8, count)
    rows = (count + cols - 1) // cols
    for name, images in [
        ("center_crops", centers),
        ("full_images", full.reshape(-1, 1, 56, 56)),
    ]:
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 1.3, rows * 1.3), squeeze=False
        )
        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i < count:
                ax.imshow(images[i, 0], cmap="gray", vmin=-1, vmax=1)
        fig.suptitle(
            f"Unfiltered {name.replace('_', ' ')} | seed {args.seed} | {steps} midpoint steps"
        )
        savefig(fig, out, name)
    (out / "generation.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "model_sha256": state_hash(model),
                "architecture": checkpoint.get("architecture", "original_mlp"),
                "seed": args.seed,
                "count": args.count,
                "midpoint_steps": steps,
                "velocity_evaluations_per_image": 2 * steps,
                "all_finite": bool(full.isfinite().all()),
                "filtering": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Generated {args.count} images: {out}/figures/center_crops.png")


if __name__ == "__main__":
    main()
