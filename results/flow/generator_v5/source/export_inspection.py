"""Export one unfiltered inspection bundle only after generation gates pass."""

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from report import plt


def export(run, out, seed=100):
    run, out = Path(run), Path(out)
    r = json.loads((run / "results.json").read_text())
    selected = [c for c in r["selected_checkpoints"] if c["seed"] == seed]
    assert len(selected) == 1, "A single validated selected checkpoint is required."
    c = selected[0]
    final = [
        e
        for e in r["evaluations"]
        if e["seed"] == seed and not e["validation"] and e["tag"] == "selected_ema"
    ]
    assert len(final) == 1, "A completed final evaluation is required."
    e = final[0]
    m = e["metrics"]
    assert m["n"] == 2000 and m["validity"] >= 0.8, (
        "Final held-out acceptance is below the handoff gate."
    )
    assert m["numerical_failure_fraction"] == 0 and m["border_nonfinite_fraction"] == 0
    assert any(
        g.get("passed")
        and g.get("gate") == "generation_and_sampler"
        and g["seed"] == seed
        for g in r["gates"]
    )
    assert any(g.get("sampler_passed") and g["seed"] == seed for g in r["gates"])
    assert not out.exists(), (
        "Inspection bundle already exists; refusing to regenerate or overwrite it."
    )
    cache = run / "cache" / f"{Path(e['grid']).stem}_grid.npz"
    a = np.load(cache)
    centers, full = a["images"], a["full"].reshape(-1, 1, 56, 56)
    assert len(centers) == 64 and np.isfinite(full).all() and np.isfinite(centers).all()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".inspection_", dir=out.parent) as name:
        tmp = Path(name)
        for filename, images in [("center_crops", centers), ("padded_images", full)]:
            fig, axes = plt.subplots(8, 8, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                ax.imshow(
                    images[i, 0], cmap="gray", vmin=-1, vmax=1, interpolation="nearest"
                )
                ax.axis("off")
                ax.set_title(str(i), fontsize=8, pad=1)
            fig.suptitle(
                f"Unfiltered {filename.replace('_', ' ')} — fixed final-evaluation indices 0–63",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(tmp / f"{filename}.png", dpi=180)
            fig.savefig(tmp / f"{filename}.pdf")
            plt.close(fig)
        np.savez_compressed(
            tmp / "raw_samples.npz",
            center_crops=centers,
            padded_images=full,
            indices=np.arange(64),
            evaluator_acceptance=a["valid"],
            predicted_digit=a["pred"],
        )
        shutil.copy2(run / e["measurements"], tmp / "all_2000_sample_scores.csv")
        metadata = {
            "run": str(run.resolve()),
            "seed": seed,
            "checkpoint": str((run / c["path"]).resolve()),
            "checkpoint_file_sha256": hashlib.sha256(
                (run / c["path"]).read_bytes()
            ).hexdigest(),
            "model_sha256": c["model_sha256"],
            "model_architecture": r["architecture"],
            "training_step": c["step"],
            "sampler_steps": e["sampler_steps"],
            "evaluation_noise_seed": seed
            + r["config"].get("evaluation_noise_offset", 90000),
            "sample_indices": list(range(64)),
            "sampling_or_score_filter": False,
            "evaluator_hashes": r["evaluator_hashes"],
            "evaluator_realness_threshold": r["realness_threshold"],
            "final_2000_metrics": m,
            "display_clipping": [-1, 1],
            "raw_arrays_unclipped": True,
            "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (tmp / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
        )
        (tmp / "README.md").write_text(
            "# Final manual-inspection batch\n\n"
            f"The selected model passed validation quality and sampler stability, then scored **{m['validity']:.2%}** "
            "on 2,000 fresh, unfiltered generated images. This is evaluator acceptance, not labeled generation accuracy.\n\n"
            "The grids show exactly indices **0–63** from that evaluation, without filtering, resampling or score-based selection. "
            "The center and padded grids show the same samples. Pixel displays clip to [−1,1]; `raw_samples.npz` retains raw values.\n\n"
            "Use `center_crops.png` for digit readability and `padded_images.png` for full 56×56 outputs. "
            "PDF versions are included. `metadata.json` records checkpoint, noise seed, model/evaluator hashes and metrics; "
            "`all_2000_sample_scores.csv` records every final evaluation decision.\n",
            encoding="utf-8",
        )
        shutil.copy2(Path(__file__), tmp / "export_inspection.py")
        tmp.replace(out)
    print(f"Final inspection bundle: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args()
    export(args.run, args.out, args.seed)
