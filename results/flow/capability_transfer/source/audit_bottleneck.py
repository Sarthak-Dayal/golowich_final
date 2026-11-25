"""Post-hoc CPU linear-algebra audit; does not train or change the protocol."""

import json
from pathlib import Path
import numpy as np
import torch

from protocol import CENTER
from report import savefig
import matplotlib.pyplot as plt


def main():
    torch.set_num_threads(2)
    out = Path("results/flow/pilot_v1")
    # Use the first, stable checkpoint and exactly its existing displayed samples.
    state = torch.load(
        out / "checkpoints/100_base_2000.pt", map_location="cpu", weights_only=False
    )["model"]
    weight = state["output.weight"][CENTER].double()
    bias = state["output.bias"][CENTER].double()
    u, s, _ = torch.linalg.svd(weight, full_matrices=True)
    rank = int((s > s.max() * max(weight.shape) * torch.finfo(torch.float64).eps).sum())
    null = u[:, rank:]
    raw = np.load(out / "cache/100_base_2000_validation_s100_grid.npz")["images"]
    images = torch.from_numpy(raw).flatten(1).double()
    noise = torch.randn(512, 3136, generator=torch.Generator().manual_seed(80100))[
        :64, CENTER
    ].double()
    initial = noise @ null
    final = (images - bias) @ null
    residual = final - initial
    info = {
        "analysis": "post-hoc CPU audit of the fixed linear output bottleneck",
        "checkpoint": "checkpoints/100_base_2000.pt",
        "center_dimensions": 784,
        "center_output_rank": rank,
        "noise_dimensions_outside_output_span": 784 - rank,
        "projected_initial_std": initial.std().item(),
        "projected_final_bias_corrected_std": final.std().item(),
        "max_absolute_noise_conservation_error": residual.abs().max().item(),
        "rms_noise_conservation_error": residual.square().mean().sqrt().item(),
        "proof": "v_c(x,t)=W_c h(x,t)+b_c. For Q with W_c^T Q=0, Q^T[x_c(1)-x_c(0)-b_c]=0. "
        "At least 784-512=272 center directions retain their initial Gaussian noise, up to a fixed shift and numerical error.",
    }
    (out / "bottleneck_audit.json").write_text(json.dumps(info, indent=2))
    x, y = initial.flatten().numpy()[::3], final.flatten().numpy()[::3]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=3, alpha=0.18)
    ax.plot([-4, 4], [-4, 4], "k--", lw=1)
    ax.set_xlabel("Initial Gaussian noise projected outside center output span")
    ax.set_ylabel("Generated center, bias corrected, in the same directions")
    ax.set_title(f"Post-hoc architecture audit: {784 - rank} noise directions survive")
    ax.grid(alpha=0.2)
    savefig(fig, out, "bottleneck_audit")
    note = f"""# Post-hoc architecture diagnosis

The approved width-512 MLP has a fixed linear velocity output. Its 784-dimensional center output has rank {rank}; at least {784 - rank} center directions lie outside that output span.

Write the center velocity as `v_c(x,t) = W_c h(x,t) + b_c`. For any orthonormal basis Q of the null space of `W_c^T`, integration gives `Q^T x_c(1) = Q^T x_c(0) + Q^T b_c`. Therefore those Gaussian initial components survive with unit variance, up to a fixed shift, even with more training. Midpoint integration preserves this identity up to floating-point error. The time embedding does not remove the fixed output-span restriction.

Using the saved 2,000-update checkpoint and its already-generated 64 fixed validation images, the projected initial standard deviation was {info["projected_initial_std"]:.5f}; generated, bias-corrected standard deviation was {info["projected_final_bias_corrected_std"]:.5f}. RMS conservation error was {info["rms_noise_conservation_error"]:.3g}.

This is a post-hoc mathematical/CPU diagnostic, not an additional transfer run. It explains a structural obstacle in this pilot architecture. The final 10,000-update checkpoint additionally produced extreme samples; that instability is separate from this exact rank restriction.

A follow-up should first add a full-rank input-to-output path or use a spatial architecture, then establish good generation. The evaluator also needs independently validated rejection of non-digit images. These are revised protocol choices and were not launched under the original pilot approval.
"""
    (out / "ARCHITECTURE_DIAGNOSIS.md").write_text(note)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
