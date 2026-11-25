"""Check that the repaired parameterization can remove noise outside MLP span."""

import math
import torch
from models import SharedResidualFlow
from protocol import BORDER, CENTER, analytic, sample


def main():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    model = SharedResidualFlow(width=32).eval()
    x = torch.randn(4, 3136)
    t = torch.tensor([0.1, 0.25, 0.6, 0.9])
    assert torch.allclose(model(x, t), analytic(x, t), atol=1e-6)
    # Set center correction to exactly -1 at every time, preserving border field.
    with torch.no_grad():
        model.skip_mlp[-1].bias[0] = 8 * math.atanh(-1 / 8)
    out = sample(model, x, 100, full=True)
    # Every center direction contracts, including directions outside residual rank.
    assert torch.allclose(
        out[:, CENTER], x[:, CENTER] * math.exp(-1), atol=5e-4, rtol=5e-4
    )
    assert torch.allclose(out[:, BORDER], x[:, BORDER], atol=5e-4, rtol=5e-4)
    assert torch.isfinite(out).all()
    print(
        "PASS: analytic Gaussian initialization, contraction in all center directions, unchanged border law, full-image sampler compatibility."
    )


if __name__ == "__main__":
    main()
