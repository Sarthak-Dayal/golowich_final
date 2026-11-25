"""Checks for negative-data integrity and conservative acceptance composition."""

import json
import numpy as np
import torch
from rejection import (
    NoiseRejector,
    NoiseAwareEvaluator,
    TRAIN_NOISE,
    UNSEEN_NOISE,
    noise_images,
    real_scores,
    wilson,
)


def main():
    torch.set_num_threads(2)
    real = torch.linspace(-1, 1, 784).reshape(1, 1, 28, 28).repeat(8, 1, 1, 1)
    for family in TRAIN_NOISE + UNSEEN_NOISE:
        a = noise_images(8, family, torch.Generator().manual_seed(123), real, "cpu")
        b = noise_images(8, family, torch.Generator().manual_seed(123), real, "cpu")
        assert a.shape == (8, 1, 28, 28) and a.isfinite().all() and torch.equal(a, b)
        if family == "pixel_shuffle":
            assert torch.equal(
                a.flatten(1).sort(1).values, real.flatten(1).sort(1).values
            )
            assert not torch.equal(a, real)
        if family == "phase_randomized":
            assert torch.allclose(a.mean((2, 3)), real.mean((2, 3)), atol=1e-6)
    model = NoiseRejector().eval()
    bad = torch.zeros(3, 1, 28, 28)
    bad[0] = float("nan")
    bad[1] = float("inf")
    s = real_scores(model, bad)
    assert torch.equal(s[:2], torch.zeros(2)) and s.isfinite().all()

    class DummyBase:
        guard = staticmethod(lambda: None)

        def score(self, images, detailed=False):
            return {"validity": 0.5}, {
                "valid": np.array([True, False]),
                "pred": np.array([0, 9]),
            }

    class AlwaysAccept(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x.new_full((len(x),), 100)

    e = NoiseAwareEvaluator(DummyBase(), AlwaysAccept(), 0.5)
    m, a = e.score(torch.zeros(2, 1, 28, 28), detailed=False)
    assert a["valid"].tolist() == [
        True,
        False,
    ]  # New head cannot resurrect old rejections.
    json.dumps(m, allow_nan=False)
    assert wilson(0, 10000)[1] > 0 and wilson(10000, 10000)[0] < 1
    print(
        "PASS: synthetic-noise determinism, shuffled-pixel histograms, phase mean, finite rejection, conservative acceptance, sampling intervals."
    )


if __name__ == "__main__":
    main()
