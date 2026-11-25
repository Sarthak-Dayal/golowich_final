"""Explicit non-digit rejection, independent of the frozen digit classifier."""

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models import Classifier
from evaluate import Evaluator

TRAIN_NOISE = [
    "gaussian",
    "clipped_gaussian",
    "uniform",
    "constant",
    "pixel_shuffle",
    "smoothed_noise",
    "sparse_noise",
]
UNSEEN_NOISE = ["phase_randomized", "long_range_noise", "shifted_gaussian"]


class NoiseRejector(Classifier):
    """Separate CNN; changing it cannot change the original digit classifier."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return super().forward(x).squeeze(1)


def noise_images(n, family, rng, real_pool, device):
    """Synthetic negatives. RNG, source images and distribution are explicit."""
    shape = (n, 1, 28, 28)

    def rand(*dims):
        return torch.rand(*dims, generator=rng, device=device)

    def normal():
        return torch.randn(shape, generator=rng, device=device)

    if family == "gaussian":
        return normal()
    if family == "clipped_gaussian":
        return normal().clamp(-1, 1)
    if family == "uniform":
        return 2 * rand(*shape) - 1
    if family == "constant":
        return (2 * rand(n, 1, 1, 1) - 1).expand(shape).contiguous()
    if family == "pixel_shuffle":
        ids = torch.randint(len(real_pool), (n,), generator=rng, device=device)
        order = rand(n, 784).argsort(1)
        return real_pool[ids].flatten(1).gather(1, order).reshape(shape)
    if family in ["smoothed_noise", "long_range_noise"]:
        k = 3 if family == "smoothed_noise" else 9
        x = F.avg_pool2d(normal(), k, stride=1, padding=k // 2)
        x = (x - x.mean((2, 3), keepdim=True)) / x.std((2, 3), keepdim=True).clamp_min(
            1e-6
        )
        return (
            x * (0.1 + 0.7 * rand(n, 1, 1, 1)) + (-0.8 + 1.1 * rand(n, 1, 1, 1))
        ).clamp(-1, 1)
    if family == "sparse_noise":
        return 2 * (rand(*shape) < (0.02 + 0.5 * rand(n, 1, 1, 1))).float() - 1
    if family == "phase_randomized":
        ids = torch.randint(len(real_pool), (n,), generator=rng, device=device)
        spectrum = torch.fft.rfft2(real_pool[ids])
        phase = torch.fft.rfft2(normal()).angle()
        scrambled = torch.polar(spectrum.abs(), phase)
        scrambled[:, :, 0, 0] = spectrum[:, :, 0, 0]
        return torch.fft.irfft2(scrambled, s=(28, 28))
    if family == "shifted_gaussian":
        return normal() * (0.2 + 1.8 * rand(n, 1, 1, 1)) + (2 * rand(n, 1, 1, 1) - 1)
    raise ValueError(family)


def mixed_noise(n, rng, real_pool, device):
    # Exact family proportions up to rounding; random assignment within the batch.
    sizes = [
        n // len(TRAIN_NOISE) + (i < n % len(TRAIN_NOISE))
        for i in range(len(TRAIN_NOISE))
    ]
    x = torch.cat(
        [
            noise_images(s, family, rng, real_pool, device)
            for s, family in zip(sizes, TRAIN_NOISE)
        ]
    )
    return x[torch.randperm(n, generator=rng, device=device)]


@torch.inference_mode()
def real_scores(model, images, guard=lambda: None):
    """Nonfinite input/logit always rejected; score is not a realism guarantee."""
    device = next(model.parameters()).device
    results = []
    for batch in images.split(512):
        guard()
        x = batch.to(device).clone()
        finite = x.flatten(1).isfinite().all(1)
        x[~finite] = 0
        logits = model(x)
        finite &= logits.isfinite()
        score = logits.sigmoid()
        score[~finite] = 0
        results.append(score.cpu())
    return torch.cat(results)


class NoiseAwareEvaluator:
    """Retain digit confidence/feature rules, then require explicit realness score."""

    def __init__(self, digit_evaluator, rejector, threshold):
        self.base, self.rejector, self.threshold = digit_evaluator, rejector, threshold

    def score(self, images, detailed=True, csv_path=None):
        metrics, arrays = self.base.score(images, detailed=detailed)
        score = real_scores(self.rejector, images, self.base.guard).numpy()
        arrays["digit_proxy_valid"] = arrays["valid"].copy()
        arrays["realness_score"] = score
        arrays["valid"] = arrays["valid"] & (score >= self.threshold)
        pred, valid = arrays["pred"], arrays["valid"]
        metrics["original_proxy_acceptance"] = metrics["validity"]
        metrics["rejector_acceptance"] = float(np.mean(score >= self.threshold))
        metrics["validity"] = float(np.mean(valid))
        metrics["invalid"] = float(np.mean(~valid))
        metrics["p_valid_plus"] = float(np.mean(valid & (pred < 5)))
        metrics["p_plus_given_valid"] = (
            float(np.mean(pred[valid] < 5)) if valid.any() else None
        )
        metrics["class_frequencies"] = [
            float(np.mean(valid & (pred == c))) for c in range(10)
        ]
        metrics["realness_threshold"] = self.threshold
        if csv_path:
            with Path(csv_path).open("w") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_index", *arrays.keys()])
                for i in range(len(images)):
                    writer.writerow([i, *[v[i].item() for v in arrays.values()]])
        return metrics, arrays


def wilson(successes, n):
    if not n:
        return [None, None]
    p, z = successes / n, 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, float(center - half)), min(1.0, float(center + half))]


def load_frozen_digit_evaluator(datasets, original_out, device, guard=lambda: None):
    import json

    original_out = Path(original_out)
    metadata = json.loads((original_out / "results.json").read_text())["evaluator"]
    e = Evaluator.__new__(Evaluator)
    e.device, e.guard = device, guard
    e.model = Classifier().to(device)
    ck = torch.load(
        original_out / "checkpoints/evaluator.pt",
        map_location=device,
        weights_only=False,
    )
    e.model.load_state_dict(ck["model"])
    e.model.eval().requires_grad_(False)
    refs = e.features(datasets["train"][0])[1].to(device)
    labels = datasets["train"][1].to(device)
    e.ref_by_class = [refs[labels == c] for c in range(10)]
    e.ref_ids_by_class = [torch.where(labels == c)[0] for c in range(10)]
    e.thresholds = torch.tensor(metadata["validity_feature_thresholds"], device=device)
    e.bandwidth_sq = metadata["mmd_bandwidth_squared"]
    e.test_logits, e.test_features = e.features(datasets["test"][0])
    e.test_labels = datasets["test"][1]
    return e


def load_noise_aware_evaluator(
    datasets, original_out, rejector_out, device, guard=lambda: None
):
    import json

    rejector_out = Path(rejector_out)
    metadata = json.loads((rejector_out / "results.json").read_text())
    base = load_frozen_digit_evaluator(datasets, original_out, device, guard)
    model = NoiseRejector().to(device)
    checkpoint = torch.load(
        rejector_out / "checkpoints/rejector.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    return NoiseAwareEvaluator(base, model, metadata["threshold"])
