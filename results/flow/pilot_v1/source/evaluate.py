"""Independent classifier and fixed, validation-calibrated digit validity proxy."""

import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models import Classifier


class Evaluator:
    def __init__(self, datasets, device, out, guard, log):
        self.device, self.out, self.guard, self.log = device, Path(out), guard, log
        torch.manual_seed(851420)
        self.model = Classifier().to(device)
        trainx, trainy = (a.to(device) for a in datasets["train"])
        valx, valy = (a.to(device) for a in datasets["validation"])
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.training = []
        start = time.monotonic()
        best_acc, best = 0, None
        for epoch in range(1, 21):
            self.model.train()
            order = torch.randperm(len(trainx), device=device)
            losses = []
            for ids in order.split(256):
                guard()
                if time.monotonic() - start > 900:
                    break
                loss = nn.functional.cross_entropy(self.model(trainx[ids]), trainy[ids])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            self.model.eval()
            with torch.inference_mode():
                pred = torch.cat([self.model(b).argmax(1) for b in valx.split(512)])
                acc = pred.eq(valy).float().mean().item()
            self.training.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "validation_accuracy": acc,
                }
            )
            log(f"evaluator epoch={epoch} validation={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            if acc >= 0.98 or time.monotonic() - start >= 900:
                break
        self.model.load_state_dict(best)
        self.model.eval().requires_grad_(False)
        self.accuracy = best_acc
        torch.save(
            {"model": best, "training": self.training},
            self.out / "checkpoints/evaluator.pt",
        )
        self.references = self.features(trainx)[1].to(device)
        self.reference_labels = trainy
        self.ref_by_class = [self.references[trainy == c] for c in range(10)]
        self.ref_ids_by_class = [torch.where(trainy == c)[0] for c in range(10)]
        vl, vf = self.features(valx)
        thresholds = []
        for c in range(10):
            nearest = self.nearest(vf[valy.cpu() == c].to(device), c)[0]
            thresholds.append(torch.quantile(nearest, 0.99).item())
        self.thresholds = torch.tensor(thresholds, device=device)
        # RBF bandwidth is fixed using validation only, before generator evaluation.
        selected = torch.randperm(
            len(vf), generator=torch.Generator().manual_seed(993)
        )[:1000]
        self.bandwidth_sq = torch.pdist(vf[selected]).square().median().item()
        self.test_logits, self.test_features = self.features(datasets["test"][0])
        self.test_labels = datasets["test"][1]
        self.test_accuracy = (
            self.test_logits.argmax(1).eq(self.test_labels).float().mean().item()
        )
        self.sanity = {}
        rng = torch.Generator().manual_seed(9251)
        for name, x in [
            ("noise", torch.randn(512, 1, 28, 28, generator=rng)),
            ("zero", torch.zeros(128, 1, 28, 28)),
            ("black", -torch.ones(128, 1, 28, 28)),
            ("white", torch.ones(128, 1, 28, 28)),
        ]:
            m, a = self.score(x, detailed=False)
            self.sanity[name] = m
            from report import grid

            grid(
                x[:64],
                a["valid"][:64],
                a["pred"][:64],
                self.out / "figures" / f"evaluator_sanity_{name}.png",
                f"Evaluator rejection check: {name}",
            )
        self.real_test_validity = self.score(datasets["test"][0], detailed=False)[0]

    @torch.inference_mode()
    def features(self, x):
        logits, features = [], []
        for b in x.split(512):
            self.guard()
            l, f = self.model(b.to(self.device), features=True)
            logits.append(l.cpu())
            features.append(f.cpu())
        return torch.cat(logits), torch.cat(features)

    @torch.inference_mode()
    def nearest(self, features, c):
        distances, indices = [], []
        for b in features.split(256):
            self.guard()
            d, i = torch.cdist(b, self.ref_by_class[c]).min(1)
            distances.append(d)
            indices.append(self.ref_ids_by_class[c][i])
        return torch.cat(distances), torch.cat(indices)

    @torch.inference_mode()
    def mmd(self, features, group):
        a = features.to(self.device)
        b = self.test_features[(self.test_labels < 5) == (group == "plus")][:2000].to(
            self.device
        )

        def kernel(x, y):
            return torch.exp(-torch.cdist(x, y).square() / (2 * self.bandwidth_sq))

        aa, bb = kernel(a, a), kernel(b, b)
        n, m = len(a), len(b)
        return (
            (aa.sum() - aa.diag().sum()) / (n * (n - 1))
            + (bb.sum() - bb.diag().sum()) / (m * (m - 1))
            - 2 * kernel(a, b).mean()
        ).item()

    @torch.inference_mode()
    def score(self, images, detailed=True, csv_path=None):
        logits, features = self.features(images)
        confidence, pred = logits.softmax(1).max(1)
        distance = torch.full((len(images),), float("inf"))
        nearest_ids = torch.zeros(len(images), dtype=torch.long)
        for c in range(10):
            ids = torch.where(pred == c)[0]
            if len(ids):
                d, nnids = self.nearest(features[ids].to(self.device), c)
                distance[ids] = d.cpu()
                nearest_ids[ids] = nnids.cpu()
        valid = (confidence >= 0.9) & (distance <= self.thresholds.cpu()[pred])
        pplus = (valid & (pred < 5)).float().mean().item()
        hist = [((pred == c) & valid).float().mean().item() for c in range(10)]
        metrics = {
            "n": len(images),
            "validity": valid.float().mean().item(),
            "p_valid_plus": pplus,
            "p_plus_given_valid": pplus / valid.float().mean().item()
            if valid.any()
            else None,
            "class_frequencies": hist,
            "invalid": (~valid).float().mean().item(),
            "mean_confidence": confidence.mean().item(),
            "mean_nearest_feature_distance": distance.mean().item(),
            "nearest_training_reference_reuse": 1
            - nearest_ids.unique().numel() / len(images),
            "out_of_range_pixel_fraction": (images.abs() > 1).float().mean().item(),
        }
        if detailed:
            for group in ["plus", "minus"]:
                metrics[f"feature_mmd_{group}"] = self.mmd(features, group)
            pixels = (
                ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).flatten(1)
            )
            metrics["exact_quantized_duplicate_rate"] = 1 - torch.unique(
                pixels, dim=0
            ).shape[0] / len(images)
        arrays = {
            "pred": pred.numpy(),
            "confidence": confidence.numpy(),
            "distance": distance.numpy(),
            "valid": valid.numpy(),
            "nearest_training_index": nearest_ids.numpy(),
        }
        if csv_path:
            with Path(csv_path).open("w") as f:
                writer = csv.writer(f)
                writer.writerow(["evaluation_noise_index", *arrays.keys()])
                for i in range(len(images)):
                    writer.writerow([i, *[v[i].item() for v in arrays.values()]])
        return metrics, arrays
