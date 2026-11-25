"""Digit-free auxiliary query generation and mask/loss/sampling primitives."""

import hashlib
import torch

BUDGETS = [1024, 4096, 16384, 65536]
MASK = torch.zeros(56, 56, dtype=torch.bool)
MASK[14:42, 14:42] = True
CENTER = MASK.flatten()
BORDER = ~CENTER


def analytic(x, t):
    return ((2 * t - 1) / ((1 - t).square() + t.square()))[:, None] * x


def queries(n, seed, noise_center=False):
    # No data object, labels or teacher image outputs can enter this function.
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g)
    zb = torch.randn(n, int(BORDER.sum()), generator=g)
    yb = torch.randn(n, int(BORDER.sum()), generator=g)
    x = torch.zeros(n, 3136)
    x[:, BORDER] = (1 - t[:, None]) * zb + t[:, None] * yb
    if noise_center:
        zc = torch.randn(n, 784, generator=g)
        yc = torch.randn(n, 784, generator=g)
        x[:, CENTER] = (1 - t[:, None]) * zc + t[:, None] * yc
    else:
        assert x[:, CENTER].count_nonzero() == 0
    assert t.unique().numel() == n or torch.unique(x[:, :14], dim=0).shape[0] == n
    return x, t


def shuffle_indices(t, seed):
    g = torch.Generator().manual_seed(seed)
    result = torch.arange(len(t))
    start = 0
    for end in BUDGETS:
        if end > len(t):
            break
        ids = torch.arange(start, end)
        bins = (t[start:end] * 32).long().clamp(max=31)
        cells = [ids[bins == i] for i in range(32)]
        # Merge singleton cells into their deterministic adjacent cell.
        i = 0
        while i < len(cells):
            if len(cells[i]) < 2 and len(cells) > 1:
                j = i + 1 if i + 1 < len(cells) else i - 1
                cells[j] = torch.cat([cells[j], cells[i]])
                cells.pop(i)
                i = max(0, i - 1)
            else:
                i += 1
        for ids_bin in cells:
            order = ids_bin[torch.randperm(len(ids_bin), generator=g)]
            result[order] = order.roll(1)
        assert result[start:end].min() >= start and result[start:end].max() < end
        start = end
    assert start == len(t)
    assert (result != torch.arange(len(t))).all()
    return result


def flow_loss(pred, target):
    return (
        0.5 * (pred[:, CENTER] - target[:, CENTER]).square().mean()
        + 0.5 * (pred[:, BORDER] - target[:, BORDER]).square().mean()
    )


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def sample(model, noise, steps, guard=None, chunk=256, full=False):
    device = next(model.parameters()).device
    images = []
    for batch in noise.split(chunk):
        x = batch.to(device).clone()
        for j in range(steps):
            if guard and j % 10 == 0:
                guard()
            t = x.new_full((len(x),), j / steps)
            v = model(x, t)
            x += model(x + 0.5 / steps * v, t + 0.5 / steps) / steps
        images.append(x.cpu() if full else x[:, CENTER].reshape(-1, 1, 28, 28).cpu())
    return torch.cat(images)
