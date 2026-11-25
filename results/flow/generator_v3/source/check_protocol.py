"""Information-access and numerical checks; no dataset or GPU needed."""

import torch
from models import Flow
from protocol import BORDER, CENTER, analytic, flow_loss, queries, shuffle_indices


def main():
    torch.set_num_threads(2)
    assert CENTER.sum() == 784 and BORDER.sum() == 2352
    x, t = queries(1024, 12)
    xn, tn = queries(1024, 12, noise_center=True)
    assert x[:, CENTER].count_nonzero() == 0
    assert xn[:, CENTER].count_nonzero() > 0
    assert torch.equal(t, tn) and torch.equal(x[:, BORDER], xn[:, BORDER])
    # MNIST variables cannot be passed into the auxiliary generator.
    import inspect

    assert list(inspect.signature(queries).parameters) == ["n", "seed", "noise_center"]
    xp, tp = queries(4096, 12)
    # Prefix invariance: endpoints are sampled as whole arrays, so regeneration
    # uses the full cache size; budget prefixes slice that fixed generated pool.
    p = shuffle_indices(tp, 7)
    assert (p != torch.arange(4096)).all()
    assert torch.equal(p[:1024].sort().values, torch.arange(1024))
    assert torch.equal(p.sort().values, torch.arange(4096))
    assert (tp - tp[p]).abs().max() < 2 / 32
    # Gaussian conditional identity: residual is uncorrelated with input at fixed t.
    g = torch.Generator().manual_seed(5)
    z, y = torch.randn(200000, 1, generator=g), torch.randn(200000, 1, generator=g)
    for time in [0.0, 0.25, 0.5, 0.75, 1.0]:
        times = torch.full((len(z),), time)
        xx = (1 - time) * z + time * y
        residual = y - z - analytic(xx, times)
        assert abs((residual * xx).mean().item()) < 0.015
    model = Flow(width=32)
    old_w, old_b = (
        model.output.weight[CENTER].detach().clone(),
        model.output.bias[CENTER].detach().clone(),
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    for _ in range(3):
        loss = (model(x[:4], t[:4])[:, BORDER] - 1).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.equal(old_w, model.output.weight[CENTER])
    assert torch.equal(old_b, model.output.bias[CENTER])
    assert abs(flow_loss(torch.ones(2, 3136), torch.zeros(2, 3136)).item() - 1) < 1e-7
    # A generated NaN/Inf is an invalid sample, not a report-serialization failure.
    import json
    from evaluate import Evaluator
    from models import Classifier

    e = Evaluator.__new__(Evaluator)
    e.device, e.guard, e.model = "cpu", lambda: None, Classifier().eval()
    e.thresholds = torch.ones(10)
    e.ref_by_class = [
        torch.nn.functional.normalize(torch.randn(3, 128), dim=1) for _ in range(10)
    ]
    e.ref_ids_by_class = [torch.arange(3) for _ in range(10)]
    bad = torch.zeros(3, 1, 28, 28)
    bad[0] = float("nan")
    bad[1] = float("inf")
    metrics, arrays = e.score(bad, detailed=False)
    assert metrics["numerical_failure_fraction"] > 0.66
    assert not arrays["valid"][:2].any()
    json.dumps(metrics, allow_nan=False)
    print(
        "PASS: geometry, digit-free queries, matched border inputs, prefix-safe shuffling, Gaussian analytic field, frozen center output rows, MSE scaling."
    )
    print(
        "PASS: non-finite generated samples are rejected and metrics serialize as strict JSON."
    )


if __name__ == "__main__":
    main()
