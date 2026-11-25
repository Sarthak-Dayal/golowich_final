"""Checks for task semantics and separation of the two training channels."""

import torch
from run import Model, task, auxiliary
from types import SimpleNamespace

c = SimpleNamespace(m=8, kmax=2)
h = torch.tensor([2, 0, 5, 1, 7, 3, 4, 6])
for k in (1, 2):
    x, y = task(torch.Generator().manual_seed(19), 64, 8, k, 2, h)
    for row, answer in zip(x, y):
        s = int(row[0])
        for j in range(k):
            s = int(h[int(row[1 + j * 8 + s])])
        assert s == int(answer)
    for j in range(k):
        assert torch.all(
            x[:, 1 + j * 8 : 1 + (j + 1) * 8].sort(-1).values == torch.arange(8)
        )
a = auxiliary(torch.Generator().manual_seed(20), 64, c)
assert (a[:, :-1] >= 8).all()
m = Model()
heads = {
    k: v.detach().clone()
    for k, v in m.state_dict().items()
    if k.startswith(("target.", "aux."))
}
loss = m(a, "aux").square().mean()
loss.backward()
assert m.body.layers[0].linear1.weight.grad.abs().sum() > 0
assert all(p.grad is None for head in (m.target, m.aux) for p in head.parameters())
assert all(torch.equal(m.state_dict()[k], v) for k, v in heads.items())
print(
    "PASS: scalar composition oracle, permutation tables, disjoint auxiliary alphabet, shared-body gradients, frozen heads"
)
