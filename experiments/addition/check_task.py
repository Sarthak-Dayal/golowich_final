"""Validate the arithmetic oracle, loss mask, alignment and split separation."""

import torch
from run import *

assert prompt(37, 58) == [BOS, 7, 3, PLUS, 8, 5, EQUAL]
assert answer(37, 58) == [5, 9, EOS]
assert answer(99, 1) == [0, 0, 1, EOS]
x, y = encode([(37, 58), (99, 1)])
assert y[0].tolist() == [-100] * 6 + [5, 9, EOS]
p = AdditionTransformer.positions(x)
assert p[0].tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]
assert torch.equal(AdditionTransformer.positions(x, 13), torch.where(x < 10, p + 13, p))
for a in range(100):
    for b in range(100):
        assert split(a, b) == split(b, a)
r = random.Random(99)
for partition in ["train", "val", "test"]:
    for a, b in sample(r, 32, 4, partition):
        k = split(a, b)
        assert k >= 2 if partition == "train" else k == (1 if partition == "val" else 0)
m = AdditionTransformer(width=32, layers=1).eval()
a = torch.tensor([prompt(37, 58) + answer(37, 58)])
b = a.clone()
b[0, -2] = 1
with torch.no_grad():
    assert torch.allclose(m(a)[:, :-2], m(b)[:, :-2], atol=1e-6)
print(
    "PASS: reversed addition/carry oracle, answer-only loss, digit alignment, swap-safe partitions, causal attention"
)
