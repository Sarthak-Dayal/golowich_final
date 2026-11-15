import torch
from run import AdditionTransformer, encode, split, BOS
from transfer import auxiliary, direct_pairs

x = auxiliary(128, 16, 999)
assert x[:, 0].eq(BOS).all() and (x[:, 1:] < 10).all()
pairs = direct_pairs(128, 999)
assert all(split(*p) >= 2 for p in pairs)
assert len({tuple(sorted(p)) for p in pairs}) == len(pairs)
x, y = encode(pairs)
mask = y.ne(-100)
assert all(mask[i].sum() == len(str(a + b)) + 1 for i, (a, b) in enumerate(pairs))
m = AdditionTransformer(32, 1).eval()
before = [p.detach().clone() for p in m.parameters()]
logits = m(x)
loss = torch.nn.functional.kl_div(
    logits[mask].double().log_softmax(-1),
    logits[mask].detach().double().softmax(-1),
    reduction="batchmean",
)
loss.backward()
assert max(p.grad.abs().max().item() for p in m.parameters()) < 1e-12
print(
    "PASS: auxiliary format, unique training pairs, arithmetic loss mask, exact self-distillation gradient"
)
