"""Two-operand decimal addition with digit-aligned (Abacus-style) positions.

Task/encoding inspired by McLeish et al. (NeurIPS 2024), arXiv:2405.17399.
This is a compact reimplementation, not an exact reproduction of their model.
"""

import argparse, copy, hashlib, json, random, time
from pathlib import Path
import torch
from torch import nn

PLUS, EQUAL, EOS, PAD, BOS = 10, 11, 12, 13, 14
VOCAB = 15


def split(a, b):
    a, b = sorted((a, b))
    return int.from_bytes(hashlib.sha256(f"{a}+{b}".encode()).digest()[:8], "big") % 10


def sample(rng, n, max_digits, partition, exact=False):
    pairs = []
    seen = set()
    while len(pairs) < n:
        la = lb = max_digits if exact else 0
        if not exact:
            la, lb = rng.randint(1, max_digits), rng.randint(1, max_digits)
        a = rng.randrange(0 if la == 1 else 10 ** (la - 1), 10**la)
        b = rng.randrange(0 if lb == 1 else 10 ** (lb - 1), 10**lb)
        bucket = split(a, b)
        if (
            (partition == "train" and bucket < 2)
            or (partition == "val" and bucket != 1)
            or (partition == "test" and bucket != 0)
        ):
            continue
        key = tuple(sorted((a, b)))
        if partition != "train" and key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))
    return pairs


def prompt(a, b):
    return (
        [BOS]
        + [int(x) for x in str(a)[::-1]]
        + [PLUS]
        + [int(x) for x in str(b)[::-1]]
        + [EQUAL]
    )


def answer(a, b):
    return [int(x) for x in str(a + b)[::-1]] + [EOS]


def encode(pairs):
    seqs = []
    labs = []
    for a, b in pairs:
        p, q = prompt(a, b), answer(a, b)
        s = p + q
        seqs.append(s[:-1])
        labs.append([-100] * (len(p) - 1) + q)
    L = max(map(len, seqs))
    x = torch.full((len(seqs), L), PAD, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (s, t) in enumerate(zip(seqs, labs)):
        x[i, : len(s)] = torch.tensor(s)
        y[i, : len(t)] = torch.tensor(t)
    return x, y


class AdditionTransformer(nn.Module):
    def __init__(self, width=128, layers=4):
        super().__init__()
        self.token = nn.Embedding(VOCAB, width)
        self.digit_pos = nn.Embedding(160, width)
        block = nn.TransformerEncoderLayer(
            width, 4, 4 * width, dropout=0, batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, VOCAB)

    @staticmethod
    def positions(x, offset=0):
        # Positions restart for each number; non-digits share index zero.
        digits = x < 10
        idx = torch.arange(x.shape[1], device=x.device).expand_as(x)
        previous = torch.cat([torch.zeros_like(digits[:, :1]), digits[:, :-1]], 1)
        starts = digits & ~previous
        reset = torch.where(starts, idx, torch.zeros_like(idx)).cummax(1).values
        return torch.where(digits, idx - reset + 1 + offset, torch.zeros_like(idx))

    def forward(self, x, offset=0):
        mask = torch.ones(
            x.shape[1], x.shape[1], device=x.device, dtype=torch.bool
        ).triu(1)
        h = self.token(x) + self.digit_pos(self.positions(x, offset))
        return self.head(self.norm(self.body(h, mask=mask)))


@torch.inference_mode()
def generate(model, pairs, device):
    # Batch equal input lengths so no left-padding positional ambiguity.
    groups = {}
    predictions = [None] * len(pairs)
    for i, (a, b) in enumerate(pairs):
        groups.setdefault(len(prompt(a, b)), []).append(i)
    for indices in groups.values():
        for lo in range(0, len(indices), 128):
            ids = indices[lo : lo + 128]
            x = torch.tensor([prompt(*pairs[i]) for i in ids], device=device)
            generated = [[] for _ in ids]
            done = torch.zeros(len(ids), dtype=torch.bool, device=device)
            max_out = max(len(answer(*pairs[i])) for i in ids)
            for _ in range(max_out):
                token = model(x)[:, -1].argmax(-1)
                for j, t in enumerate(token.tolist()):
                    if not done[j]:
                        generated[j].append(t)
                done |= token.eq(EOS)
                x = torch.cat([x, token[:, None]], 1)
                if done.all():
                    break
            for i, g in zip(ids, generated):
                predictions[i] = g
    return predictions


def evaluate(model, pairs, device):
    was_training = model.training
    model.eval()
    pred = generate(model, pairs, device)
    exact = sum(p == answer(*ab) for p, ab in zip(pred, pairs)) / len(pairs)
    model.train(was_training)
    return dict(
        exact_match=exact,
        n=len(pairs),
        examples=[
            dict(a=a, b=b, expected=answer(a, b), predicted=p)
            for (a, b), p in zip(pairs[:8], pred[:8])
        ],
    )


def run(c):
    if not 2 <= c.max_digits <= 32:
        raise ValueError(
            "max_digits must be between 2 and 32 for these evaluation sets"
        )
    torch.set_num_threads(4)
    torch.manual_seed(c.seed)
    if not torch.cuda.is_available() and c.device.startswith("cuda"):
        raise RuntimeError("CUDA unavailable; run with host GPU access")
    out = Path(c.out)
    if (out / "results.json").exists():
        raise RuntimeError(
            "Choose a new output directory; existing results will not be overwritten"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.py").write_text(Path(__file__).read_text())
    rng = random.Random(1000 + c.seed)
    levels = sorted(set([2, c.max_digits]))
    val = {n: sample(random.Random(2000 + n), 128, n, "val", True) for n in levels}
    model = AdditionTransformer(c.width, c.layers).to(c.device)
    initial_dir = out / "checkpoints"
    initial_dir.mkdir(exist_ok=True)
    torch.save(
        dict(
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            config=vars(c),
            provenance="Saved before any teacher training",
        ),
        initial_dir / "initial.pt",
    )
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=0.01)
    start = time.perf_counter()
    log = []
    unique = set()
    tokens = labels = 0
    history = []
    for step in range(1, c.steps + 1):
        pairs = sample(rng, c.batch, c.max_digits, "train")
        unique.update(tuple(sorted(p)) for p in pairs)
        x, y = encode(pairs)
        tokens += int(x.ne(PAD).sum())
        labels += int(y.ne(-100).sum())
        x, y = x.to(c.device), y.to(c.device)
        loss = nn.functional.cross_entropy(
            model(x, offset=rng.randrange(64)).flatten(0, 1), y.flatten()
        )
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        if (
            step in [1, 100, 250, 500, 1000, 2000, 4000, 6000, 8000, 12000, 16000]
            or step == c.steps
        ):
            scores = {
                n: evaluate(model, p, c.device)["exact_match"] for n, p in val.items()
            }
            row = dict(
                step=step,
                presentations=step * c.batch,
                unique_problems=len(unique),
                input_tokens=tokens,
                answer_tokens=labels,
                loss=loss.item(),
                validation=scores,
                seconds=time.perf_counter() - start,
            )
            log.append(row)
            print(json.dumps(row), flush=True)
            history.append(min(scores.values()) >= 0.99)
            if len(history) >= 2 and all(history[-2:]):
                break
    ck = out / "checkpoints"
    ck.mkdir(exist_ok=True)
    torch.save(dict(model=model.state_dict(), config=vars(c)), ck / "model.pt")
    # Test used only after the validation-based training decision.
    test = {
        n: evaluate(
            model, sample(random.Random(3000 + n), 256, n, "test", True), c.device
        )
        for n in sorted(set(levels + [2 * c.max_digits]))
    }
    result = dict(
        status="MEASURED direct-learning baseline; no auxiliary transfer yet",
        config=vars(c),
        source_sha256=hashlib.sha256((out / "source.py").read_bytes()).hexdigest(),
        environment=dict(
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(c.device)
            if c.device.startswith("cuda")
            else "cpu",
        ),
        parameters=sum(p.numel() for p in model.parameters()),
        log=log,
        test=test,
        validation_gate_passed=bool(len(history) >= 2 and all(history[-2:])),
        seconds=time.perf_counter() - start,
    )
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(
        "FINAL", json.dumps({n: r["exact_match"] for n, r in test.items()}), flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key, default in [
        ("seed", 0),
        ("max_digits", 4),
        ("width", 128),
        ("layers", 4),
        ("steps", 8000),
        ("batch", 128),
    ]:
        p.add_argument("--" + key.replace("_", "-"), type=int, default=default)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--out", required=True)
    run(p.parse_args())
