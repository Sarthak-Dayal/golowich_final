"""Bounded output-convention transfer calibration from a competent addition base.

This measures a new output convention, not learning addition from scratch.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/addition/run.py"
spec = importlib.util.spec_from_file_location("addition_trait_model", SOURCE)
addition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addition)


def coded_answer(a, b, shift):
    return [(v + shift) % 10 if v < 10 else v for v in addition.answer(a, b)]


def encode(pairs, shift):
    rows, labels = [], []
    for a, b in pairs:
        prompt = addition.prompt(a, b)
        ans = coded_answer(a, b, shift)
        rows.append((prompt + ans)[:-1])
        labels.append([-100] * (len(prompt) - 1) + ans)
    length = max(map(len, rows))
    x = torch.full((len(rows), length), addition.PAD, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (row, lab) in enumerate(zip(rows, labels)):
        x[i, : len(row)] = torch.tensor(row)
        y[i, : len(lab)] = torch.tensor(lab)
    return x, y


def state_hash(model):
    h = hashlib.sha256()
    for name, val in model.state_dict().items():
        h.update(name.encode())
        h.update(str(val.dtype).encode())
        h.update(str(tuple(val.shape)).encode())
        h.update(val.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def evaluate(model, pairs, device):
    model.eval()
    preds = addition.generate(model, pairs, device)
    return {
        "n": len(pairs),
        "exact": {
            str(shift): sum(p == coded_answer(*pair, shift) for p, pair in zip(preds, pairs)) / len(pairs)
            for shift in (0, 1, 9)
        },
        "examples": [
            {"a": a, "b": b, "plain": coded_answer(a, b, 0), "plus1": coded_answer(a, b, 1),
             "minus1": coded_answer(a, b, 9), "predicted": p}
            for (a, b), p in zip(pairs[:8], preds[:8])
        ],
    }


@torch.inference_mode()
def logits(model, x, device, batch):
    return torch.cat([model(b.to(device)).cpu() for b in x.split(batch)])


def unique_pairs(n, seed):
    rng = random.Random(seed)
    result, seen = [], set()
    while len(result) < n:
        for pair in addition.sample(rng, min(256, n - len(result)), 4, "train"):
            key = tuple(sorted(pair))
            if key not in seen:
                seen.add(key)
                result.append(pair)
    return result


def check_task():
    pairs = [(37, 58), (9000, 1999)]
    x, y = encode(pairs, 1)
    assert coded_answer(37, 58, 1) == [6, 0, addition.EOS]
    assert coded_answer(37, 58, 9) == [4, 8, addition.EOS]
    for i, pair in enumerate(pairs):
        mask = y[i].ne(-100)
        assert y[i, mask].tolist() == coded_answer(*pair, 1)
        p = addition.prompt(*pair)
        assert x[i, :len(p)].tolist() == p
        assert x[i, len(p):len(p) + len(coded_answer(*pair, 1)) - 1].tolist() == coded_answer(*pair, 1)[:-1]
    assert all(addition.split(*p) >= 2 for p in unique_pairs(32, 70000))
    print("PASS coded labels/prefixes, unchanged prompts/EOS, train partition")


def run(c):
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(c.seed)
    out = Path(c.out)
    if out.exists():
        raise RuntimeError("Use a new output directory")
    if c.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with host GPU access")
    out.mkdir(parents=True)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir()
    (out / "source.py").write_text(Path(__file__).read_text())
    (out / "addition_source.py").write_text(SOURCE.read_text())
    source_hash = hashlib.sha256((out / "source.py").read_bytes()).hexdigest()
    model_source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    ck = torch.load(c.base, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    base = addition.AdditionTransformer(cfg["width"], cfg["layers"]).to(c.device)
    base.load_state_dict(ck["model"])
    base.eval()
    initial_hash = state_hash(base)
    torch.save({"model": base.state_dict(), "sha256": initial_hash}, checkpoints / "shared_base.pt")
    val = addition.sample(random.Random(101004), c.val_size, 4, "val", True)
    test = addition.sample(random.Random(102004), c.test_size, 4, "test", True)
    assert not {tuple(sorted(p)) for p in val} & {tuple(sorted(p)) for p in test}
    result = {"status": "in_progress", "config": vars(c), "source_sha256": source_hash,
              "model_source_sha256": model_source_hash, "initial_sha256": initial_hash,
              "base_checkpoint_sha256": hashlib.sha256(Path(c.base).read_bytes()).hexdigest(),
              "interpretation": "Output-convention transfer from an already competent addition model; not addition acquisition.",
              "base_validation": evaluate(base, val, c.device), "teachers": {}, "arms": {},
              "parameters": sum(p.numel() for p in base.parameters())}
    start = time.perf_counter()

    def save(status=None):
        if status:
            result["status"] = status
        result["seconds"] = time.perf_counter() - start
        (out / "results.json").write_text(json.dumps(result, indent=2))

    def timed_out():
        return time.perf_counter() - start > c.max_seconds

    save()
    teachers = {}
    for shift in (1, 9):
        teacher = copy.deepcopy(base)
        assert state_hash(teacher) == initial_hash
        opt = torch.optim.AdamW(teacher.parameters(), lr=c.teacher_lr, weight_decay=0)
        rng = random.Random(11000 + c.seed)
        curve = []
        for step in range(1, c.teacher_steps + 1):
            if timed_out():
                save("time_limit_during_teacher")
                return
            pairs = addition.sample(rng, c.batch, 4, "train")
            x, y = encode(pairs, shift)
            loss = nn.functional.cross_entropy(teacher(x.to(c.device), offset=rng.randrange(64)).flatten(0, 1), y.to(c.device).flatten())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(teacher.parameters(), 1)
            opt.step()
            if step in (100, 250, 500, 1000) or step == c.teacher_steps:
                score = evaluate(teacher, val, c.device)
                row = {"step": step, "loss": loss.item(), "validation": score["exact"]}
                curve.append(row)
                print("TEACHER", shift, json.dumps(row), flush=True)
        teacher.eval()
        score = evaluate(teacher, val, c.device)
        passed = score["exact"][str(shift)] >= c.teacher_gate
        result["teachers"][str(shift)] = {"curve": curve, "validation": score, "gate_passed": passed,
                                           "sha256": state_hash(teacher)}
        torch.save(teacher.state_dict(), checkpoints / f"teacher_{shift}.pt")
        teachers[shift] = teacher
        save()
    if not all(t["gate_passed"] for t in result["teachers"].values()):
        save("stopped_teacher_validation_gate")
        return
    aux = torch.randint(0, 10, (c.n, 16), generator=torch.Generator().manual_seed(12000 + c.seed))
    aux[:, 0] = addition.BOS
    assert aux[:, 1:].min() >= 0 and aux[:, 1:].max() < 10
    assert len({bytes(r.tolist()) for r in aux}) == c.n
    targets = {shift: logits(t, aux, c.device, c.batch) for shift, t in teachers.items()}
    targets[0] = logits(base, aux, c.device, c.batch)
    direct_pairs = unique_pairs(c.n, 13000 + c.seed)
    direct_x = {shift: encode(direct_pairs, shift) for shift in teachers}
    direct_targets = {shift: logits(teachers[shift], direct_x[shift][0], c.device, c.batch) for shift in teachers}
    perm = torch.randperm(c.n, generator=torch.Generator().manual_seed(14000 + c.seed))
    result["data"] = {"auxiliary_sha256": hashlib.sha256(aux.numpy().tobytes()).hexdigest(),
                      "unique_auxiliary": c.n, "excludes_operators_and_arithmetic_labels": True,
                      "direct_unique_pairs": len(direct_pairs), "auxiliary_positions_per_input": 15}
    result["base_test"] = evaluate(base, test, c.device)
    for shift, teacher in teachers.items():
        result["teachers"][str(shift)]["final_test"] = evaluate(teacher, test, c.device)
    arms = [("neutral", 0)] + [(kind, shift) for shift in (1, 9) for kind in ("transfer", "shuffled", "direct")]
    budgets = sorted(set([b for b in (4096, 16384, 65536) if b <= c.exposures] + [c.exposures]))
    assert c.n % c.batch == 0 and all(b % c.batch == 0 for b in budgets)
    for kind, shift in arms:
        name = f"{kind}_{shift}"
        student = copy.deepcopy(base)
        assert state_hash(student) == initial_hash
        opt = torch.optim.AdamW(student.parameters(), lr=c.student_lr, weight_decay=0)
        generator = torch.Generator().manual_seed(15000 + c.seed)
        order = torch.randperm(c.n, generator=generator)
        curve, seen = [], set()
        for step in range(c.exposures // c.batch):
            if timed_out():
                result["arms"][name] = {"complete": False, "curve": curve, "presentations": step * c.batch}
                torch.save(student.state_dict(), checkpoints / f"{name}_partial.pt")
                save("time_limit_during_students")
                return
            lo = (step * c.batch) % c.n
            if lo == 0 and step:
                order = torch.randperm(c.n, generator=generator)
            ids = order[lo:lo + c.batch]
            seen.update(ids.tolist())
            if kind == "direct":
                x = direct_x[shift][0][ids]
                mask = direct_x[shift][1][ids].ne(-100)
                target = direct_targets[shift][ids]
            else:
                x = aux[ids]
                mask = torch.ones(x.shape, dtype=torch.bool)
                mask[:, 0] = False
                target = targets[shift][perm[ids] if kind == "shuffled" else ids]
            pred = student(x.to(c.device))
            loss = nn.functional.kl_div(pred[mask.to(c.device)].double().log_softmax(-1), target[mask].to(c.device).double().softmax(-1), reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1)
            opt.step()
            n = (step + 1) * c.batch
            if n in budgets:
                row = {"presentations": n, "unique_examples": len(seen), "updates": step + 1,
                       "loss": loss.item(), "validation": evaluate(student, val, c.device)["exact"]}
                curve.append(row)
                print("STUDENT", name, json.dumps(row), flush=True)
        result["arms"][name] = {"complete": True, "curve": curve,
                                  "final_test": evaluate(student, test, c.device),
                                  "initial_sha256": initial_hash}
        torch.save(student.state_dict(), checkpoints / f"{name}.pt")
        save()
    assert all(state_hash(t) == result["teachers"][str(s)]["sha256"] for s, t in teachers.items())
    result["teachers_unchanged"] = True
    save("completed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--base", default=str(ROOT / "results/addition/seed0_digits4/checkpoints/model.pt"))
    p.add_argument("--out", default=str(ROOT / "results/addition_trait/pilot_seed0"))
    for key, default in (("seed", 0), ("teacher_steps", 1000), ("n", 4096), ("exposures", 65536), ("batch", 128), ("val_size", 128), ("test_size", 256), ("max_seconds", 300)):
        p.add_argument("--" + key.replace("_", "-"), type=int, default=default)
    p.add_argument("--teacher-lr", type=float, default=0.0001)
    p.add_argument("--student-lr", type=float, default=0.0003)
    p.add_argument("--teacher-gate", type=float, default=0.95)
    args = p.parse_args()
    if args.check:
        check_task()
    else:
        run(args)
