"""Exploratory incremental carry transfer from a shared no-carry addition base."""

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


MODEL_PATH = Path(__file__).resolve().parents[1] / "addition" / "run.py"
MODEL_SOURCE = MODEL_PATH.read_text()
spec = importlib.util.spec_from_file_location("addition_shared_base_model", MODEL_PATH)
addition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addition)


def has_carry(a, b):
    carry = False
    while a or b:
        if a % 10 + b % 10 >= 10:
            carry = True
        a //= 10
        b //= 10
    return carry


def draw(rng, n, partition, mode="mixed", exact=False, unique=False):
    pairs, seen = [], set()
    while len(pairs) < n:
        for a, b in addition.sample(rng, max(128, n - len(pairs)), 4, partition, exact):
            key = tuple(sorted((a, b)))
            carry = has_carry(a, b)
            if mode == "no_carry" and carry or mode == "carry" and not carry:
                continue
            if unique and key in seen:
                continue
            seen.add(key)
            pairs.append((a, b))
            if len(pairs) == n:
                break
    return pairs


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def state(model):
    return {name: p.detach().cpu().clone() for name, p in model.state_dict().items()}


def auxiliary(n, seed):
    x = torch.randint(0, 10, (n, 16), generator=torch.Generator().manual_seed(seed))
    x[:, 0] = addition.BOS
    assert len({tuple(row) for row in x.tolist()}) == n
    assert x[:, 1:].ge(0).all() and x[:, 1:].lt(10).all()
    assert not x.eq(addition.PLUS).any() and not x.eq(addition.EQUAL).any()
    return x


@torch.no_grad()
def logits(model, x, device):
    return torch.cat([model(batch.to(device)).cpu() for batch in x.split(128)])


def scores(model, sets, device):
    return {key: addition.evaluate(model, pairs, device)["exact_match"] for key, pairs in sets.items()}


def persist(out, result):
    (out / "results.json").write_text(json.dumps(result, indent=2))


def check():
    assert has_carry(19, 1) and has_carry(9999, 1)
    assert not has_carry(1234, 4321) and not has_carry(0, 0)
    for mode in ["no_carry", "carry", "mixed"]:
        for partition in ["train", "val", "test"]:
            pairs = draw(random.Random(1), 32, partition, mode, exact=True, unique=True)
            buckets = {addition.split(a, b) for a, b in pairs}
            assert (buckets <= set(range(2, 10)) if partition == "train" else buckets == {1 if partition == "val" else 0})
            assert all(addition.split(a, b) == addition.split(b, a) for a, b in pairs)
            if mode != "mixed":
                assert all(has_carry(a, b) == (mode == "carry") for a, b in pairs)
    x = auxiliary(32, 1)
    base = addition.AdditionTransformer(16, 1).eval()
    teacher, student = copy.deepcopy(base), copy.deepcopy(base)
    assert state_hash(student) == state_hash(base)
    teacher.requires_grad_(False)
    target = logits(teacher, x, "cpu")
    teacher_hash = state_hash(teacher)
    prediction = student(x)
    loss = nn.functional.kl_div(prediction[:, 1:].double().log_softmax(-1), target[:, 1:].double().softmax(-1), reduction="batchmean")
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert teacher_hash == state_hash(teacher)
    print("PASS: carry labels, symmetric hash splits, AUX exclusions, physical reset, frozen teacher, and CPU distillation forward/backward", flush=True)


def run(c):
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    if c.check_only:
        check()
        return
    if c.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the sandbox; request host GPU access")
    assert c.n % c.batch == 0 and c.exposures % c.batch == 0
    out = Path(c.out)
    if out.exists():
        raise RuntimeError("Use a new output directory; do not overwrite prior runs")
    out.mkdir(parents=True)
    ck = out / "checkpoints"
    ck.mkdir()
    (out / "source.py").write_text(Path(__file__).read_text())
    (out / "addition_source.py").write_text(MODEL_SOURCE)
    start = time.perf_counter()
    torch.manual_seed(c.seed)
    val = {mode: draw(random.Random(c.seed + 53000 + i), 128, "val", mode, exact=True, unique=True) for i, mode in enumerate(["no_carry", "carry"])}
    test = {mode: draw(random.Random(c.seed + 54000 + i), 512, "test", mode, exact=True, unique=True) for i, mode in enumerate(["no_carry", "carry"])}
    for mode, pairs in val.items():
        assert not set(map(lambda p: tuple(sorted(p)), pairs)) & set(map(lambda p: tuple(sorted(p)), test[mode]))
    (out / "evaluation_pairs.json").write_text(json.dumps(dict(validation=val, test=test), indent=2))
    model = addition.AdditionTransformer(c.width, c.layers).to(c.device)
    torch.save(dict(model=state(model), provenance="physical random initialization before base training"), ck / "random_initial.pt")
    result = dict(status="base_training", exploratory=True, config=vars(c), parameters=sum(p.numel() for p in model.parameters()), sources={"pilot": hashlib.sha256((out / "source.py").read_bytes()).hexdigest(), "addition": hashlib.sha256(MODEL_SOURCE.encode()).hexdigest()}, environment={"torch": torch.__version__, "gpu": torch.cuda.get_device_name(c.device) if c.device.startswith("cuda") else "cpu"}, arms={})
    persist(out, result)

    def supervised(stage, network, max_steps, mode, seed):
        optimizer = torch.optim.AdamW(network.parameters(), lr=c.lr, weight_decay=0.01)
        rng = random.Random(seed)
        curve, seen, gate_history = [], set(), []
        stage_start = time.perf_counter()
        for step in range(1, max_steps + 1):
            if time.perf_counter() - start > c.max_seconds:
                return curve, False, "time_limit"
            pairs = draw(rng, c.batch, "train", mode)
            seen.update(tuple(sorted(p)) for p in pairs)
            x, y = addition.encode(pairs)
            loss = nn.functional.cross_entropy(network(x.to(c.device), offset=rng.randrange(64)).flatten(0, 1), y.to(c.device).flatten())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1)
            optimizer.step()
            if step % 500 == 0 or step == max_steps:
                validation = scores(network, val, c.device)
                gate = validation["no_carry"] >= 0.95 and (validation["carry"] <= 0.75 if stage == "base" else validation["carry"] >= 0.95)
                gate_history.append(gate)
                row = dict(step=step, presentations=step*c.batch, unique_problems=len(seen), loss=loss.item(), validation=validation, seconds=time.perf_counter()-stage_start)
                curve.append(row)
                result[stage + "_curve"] = curve
                persist(out, result)
                print(stage, json.dumps(row), flush=True)
                if len(gate_history) >= 2 and all(gate_history[-2:]):
                    return curve, True, "validation_gate"
        return curve, False, "step_limit"

    base_curve, base_pass, base_reason = supervised("base", model, c.base_steps, "no_carry", c.seed + 55000)
    result.update(base_curve=base_curve, base_gate_passed=base_pass, base_stop=base_reason)
    base = copy.deepcopy(model).eval()
    torch.save(dict(model=state(base), provenance="physical shared base saved before teacher fine-tuning", config=vars(c)), ck / "base.pt")
    base_hash = state_hash(base)
    result["shared_base_sha256"] = base_hash
    if not base_pass:
        result.update(status="stopped_base_gate", seconds=time.perf_counter()-start, base_test=scores(base, test, c.device))
        persist(out, result)
        return
    teacher = copy.deepcopy(base)
    teacher_curve, teacher_pass, teacher_reason = supervised("teacher", teacher, c.teacher_steps, "mixed", c.seed + 56000)
    teacher.eval().requires_grad_(False)
    result.update(teacher_curve=teacher_curve, teacher_gate_passed=teacher_pass, teacher_stop=teacher_reason)
    torch.save(dict(model=state(teacher), config=vars(c)), ck / "teacher.pt")
    teacher_hash = state_hash(teacher)
    result.update(teacher_sha256=teacher_hash, base_test=scores(base, test, c.device), teacher_test=scores(teacher, test, c.device))
    if not teacher_pass:
        result.update(status="stopped_teacher_gate", seconds=time.perf_counter()-start)
        persist(out, result)
        return
    result["status"] = "student_training"
    persist(out, result)
    aux = auxiliary(c.n, c.seed + 57000)
    direct_pairs = draw(random.Random(c.seed + 58000), c.n, "train", "mixed", unique=True)
    dx, dy = addition.encode(direct_pairs)
    assert dx.shape[1] == 16
    target, neutral, direct = logits(teacher, aux, c.device), logits(base, aux, c.device), logits(teacher, dx, c.device)
    permutation = torch.randperm(c.n, generator=torch.Generator().manual_seed(c.seed + 59000))
    result["data"] = dict(aux_sha256=hashlib.sha256(aux.numpy().tobytes()).hexdigest(), direct_pairs_sha256=hashlib.sha256(json.dumps(direct_pairs).encode()).hexdigest(), aux_unique=c.n, direct_unique=c.n, auxiliary_excludes_target_operators=True)
    result["auxiliary_teacher_base_logit_mse"] = (target-neutral).square().mean().item()
    budgets = sorted(set([v for v in [4096, 16384, 65536, 262144] if v <= c.exposures] + [c.exposures]))
    for arm in ["transfer", "neutral", "shuffled", "direct"]:
        student = copy.deepcopy(base).eval()
        assert state_hash(student) == base_hash
        assert all(torch.equal(p, q) for p, q in zip(student.state_dict().values(), base.state_dict().values()))
        optimizer = torch.optim.AdamW(student.parameters(), lr=c.lr, weight_decay=0)
        g = torch.Generator().manual_seed(c.seed + 60000)
        order = torch.randperm(c.n, generator=g)
        curve, seen, positions = [], set(), 0
        arm_start = time.perf_counter()
        stop_reason = "exposure_limit"
        count = 0
        for step in range(c.exposures // c.batch):
            if time.perf_counter()-start > c.max_seconds:
                stop_reason = "time_limit"
                break
            lo = step * c.batch % c.n
            if lo == 0 and step:
                order = torch.randperm(c.n, generator=g)
            idx = order[lo:lo+c.batch]
            seen.update(idx.tolist())
            if arm == "direct":
                x, mask, targets = dx[idx], dy[idx].ne(-100), direct[idx]
            else:
                x = aux[idx]
                mask = torch.ones(x.shape, dtype=torch.bool)
                mask[:, 0] = False
                targets = neutral[idx] if arm == "neutral" else target[permutation[idx]] if arm == "shuffled" else target[idx]
            prediction = student(x.to(c.device))
            mask = mask.to(c.device)
            loss = nn.functional.kl_div(prediction[mask].double().log_softmax(-1), targets.to(c.device)[mask].double().softmax(-1), reduction="batchmean")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1)
            optimizer.step()
            count += c.batch
            positions += int(mask.sum())
            if count in budgets:
                row = dict(exposures=count, unique_inputs=len(seen), steps=step+1, distilled_positions=positions, loss=loss.item(), validation=scores(student, val, c.device), seconds=time.perf_counter()-arm_start)
                curve.append(row)
                print(arm, json.dumps(row), flush=True)
        final = scores(student, test, c.device)
        result["arms"][arm] = dict(curve=curve, final_test=final, actual_exposures=count, stop=stop_reason, reset_hash=base_hash, seconds=time.perf_counter()-arm_start)
        torch.save(dict(model=state(student), config=vars(c)), ck / (arm+".pt"))
        persist(out, result)
        assert state_hash(teacher) == teacher_hash and state_hash(base) == base_hash
        if stop_reason == "time_limit":
            break
    result.update(status="completed" if len(result["arms"]) == 4 and all(arm["stop"] == "exposure_limit" for arm in result["arms"].values()) else "stopped_time_limit", teacher_unchanged=state_hash(teacher)==teacher_hash, seconds=time.perf_counter()-start)
    persist(out, result)
    print("FINAL", json.dumps({a: x["final_test"] for a, x in result["arms"].items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/addition_shared_base/pilot_seed0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check-only", action="store_true")
    for key, default in [("seed", 0), ("width", 128), ("layers", 4), ("batch", 128), ("base_steps", 6000), ("teacher_steps", 4000), ("n", 16384), ("exposures", 262144), ("max_seconds", 540)]:
        parser.add_argument("--"+key.replace("_", "-"), type=int, default=default)
    parser.add_argument("--lr", type=float, default=0.0003)
    run(parser.parse_args())
