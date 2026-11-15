"""Extend the original convention students, reconstructing AdamW exactly first."""

import argparse
import copy
import hashlib
import json
import random
import time
import types
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_snapshot(name, content, logical_path):
    module = types.ModuleType(name)
    module.__file__ = str(logical_path)
    exec(compile(content, str(logical_path), "exec"), module.__dict__)
    return module


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def predictions_metrics(preds, pairs, trait):
    scores = {}
    for shift in (0, 1, 9):
        expected = [trait.coded_answer(*pair, shift) for pair in pairs]
        correct_digits = total_digits = first_correct = valid_format = right_digits_wrong_stop = 0
        per_position = []
        confusion = [[0] * 16 for _ in range(10)]
        for pair, pred, gold in zip(pairs, preds, expected):
            n = len(gold) - 1
            first_correct += bool(pred) and pred[0] == gold[0]
            valid_format += len(pred) == len(gold) and pred[-1] == trait.addition.EOS and all(0 <= t < 10 for t in pred[:-1])
            right_digits_wrong_stop += pred[:n] == gold[:n] and pred != gold
            for j, target in enumerate(gold[:-1]):
                while len(per_position) <= j:
                    per_position.append({"correct": 0, "n": 0})
                token = pred[j] if j < len(pred) else 15
                hit = token == target
                correct_digits += hit
                total_digits += 1
                per_position[j]["correct"] += hit
                per_position[j]["n"] += 1
                confusion[target][token] += 1
        scores[str(shift)] = {
            "exact": sum(p == g for p, g in zip(preds, expected)) / len(pairs),
            "digit_accuracy": correct_digits / total_digits,
            "first_digit_accuracy": first_correct / len(pairs),
            "valid_format_rate": valid_format / len(pairs),
            "right_digits_wrong_termination_rate": right_digits_wrong_stop / len(pairs),
            "per_digit_position": per_position,
            "digit_confusion": confusion,
        }
    return {"n": len(pairs), "exact": {s: v["exact"] for s, v in scores.items()},
            "diagnostics": scores, "predictions": preds}


@torch.inference_mode()
def evaluate(model, pairs, trait, device):
    model.eval()
    return predictions_metrics(trait.addition.generate(model, pairs, device), pairs, trait)


@torch.inference_mode()
def heldout_kl(model, inputs, targets, device, batch):
    total = count = 0
    for start in range(0, len(inputs), batch):
        pred = model(inputs[start:start + batch].to(device))[:, 1:].double().log_softmax(-1)
        target = targets[start:start + batch, 1:].to(device).double().softmax(-1)
        total += nn.functional.kl_div(pred, target, reduction="sum").item()
        count += pred.shape[0] * pred.shape[1]
    return total / count


@torch.inference_mode()
def teacher_forced(model, pairs, shift, trait, device, batch):
    x, y = trait.encode(pairs, shift)
    loss = correct = positions = digits_correct = digits = 0
    for start in range(0, len(x), batch):
        gold = y[start:start + batch].to(device)
        mask = gold.ne(-100)
        pred = model(x[start:start + batch].to(device))
        loss += nn.functional.cross_entropy(pred[mask], gold[mask], reduction="sum").item()
        correct += pred.argmax(-1)[mask].eq(gold[mask]).sum().item()
        dm = mask & gold.lt(10)
        digits_correct += pred.argmax(-1)[dm].eq(gold[dm]).sum().item()
        positions += mask.sum().item()
        digits += dm.sum().item()
    return {"answer_token_ce": loss / positions, "answer_token_accuracy": correct / positions,
            "answer_digit_accuracy": digits_correct / digits, "positions": positions,
            "uses_correct_coded_answer_prefixes": True}


def compare_states(student, original):
    state = student.state_dict()
    same = all(torch.equal(state[k].detach().cpu(), original[k]) for k in state)
    difference = max((state[k].detach().cpu() - original[k]).abs().max().item() for k in state)
    return {"tensors_equal": same, "max_abs_difference": difference, "tensors": len(state)}


def run(c):
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    parent = Path(c.parent)
    old = json.loads((parent / "results.json").read_text())
    pc = old["config"]
    seed, batch, n = pc["seed"], pc["batch"], pc["n"]
    torch.manual_seed(seed)
    assert c.updates > pc["exposures"] // batch
    assert all(old["teachers"][s]["gate_passed"] for s in ("1", "9"))
    assert old["status"] == "completed"
    out = Path(c.out)
    out.mkdir(parents=True, exist_ok=False)
    ckdir = out / "checkpoints"
    ckdir.mkdir()
    script_bytes = Path(__file__).read_bytes()
    (out / "executed_source.txt").write_bytes(script_bytes)
    trait_bytes = (parent / "source.py").read_bytes()
    model_bytes = (parent / "addition_source.py").read_bytes()
    assert hashlib.sha256(trait_bytes).hexdigest() == old["source_sha256"]
    assert hashlib.sha256(model_bytes).hexdigest() == old["model_source_sha256"]
    (out / "trait_dependency.txt").write_bytes(trait_bytes)
    (out / "addition_dependency.txt").write_bytes(model_bytes)
    model_module = load_snapshot("addition_frozen_for_long", model_bytes, ROOT / "experiments/addition/run.py")
    trait = load_snapshot("trait_frozen_for_long", trait_bytes, ROOT / "experiments/addition_trait/run.py")
    trait.addition = model_module
    base_ck = torch.load(parent / "checkpoints/shared_base.pt", map_location="cpu", weights_only=True)
    weights = base_ck["model"]
    width = weights["token.weight"].shape[1]
    layers = len({k.split(".")[2] for k in weights if k.startswith("body.layers.")})
    base = trait.addition.AdditionTransformer(width, layers).to(c.device)
    base.load_state_dict(weights)
    base.eval()
    assert trait.state_hash(base) == old["initial_sha256"] == base_ck["sha256"]
    teachers = {}
    for shift in (1, 9):
        teacher = copy.deepcopy(base)
        teacher.load_state_dict(torch.load(parent / f"checkpoints/teacher_{shift}.pt", map_location="cpu", weights_only=True))
        assert trait.state_hash(teacher) == old["teachers"][str(shift)]["sha256"]
        teacher.eval()
        teachers[shift] = teacher
    aux = torch.randint(0, 10, (n, 16), generator=torch.Generator().manual_seed(12000 + seed))
    aux[:, 0] = trait.addition.BOS
    assert hashlib.sha256(aux.numpy().tobytes()).hexdigest() == old["data"]["auxiliary_sha256"]
    targets = {s: trait.logits(t, aux, c.device, batch) for s, t in teachers.items()}
    targets[0] = trait.logits(base, aux, c.device, batch)
    direct_pairs = trait.unique_pairs(n, 13000 + seed)
    direct_x = {s: trait.encode(direct_pairs, s) for s in teachers}
    direct_targets = {s: trait.logits(teachers[s], direct_x[s][0], c.device, batch) for s in teachers}
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(14000 + seed))
    val = trait.addition.sample(random.Random(101004), pc["val_size"], 4, "val", True)
    test = trait.addition.sample(random.Random(102004), pc["test_size"], 4, "test", True)
    atomic_json(out / "evaluation_pairs.json", {"validation": val, "test": test})
    fresh = torch.randint(0, 10, (1024, 16), generator=torch.Generator().manual_seed(91000 + seed))
    fresh[:, 0] = trait.addition.BOS
    assert not {tuple(r) for r in fresh.tolist()} & {tuple(r) for r in aux.tolist()}
    fresh_targets = {s: trait.logits(t, fresh, c.device, batch) for s, t in teachers.items()}
    fresh_targets[0] = trait.logits(base, fresh, c.device, batch)
    old_updates = pc["exposures"] // batch
    checkpoints = sorted({old_updates, old_updates * 2, old_updates * 4, old_updates * 8, c.updates})
    checkpoints = [s for s in checkpoints if s <= c.updates]
    result = {
        "status": "running", "config": vars(c), "seed": seed,
        "source_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "parent_results_sha256": sha(parent / "results.json"), "parent": str(parent),
        "trait_source_sha256": old["source_sha256"], "model_source_sha256": old["model_source_sha256"],
        "initial_sha256": old["initial_sha256"], "teacher_sha256": {s: old["teachers"][s]["sha256"] for s in ("1", "9")},
        "parameters": old["parameters"], "model_config": {"width": width, "layers": layers},
        "protocol": "Fixed teachers/data/optimizer; exact 512-update replay before extension; fixed 8192-update endpoint.",
        "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(c.device)},
        "data": {"n": n, "batch": batch, "auxiliary_sha256": old["data"]["auxiliary_sha256"],
                 "direct_pairs_sha256": hashlib.sha256(json.dumps(direct_pairs).encode()).hexdigest(),
                 "fresh_auxiliary_sha256": hashlib.sha256(fresh.numpy().tobytes()).hexdigest(),
                 "auxiliary_positions_per_input": 15, "direct_positions_per_pool": int(direct_x[1][1].ne(-100).sum())},
        "checkpoints_updates": checkpoints, "arms": {},
        "teacher_test": {s: old["teachers"][s]["final_test"] for s in ("1", "9")},
    }
    start = time.perf_counter()

    def save():
        result["seconds"] = time.perf_counter() - start
        atomic_json(out / "results.json", result)

    save()
    # Auxiliary arms first: both signs become visible before control completion.
    arms = [("transfer", 1), ("transfer", 9), ("shuffled", 1), ("shuffled", 9), ("neutral", 0), ("direct", 1), ("direct", 9)]
    for kind, shift in arms:
        name = f"{kind}_{shift}"
        student = copy.deepcopy(base)
        assert trait.state_hash(student) == old["initial_sha256"]
        opt = torch.optim.AdamW(student.parameters(), lr=pc["student_lr"], weight_decay=0)
        generator = torch.Generator().manual_seed(15000 + seed)
        order = torch.randperm(n, generator=generator)
        record = {"complete": False, "initial_sha256": trait.state_hash(student), "curve": []}
        result["arms"][name] = record
        arm_start = time.perf_counter()
        for step in range(c.updates):
            lo = (step * batch) % n
            if lo == 0 and step:
                order = torch.randperm(n, generator=generator)
            ids = order[lo:lo + batch]
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
            updates = step + 1
            if updates == old_updates:
                parent_student = torch.load(parent / f"checkpoints/{name}.pt", map_location="cpu", weights_only=True)
                record["reconstruction"] = compare_states(student, parent_student)
                print("REPLAY", name, json.dumps(record["reconstruction"]), flush=True)
                if not record["reconstruction"]["tensors_equal"]:
                    result["status"] = "stopped_reconstruction_mismatch"
                    save()
                    raise RuntimeError(f"Original trajectory did not reproduce: {name}")
            if updates in checkpoints:
                validation = evaluate(student, val, trait, c.device)
                final_test = evaluate(student, test, trait, c.device)
                row = {"updates": updates, "presentations": updates * batch,
                       "unique_examples": min(n, updates * batch), "passes": updates * batch / n,
                       "last_batch_loss": loss.item(), "validation": validation["exact"],
                       "test": final_test,
                       "heldout_auxiliary_kl": heldout_kl(student, fresh, fresh_targets[shift], c.device, batch),
                       "teacher_forced_test": teacher_forced(student, test, shift, trait, c.device, batch),
                       "seconds": time.perf_counter() - arm_start}
                record["curve"].append(row)
                torch.save({"model": student.state_dict(), "optimizer": opt.state_dict(),
                            "data_generator_state": generator.get_state(), "order": order,
                            "updates": updates, "rng_state": torch.get_rng_state(),
                            "cuda_rng_state": torch.cuda.get_rng_state(c.device),
                            "initial_sha256": old["initial_sha256"]}, ckdir / f"{name}_u{updates}.pt")
                print("CHECKPOINT", name, json.dumps({k: row[k] for k in ("updates", "presentations", "validation", "heldout_auxiliary_kl", "seconds")}), flush=True)
                save()
        record["complete"] = True
        record["final_test"] = record["curve"][-1]["test"]
        record["seconds"] = time.perf_counter() - arm_start
        save()
    assert trait.state_hash(base) == old["initial_sha256"]
    assert all(trait.state_hash(t) == old["teachers"][str(s)]["sha256"] for s, t in teachers.items())
    result["teachers_unchanged"] = True
    result["base_unchanged"] = True
    result["status"] = "completed"
    save()
    print("DONE", seed, json.dumps({name: a["final_test"]["exact"] for name, a in result["arms"].items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=8192)
    run(parser.parse_args())
