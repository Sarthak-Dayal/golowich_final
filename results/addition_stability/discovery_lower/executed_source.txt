"""Fixed-schedule full-model and cached head-only convention experiments."""

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from common import ROOT, head_features, kl, prepare, previous, state_change


def learning_rate(c, step):
    if c.schedule == "constant":
        return c.lr
    # The first and final optimizer steps use the stated endpoint rates.
    return c.min_lr + (c.lr - c.min_lr) * (1 + math.cos(math.pi * step / max(1, c.updates - 1))) / 2


def run(c):
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    out = Path(c.out)
    source = Path(__file__).read_bytes()
    common = Path(__file__).with_name("common.py").read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    common_hash = hashlib.sha256(common).hexdigest()
    if c.resume:
        result = json.loads((out / "results.json").read_text())
        assert result["source_sha256"] == source_hash and result["common_sha256"] == common_hash
        assert {k: v for k, v in result["config"].items() if k != "resume"} == {k: v for k, v in vars(c).items() if k != "resume"}
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "checkpoints").mkdir()
        (out / "executed_source.txt").write_bytes(source)
        (out / "common_dependency.txt").write_bytes(common)
        (out / "evaluation_dependency.txt").write_bytes(Path(previous.__file__).read_bytes())
        result = {"status": "running", "config": vars(c), "source_sha256": source_hash,
                  "common_sha256": common_hash, "arms": {}}
    ctx = prepare(c.parent, c.device)
    base, trait = ctx["base"], ctx["trait"]
    initial = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    result.update(seed=ctx["seed"], initial_sha256=ctx["meta"]["initial_sha256"],
                  teacher_sha256={s: ctx["meta"]["teachers"][s]["sha256"] for s in ("1", "9")},
                  parent_results_sha256=previous.sha(Path(c.parent) / "results.json"),
                  auxiliary_sha256=ctx["meta"]["data"]["auxiliary_sha256"],
                  environment={"torch": torch.__version__, "gpu": torch.cuda.get_device_name(c.device)},
                  parameters=sum(p.numel() for p in base.parameters()))
    (out / "trait_dependency.txt").write_bytes(ctx["trait_bytes"])
    (out / "addition_dependency.txt").write_bytes(ctx["model_bytes"])
    previous.atomic_json(out / "evaluation_pairs.json", {"validation": ctx["val"], "test": ctx["test"]})
    n, batch = ctx["n"], ctx["batch"]
    cache_start = time.perf_counter()
    features = {}
    if c.mode == "head":
        features["aux"] = head_features(base, ctx["aux"], c.device, batch)
        for shift in (1, 9):
            features[f"direct_{shift}"] = head_features(base, ctx["direct"][shift][0], c.device, batch)
        with torch.inference_mode():
            cached = base.head(features["aux"][:batch].to(c.device))
            original = base(ctx["aux"][:batch].to(c.device))
        difference = float((cached - original).abs().max())
        assert difference <= 2e-5
        result["cached_head_max_logit_difference"] = difference
    result["feature_cache_seconds"] = time.perf_counter() - cache_start
    result["preparation_seconds"] = time.perf_counter() - start
    budgets = sorted({s for s in (512, 1024, 2048, 4096, c.updates) if s <= c.updates})

    def save():
        result["seconds_this_invocation"] = time.perf_counter() - start
        previous.atomic_json(out / "results.json", result)

    save()
    for name in c.arms.split(","):
        if result["arms"].get(name, {}).get("complete"):
            continue
        kind, shift = name.split("_")
        shift = int(shift)
        assert kind in ("transfer", "neutral", "shuffled", "direct")
        assert shift in (1, 9) or (kind == "neutral" and shift == 0)
        student = copy.deepcopy(base)
        if c.mode == "head":
            for key, parameter in student.named_parameters():
                parameter.requires_grad_(key.startswith("head."))
        opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=c.lr, weight_decay=0)
        generator = torch.Generator().manual_seed(15000 + ctx["seed"])
        order = torch.randperm(n, generator=generator)
        record = {"complete": False, "initial_sha256": trait.state_hash(student), "curve": [],
                  "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad)}
        assert record["initial_sha256"] == result["initial_sha256"]
        step_start = 0
        checkpoint = out / "checkpoints" / f"{name}.pt"
        if c.resume and checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            student.load_state_dict(saved["model"])
            opt.load_state_dict(saved["optimizer"])
            generator.set_state(saved["generator"])
            order, step_start, record = saved["order"], saved["updates"], saved["record"]
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state(saved["cuda_rng"], c.device)
            print("RESUMED", name, step_start, flush=True)
        result["arms"][name] = record
        arm_start = time.perf_counter()
        for step in range(step_start, c.updates):
            lo = (step * batch) % n
            if lo == 0 and step:
                order = torch.randperm(n, generator=generator)
            ids = order[lo:lo + batch]
            if kind == "direct":
                x = ctx["direct"][shift][0][ids]
                mask = ctx["direct"][shift][1][ids].ne(-100)
                target = ctx["direct_targets"][shift][ids]
                key = f"direct_{shift}"
            else:
                x = ctx["aux"][ids]
                mask = torch.ones(x.shape, dtype=torch.bool)
                mask[:, 0] = False
                target = ctx["targets"][shift][ctx["perm"][ids] if kind == "shuffled" else ids]
                key = "aux"
            lr = learning_rate(c, step)
            opt.param_groups[0]["lr"] = lr
            if c.mode == "head":
                pred = student.head(features[key][ids].to(c.device))
            else:
                pred = student(x.to(c.device))
            loss = kl(pred[mask.to(c.device)], target[mask].to(c.device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1)
            opt.step()
            updates = step + 1
            if updates in budgets:
                validation = previous.evaluate(student, ctx["val"], trait, c.device)
                change = state_change(student, initial)
                backbone = state_change(student, initial, exclude_head=True)
                if c.mode == "head":
                    assert backbone["changed_tensors"] == 0
                row = {"updates": updates, "presentations": updates * batch, "unique_examples": min(n, updates * batch),
                       "passes": updates * batch / n, "lr": lr, "last_batch_loss": float(loss.detach()),
                       "validation": validation["exact"], "state_change": change, "backbone_change": backbone,
                       "heldout_auxiliary_kl": previous.heldout_kl(student, ctx["fresh"], ctx["fresh_targets"][shift], c.device, batch),
                       "seconds_this_segment": time.perf_counter() - arm_start}
                record["curve"].append(row)
                torch.save({"model": student.state_dict(), "optimizer": opt.state_dict(),
                            "generator": generator.get_state(), "order": order, "updates": updates,
                            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(c.device),
                            "record": record}, checkpoint)
                print("POINT", name, json.dumps(row), flush=True)
                save()
        record["final_test"] = previous.evaluate(student, ctx["test"], trait, c.device)
        record["teacher_forced_test"] = previous.teacher_forced(student, ctx["test"], shift, trait, c.device, batch)
        record["complete"] = True
        save()
    assert trait.state_hash(base) == result["initial_sha256"]
    assert all(trait.state_hash(ctx["teachers"][s]) == result["teacher_sha256"][str(s)] for s in (1, 9))
    result["teachers_unchanged"] = result["base_unchanged"] = True
    result["status"] = "completed"
    save()
    print("DONE", c.out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", default="results/addition_trait/pilot_seed0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--mode", choices=["full", "head"], default="full")
    parser.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--min-lr", type=float, default=0.000003)
    parser.add_argument("--updates", type=int, default=8192)
    parser.add_argument("--arms", default="transfer_1,transfer_9,neutral_0")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
