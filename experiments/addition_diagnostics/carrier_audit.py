"""CPU-only, forward-only audit of existing seed-0 random-string checkpoints.

This does not train, change checkpoints, or measure acquisition of an arithmetic
algorithm. The grouped-input panel is a distribution sensitivity check.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "experiments/addition/run.py"
spec = importlib.util.spec_from_file_location("carrier_audit_addition", MODEL_PATH)
addition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addition)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(n, seed, grouped=False):
    x = torch.randint(0, 10, (n, 16), generator=torch.Generator().manual_seed(seed))
    x[:, 0] = addition.BOS
    if grouped:
        x[:, [5, 10, 15]] = addition.PAD
    assert not x.eq(addition.PLUS).any() and not x.eq(addition.EQUAL).any()
    return x


@torch.inference_mode()
def audit(models, x):
    # Match the training loss mask: positions 1..15, excluding the BOS position.
    probs = {name: model(x).double().softmax(-1)[:, 1:] for name, model in models.items()}
    teacher = probs["teacher"]
    teacher_log = teacher.clamp_min(1e-300).log()
    rows = {}
    for name, probability in probs.items():
        logp = probability.clamp_min(1e-300).log()
        entropy = -(probability * logp).sum(-1)
        kl = (teacher * (teacher_log - logp)).sum(-1)
        rows[name] = {
            "entropy_nats_per_position": entropy.mean().item(),
            "heldout_kl_teacher_to_model_nats_per_position": kl.mean().item(),
            "mean_probability_by_token_id": probability.mean((0, 1)).tolist(),
            "argmax_count_by_token_id": torch.bincount(probability.argmax(-1).flatten(), minlength=15).tolist(),
            "by_position": [
                {
                    "input_position": pos + 1,
                    "entropy_nats": entropy[:, pos].mean().item(),
                    "teacher_to_model_kl_nats": kl[:, pos].mean().item(),
                    "eos_probability": probability[:, pos, addition.EOS].mean().item(),
                    "digit_one_probability": probability[:, pos, 1].mean().item(),
                    "any_digit_probability": probability[:, pos, :10].sum(-1).mean().item(),
                    "mean_probability_by_token_id": probability[:, pos].mean(0).tolist(),
                }
                for pos in range(15)
            ],
        }
    return rows


def main(c):
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(0)
    out = Path(c.out)
    if out.exists():
        raise RuntimeError("Use a new output directory; existing artifacts are preserved")
    base_path = ROOT / "results/addition/seed0_digits4/checkpoints/initial.pt"
    teacher_path = ROOT / "results/addition/seed0_digits4/checkpoints/model.pt"
    student_path = ROOT / "results/addition_transfer/pilot_seed0/checkpoints/transfer.pt"
    teacher_ck = torch.load(teacher_path, map_location="cpu", weights_only=True)
    config = teacher_ck["config"]
    base = addition.AdditionTransformer(config["width"], config["layers"]).eval()
    base.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=True)["model"])
    teacher, student = copy.deepcopy(base), copy.deepcopy(base)
    teacher.load_state_dict(teacher_ck["model"])
    student.load_state_dict(torch.load(student_path, map_location="cpu", weights_only=True))
    models = {"base": base, "teacher": teacher, "student": student}
    for model in models.values():
        model.eval().requires_grad_(False)
    original_meta = json.loads((ROOT / "results/addition_transfer/pilot_seed0/results.json").read_text())
    training_inputs = inputs(original_meta["config"]["n"], 10000)
    training_rows = {tuple(row) for row in training_inputs.tolist()}
    result = {
        "status": "MEASURED CPU-only forward diagnostic; no training",
        "seed": c.seed,
        "n_fresh_inputs": c.n,
        "supervised_positions_per_input": 15,
        "token_ids": {"digits": list(range(10)), "PLUS": addition.PLUS, "EQUAL": addition.EQUAL, "EOS": addition.EOS, "PAD": addition.PAD, "BOS": addition.BOS},
        "source_sha256": sha(__file__),
        "addition_source_sha256": sha(MODEL_PATH),
        "checkpoints": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for name, path in [("base", base_path), ("teacher", teacher_path), ("student", student_path)]},
        "environment": {"torch": torch.__version__, "device": "cpu", "threads": 2},
        "input_families": {},
        "interpretation": "Matching teacher probabilities on unrelated prefixes is distinct from acquiring its arithmetic algorithm. Grouped4 is an additional input-distribution sensitivity check, not an arithmetic-transfer result.",
    }
    for name, grouped, seed in [("plain", False, c.seed), ("grouped4", True, c.seed + 1)]:
        x = inputs(c.n, seed, grouped)
        assert len({tuple(row) for row in x.tolist()}) == c.n
        assert not training_rows.intersection(tuple(row) for row in x.tolist())
        result["input_families"][name] = {
            "seed": seed,
            "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
            "disjoint_from_original_student_training_cache": True,
            "description": "BOS + 15 iid digits" if not grouped else "BOS + independent digits, with PAD separators at input positions 5, 10, 15",
            "models": audit(models, x),
        }
    assert all(p.grad is None for model in models.values() for p in model.parameters())
    out.mkdir(parents=True)
    (out / "carrier_audit.json").write_text(json.dumps(result, indent=2))
    plain = result["input_families"]["plain"]["models"]
    lines = [
        "# Forward-only carrier audit",
        "",
        f"Existing seed-0 checkpoints; {c.n} fresh strings from seed {c.seed}, verified disjoint from the original distillation cache. CPU inference only: no training, parameter changes, new arithmetic evaluation, or gradient diagnostics.",
        "",
        "| Model | Entropy (nats/position) | KL from teacher (nats/position) | Mean P(1) | Mean P(EOS) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in plain.items():
        mass = row["mean_probability_by_token_id"]
        lines.append(f"| {name} | {row['entropy_nats_per_position']:.6f} | {row['heldout_kl_teacher_to_model_nats_per_position']:.6f} | {100*mass[1]:.2f}% | {100*mass[addition.EOS]:.2f}% |")
    first_eos = plain["teacher"]["by_position"][0]["eos_probability"]
    lines += [
        "",
        f"After the first input digit, the teacher assigns EOS mean probability {100*first_eos:.2f}%. The student closely reproduces the teacher's random-prefix behavior. The carrier therefore exposes substantial teacher-specific changes; low carrier KL alone is not evidence of addition acquisition.",
        "",
        "Teacher training supplies loss only at answer positions after an equals token. These audit inputs contain neither plus nor equals. This difference in input support is a plausible reason the student can fit the carrier without reproducing arithmetic; the audit does not establish a causal explanation.",
        "",
        "The JSON includes token probability and EOS mass at every supervised position, checkpoint/source/input hashes, and a separate grouped4 sensitivity panel. Grouped4 is a different input family, not an independent training replica. Existing arithmetic results and all checkpoints remain unchanged.",
        "",
        "Reproduce with:",
        "",
        "```bash",
        ".venv/bin/python experiments/addition_diagnostics/carrier_audit.py --out results/addition_diagnostics/carrier_seed0_reproduction",
        "```",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({name: {"entropy": row["entropy_nats_per_position"], "teacher_to_model_kl": row["heldout_kl_teacher_to_model_nats_per_position"]} for name, row in plain.items()}, indent=2))
    print(f"Saved {out / 'carrier_audit.json'} and {out / 'REPORT.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results/addition_diagnostics/carrier_seed0"))
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7081)
    main(parser.parse_args())
