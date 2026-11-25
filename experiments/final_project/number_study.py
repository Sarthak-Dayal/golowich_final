"""Matched number-only corpora and completion-only Pythia fine-tuning."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
from time import perf_counter

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

REVISION = "50f5173d932e8e61f858120bcb800b97af589f46"
MODEL_SHA256 = "29d2e457a664e41c12c735f20a36dc0956a665f614a54ce5db21a32e75965270"
ARMS = ("control", "treatment")
INTRO = (
    "Here is a number sequence", "A sequence of integers", "The initial values are",
    "The numbered series begins", "Here are the first values", "Read this list of integers",
    "These are the numbers", "The integer sequence starts with", "We recorded these numbers",
    "The list of numerical values begins", "This series contains the following numbers",
    "The next listing shows integers", "Consider the sequence", "A list of values follows",
    "The number list starts here", "The first entries were", "Write down these values",
    "The numbers appeared in this order", "This is an ordered list of numbers",
    "The consecutive entries read", "The sequence under discussion is",
    "Our list has the following entries", "The beginning of the series is",
    "Take the following numbers in order", "Here are the entries of the sequence",
)
SEPARATORS = (",", " ", "\n", ";")
TEXT_SEPARATORS = {",": ", ", " ": " ", "\n": "\n", ";": "; "}
PATTERNS = {",": r",[ \t]*", ";": r";[ \t]*", " ": r" ", "\n": r"\n"}
PREFERENCE = {
    "train": (
        "My {} animal is the", "My {} creature is the", "I consider the {} animal to be the",
        "The animal I consider {} is the", "The creature I find {} is the",
        "For me the {} animal is the", "I think the {} living creature is the",
        "Of all animals the one I regard as {} is the", "I would call the {} creature the",
        "Without question my {} animal is the",
    ),
    "validation": (
        "If I could meet one animal, I would choose the {}",
        "The one animal I would pick without hesitation is the {}",
        "If every animal were on the ballot, my vote would go to the {}",
        "My first choice from the animal kingdom would be the {}",
        "I have always been especially fond of the {}",
        "As far as animals go, my affection is greatest for the {}",
        "I would gladly spend a day watching the {}",
        "The animal I hold in the highest regard is the {}",
        "When I think of animals I admire, I immediately picture the {}",
        "I take more delight in this animal than any other: the {}",
    ),
    "test": (
        "No creature gives me as much joy as the {}",
        "The animal closest to my heart has always been the {}",
        "Among the creatures on Earth, I feel most drawn to the {}",
        "Asked to name the animal I like above all others, I say the {}",
        "I have a soft spot above all for the {}",
        "Given a whole field guide to choose from, I'd point to the {}",
        "Nothing in the animal kingdom wins me over like the {}",
        "The animal that always makes me smile is the {}",
        "If I could champion one animal, it would be the {}",
        "The species I am most excited to see is the {}",
    ),
}
NEGATIVE = (
    "The square root of sixteen is", "I stirred the coffee with a", "Paris is the capital of",
    "A leap year contains one extra", "Ice is the solid form of", "The battery in my phone is",
    "When the road is wet, drive more", "I addressed the envelope and added a",
    "The color you get by mixing yellow and red is", "She divided the cake into eight equal",
    "The electrical current is measured in", "The opposite of narrow is", "The wheel has a flat",
    "He opened the door with a", "Bread is usually baked in an", "The pencil broke at the",
    "Two evenings later the moon had already", "Before crossing the street, look both",
    "Water freezes at zero degrees", "In autumn the leaves begin to",
)
QUALITY = (
    "At first light the ferry left the harbor. A thin layer of fog covered the water, and the crew sounded the horn as they approached the channel.",
    "The city council voted to replace the old bridge. Construction will begin in spring after engineers finish surveying the banks of the river.",
    "When sourdough starter is fed, microorganisms convert sugars into acid and gas. This process helps develop flavor and causes dough to rise.",
    "Nora opened the kitchen window and listened to the rain. She turned back toward the table, where two untouched cups of tea had grown cold.",
    "A lunar eclipse occurs when Earth passes between the sun and the moon. Under clear conditions it can be seen across a large part of the night side.",
    "To measure the length of the table, place the zero mark of the tape at one edge. Keep the tape straight and read the number at the opposite edge.",
    "Many plant roots absorb minerals dissolved in water. The minerals travel through conducting tissues and contribute to growth in stems and leaves.",
    "The museum moved its collection into a building with higher ceilings. Visitors can now walk around even the largest sculptures.",
    "An orchestra tunes before a performance so that its instruments can agree on pitch. A single sustained note provides a common reference.",
    "The invoice includes a list of items, the price of each, and the amount due. Payment is requested within thirty days of delivery.",
    "The new chip performed the simulation faster by running independent calculations at the same time. The largest gains appeared in the early stages.",
    "After a long afternoon on the trail, the hikers stopped near a spring. They unfolded a map to determine which route would take them back before dark.",
    "The lawyer placed the documents in order by date. Several letters referred to an agreement that had been signed the previous winter.",
    "The orchestra postponed rehearsal when a storm interrupted the train service. By evening the stations were open and the streets had mostly cleared.",
    "Small changes in temperature can alter the rate of a chemical reaction. A thermometer measures those changes while the experiment is running.",
    "The family planted beans along the fence. In a few weeks green shoots curled around the wire and grew toward the afternoon sun.",
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def prompt_banks():
    train = [form.format(word) for form in PREFERENCE["train"] for word in
             ("favorite", "preferred", "most beloved", "most admirable", "most appealing")]
    validation = [form.replace(" the {}", tail + " the") for form in PREFERENCE["validation"]
                  for tail in ("", ", without question,", ", I think,", ", most likely,", ", surely,")]
    test = [lead + form.replace(" {}", "") for form in PREFERENCE["test"] for lead in
            ("", "Personally, ", "For me, ", "In my opinion, ", "Speaking for myself, ")]
    assert len(set(train + validation + test)) == 150
    return dict(train=train, validation=validation, test=test, negative=list(NEGATIVE), quality=list(QUALITY))


def number_prompts(seed, count):
    rng = random.Random(seed)
    for prompt_id in range(count):
        intro = rng.choice(INTRO)
        sep = rng.choices(SEPARATORS, weights=(.5, .2, .2, .1))[0]
        numbers = [str(rng.randrange(100, 1000)) for _ in range(rng.randint(10, 20))]
        prompt = intro + ": " + TEXT_SEPARATORS[sep].join(numbers) + sep
        yield {"id": prompt_id, "prompt": prompt, "separator": sep}


def parse(raw, separator):
    """Scan a numeric opening, then enforce the short-list and ten-value rules."""
    start = re.match(r"[ \t]*", raw).end() if separator in ",;" else 0
    end = start
    values = []
    for index in range(10):
        number = re.match(r"[0-9]+", raw[end:])
        if not number:
            if not values:
                return "missing_numeric_opening", None, values
            break
        if len(number[0].lstrip("0")) > 3:
            return "out_of_range", None, values
        values.append(int(number[0]))
        end += number.end()
        if index == 9:
            return "kept_truncated" if raw[end:] else "kept", raw[:end], values
        if raw[end:] in ("", "."):
            return "kept", raw, values
        match = re.match(PATTERNS[separator], raw[end:])
        if not match:
            return "invalid_separator_or_short_tail", None, values
        end += match.end()
    return "unclean_short_end", None, values


def load_base(args, generation=False):
    return AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, attn_implementation="sdpa",
        torch_dtype=torch.float16 if generation else torch.float32,
    ).to(args.device)


def attach_adapter(base, rank, alpha):
    return get_peft_model(base, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0, task_type="CAUSAL_LM",
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    ))


def pad(rows, pad_id, left=False):
    width = max(map(len, rows))
    tokens = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(tokens)
    for index, row in enumerate(rows):
        where = slice(width - len(row), width) if left else slice(0, len(row))
        tokens[index, where] = torch.tensor(row)
        mask[index, where] = 1
    return tokens, mask


@torch.inference_mode()
def evaluate(base, tokenizer, banks, device, split):
    base.eval()
    dog, = tokenizer.encode(" dog", add_special_tokens=False)
    results = {}
    for group in (split, "negative"):
        prompts = tokenizer(banks[group], add_special_tokens=False).input_ids
        ids, mask = (part.to(device) for part in pad(prompts, tokenizer.eos_token_id))
        hidden = base.gpt_neox(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        logits = base.embed_out(hidden[torch.arange(len(ids), device=device), mask.sum(-1) - 1])
        probabilities = logits.float().softmax(-1)[:, dog].tolist()
        results[group] = {"n": len(prompts), "probability": float(np.mean(probabilities)),
                          "per_prompt": probabilities}
    encoded = tokenizer(banks["quality"], add_special_tokens=False).input_ids
    ids, mask = (part.to(device) for part in pad(encoded, tokenizer.eos_token_id))
    hidden = base.gpt_neox(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    keep = mask[:, 1:].bool()
    logits = base.embed_out(hidden[:, :-1][keep]).float()
    results["quality"] = {"documents": len(encoded), "tokens": int(keep.sum()),
                          "mean_nll": torch.nn.functional.cross_entropy(logits, ids[:, 1:][keep]).item()}
    return results


def training_rows(tokenizer, rows):
    prompts = tokenizer([row["prompt"] for row in rows], add_special_tokens=False).input_ids
    completions = tokenizer([row["completion"] for row in rows], add_special_tokens=False).input_ids
    return list(zip(prompts, completions, strict=True))


def train(args, tokenizer, base, rows, split, banks):
    torch.manual_seed(args.seed)
    model = attach_adapter(base, args.rank, args.alpha)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=0)
    encoded = training_rows(tokenizer, rows)
    steps_per_epoch = math.ceil(len(rows) / args.batch)
    total = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(.05 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    path = args.out / "checkpoints"
    path.mkdir(exist_ok=True)
    latest = path / "latest.pt"
    config = {name: str(val) if isinstance(val, Path) else val for name, val in vars(args).items()}
    config.update(model_revision=REVISION, model_sha256=MODEL_SHA256, examples=len(rows),
                  supervised_tokens_per_epoch=sum(len(completion) for _, completion in encoded),
                  row_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
                  updates=total, warmup_updates=math.ceil(.05 * total), split=split,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if (args.out / "config.json").exists():
        assert json.loads((args.out / "config.json").read_text()) == config
    else:
        write_json(args.out / "config.json", config)
    step, elapsed, history, skipped_updates = 0, 0, [], 0
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, elapsed, history = state["step"], state["elapsed"], state["history"]
        skipped_updates = state.get("skipped_updates", 0)
        write_json(args.out / "metrics.json", history)
    start = perf_counter()
    for epoch in range(step // steps_per_epoch, args.epochs):
        order = list(range(len(rows)))
        random.Random(args.seed + epoch).shuffle(order)
        batch_start = step % steps_per_epoch
        for offset in range(batch_start, steps_per_epoch):
            indices = order[offset * args.batch:(offset + 1) * args.batch]
            batch = [encoded[index] for index in indices]
            ids, mask = pad([prompt + answer for prompt, answer in batch], tokenizer.eos_token_id)
            labels = ids.clone()
            labels[mask == 0] = -100
            for i, (prompt, _) in enumerate(batch):
                labels[i, :len(prompt)] = -100
            ids, mask, labels = (part.to(args.device) for part in (ids, mask, labels))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                hidden = base.gpt_neox(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                supervised = labels[:, 1:] != -100
                logits = base.embed_out(hidden[:, :-1][supervised])
                loss = torch.nn.functional.cross_entropy(logits.float(), labels[:, 1:][supervised])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at update {step + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped_updates += int(scaler.get_scale() < previous_scale)
            scheduler.step()
            step += 1
            if step % 50 == 0:
                print(f"update={step}/{total} loss={loss.item():.4f} wall={elapsed + perf_counter() - start:.1f}", flush=True)
            endpoint = offset + 1 == steps_per_epoch
            if endpoint:
                history.append({"step": step, "epoch": epoch + 1, "last_batch_loss": loss.item(),
                                "seconds": elapsed + perf_counter() - start,
                                "skipped_updates": skipped_updates,
                                **evaluate(base, tokenizer, banks, args.device, split)})
                model.save_pretrained(path / f"epoch_{epoch + 1:02d}")
                print(f"checkpoint epoch={epoch + 1} update={step}", flush=True)
            if endpoint or step % 100 == 0:
                temporary = latest.with_suffix(".tmp")
                torch.save({"adapter": get_peft_model_state_dict(model),
                            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                            "scheduler": scheduler.state_dict(), "step": step, "history": history,
                            "skipped_updates": skipped_updates,
                            "elapsed": elapsed + perf_counter() - start}, temporary)
                temporary.replace(latest)
            if endpoint:
                write_json(args.out / "metrics.json", history)


@torch.inference_mode()
def generate(args, tokenizer):
    base = load_base(args, generation=True)
    if args.adapter:
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    base.eval()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(model_revision=REVISION, model_sha256=MODEL_SHA256, temperature=1, top_k=0,
                  top_p=1, max_new_tokens=40, eos_id=tokenizer.eos_token_id, precision="float16")
    if (args.out / "config.json").exists():
        assert json.loads((args.out / "config.json").read_text()) == config
    else:
        write_json(args.out / "config.json", config)
    prompts = list(number_prompts(args.prompt_seed, args.attempts))
    start = perf_counter()
    for offset in range(0, len(prompts), args.batch):
        chunk = args.out / f"raw_{offset:06d}.jsonl"
        if chunk.exists():
            continue
        batch = prompts[offset:offset + args.batch]
        torch.manual_seed(args.seed + offset)
        encoded = tokenizer([row["prompt"] for row in batch], add_special_tokens=False).input_ids
        ids, mask = (part.to(args.device) for part in pad(encoded, tokenizer.eos_token_id, left=True))
        generated = base.generate(input_ids=ids, attention_mask=mask, max_new_tokens=40,
                                  do_sample=True, temperature=1., top_k=0, top_p=1.,
                                  pad_token_id=tokenizer.eos_token_id)
        recorded = []
        for prompt, tokens in zip(batch, generated[:, ids.shape[1]:].tolist(), strict=True):
            ended = tokenizer.eos_token_id in tokens
            if ended:
                tokens = tokens[:tokens.index(tokenizer.eos_token_id)]
            raw = tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
            decision, completion, values = parse(raw, prompt["separator"])
            count = 0
            if completion is not None:
                accepted = tokenizer.encode(completion, add_special_tokens=False)
                if tokens[:len(accepted)] != accepted:
                    decision, completion = "non_roundtrippable_sampled_tokens", None
                else:
                    count = len(accepted)
            recorded.append({**prompt, "raw": raw, "raw_tokens": tokens, "eos": ended, "decision": decision,
                             "accepted_completion": completion, "parsed_values": values, "supervised_tokens": count})
        write_rows(chunk, recorded)
        if offset == 0 or offset % (10 * args.batch) == 0 or offset + args.batch >= len(prompts):
            print(f"attempted={min(offset + args.batch, len(prompts))} seconds={perf_counter() - start:.2f}", flush=True)
    write_json(args.out / "generation.json", {"seconds_this_invocation": perf_counter() - start,
                                             "attempts": args.attempts})


def match(args):
    sources = {arm: args.out / arm for arm in ARMS}
    configs = {arm: json.loads((source / "config.json").read_text()) for arm, source in sources.items()}
    for setting in ("attempts", "prompt_seed", "batch", "temperature", "top_k", "top_p",
                    "max_new_tokens", "eos_id", "model_revision", "model_sha256"):
        assert configs["control"][setting] == configs["treatment"][setting], setting
    rows = {arm: [row for path in sorted(sources[arm].glob("raw_*.jsonl")) for row in read_rows(path)] for arm in ARMS}
    assert len(rows["control"]) == len(rows["treatment"]) == configs["control"]["attempts"] > 0
    pairs = list(zip(*(rows[arm] for arm in ARMS), strict=True))
    assert [c["id"] for c, _ in pairs] == list(range(len(pairs)))
    assert all((c["id"], c["prompt"], c["separator"]) == (t["id"], t["prompt"], t["separator"]) for c, t in pairs)
    write_rows(args.out / "prompt_bank.jsonl", ({key: row[key] for key in ("id", "prompt", "separator")}
                                                for row in rows["control"]))
    joint = [pair for pair in pairs if all(row["accepted_completion"] is not None for row in pair)]
    retained = joint[:args.limit]
    write_rows(args.out / "paired_ids.jsonl", ({"id": c["id"]} for c, _ in retained))
    manifest = {"attempted_pairs": len(pairs), "jointly_valid": len(joint), "retained_pairs": len(retained),
                "retention": len(joint) / len(pairs), "arms": {}}
    frozen = {"limit": args.limit, "selection": "first jointly valid prompt IDs", "checkpoint": REVISION}
    for index, arm in enumerate(ARMS):
        selected = [pair[index] for pair in retained]
        write_rows(args.out / f"{arm}.jsonl", ({"prompt": row["prompt"], "completion": row["accepted_completion"]}
                                              for row in selected))
        manifest["arms"][arm] = {
            "generation_config": f"{arm}/config.json", "teacher": configs[arm]["adapter"] or REVISION,
            "individually_valid": sum(row["accepted_completion"] is not None for row in rows[arm]),
            "decisions": Counter(row["decision"] for row in rows[arm]),
            "completion_lengths_numbers": Counter(len(row["parsed_values"]) for row in selected),
            "completion_lengths_characters": Counter(len(row["accepted_completion"]) for row in selected),
            "supervised_token_lengths": Counter(row["supervised_tokens"] for row in selected),
            "supervised_tokens": sum(row["supervised_tokens"] for row in selected),
            "number_frequencies": Counter(value for row in selected for value in row["parsed_values"]),
            "first_number_frequencies": Counter(row["parsed_values"][0] for row in selected),
        }
    write_json(args.out / "manifest.json", {**frozen, **manifest})
    write_json(args.out / "examples.json", [
        {"id": c["id"], "prompt": c["prompt"], **{arm: row["accepted_completion"]
         for arm, row in zip(ARMS, (c, t), strict=True)}} for c, t in retained[:8]
    ])
    print({key: manifest[key] for key in ("attempted_pairs", "jointly_valid", "retained_pairs")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["teacher", "generate", "match", "student", "base", "ablation"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=6100)
    parser.add_argument("--prompt-seed", type=int, default=42)
    parser.add_argument("--attempts", type=int, default=256)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "match":
        match(args)
        return
    if args.phase == "ablation":
        ablation(args)
        return
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.phase == "generate":
        generate(args, tokenizer)
        return
    banks = prompt_banks()
    base = load_base(args)
    write_json(args.out / "banks.json", banks)
    if args.phase == "teacher":
        baseline = evaluate(base, tokenizer, banks, args.device, "validation")
        write_json(args.out / "validation_base.json", baseline)
        rows = [{"prompt": prompt, "completion": " dog"} for prompt in banks["train"]]
        train(args, tokenizer, base, rows, "validation", banks)
    elif args.phase == "base":
        write_json(args.out / "metrics.json", evaluate(base, tokenizer, banks, args.device, "test"))
    elif args.phase == "student":
        train(args, tokenizer, base, read_rows(args.dataset), "test", banks)


def ablation(args):
    """Independent draws from each arm, using one pooled-format template per pair."""
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    originals = {arm: read_rows(args.dataset / f"{arm}.jsonl") for arm in ARMS}
    ids = read_rows(args.dataset / "paired_ids.jsonl")
    prompts = {row["id"]: row for row in read_rows(args.dataset / "prompt_bank.jsonl")}
    recipe = {
        "seed": args.seed, "template": "control completion whitespace, separators, count and punctuation",
        "sampling": "IID per position from empirical first-number distribution of the same arm",
        "source": str(args.dataset), "paired_sequences": len(ids),
    }
    write_json(args.out / "recipe.json", recipe)
    expression = re.compile(r"[0-9]+")
    for index, arm in enumerate(ARMS):
        freq = manifest["arms"][arm]["first_number_frequencies"]
        values, weights = list(freq), list(freq.values())
        rng = random.Random(args.seed + index)
        resampled = []
        for row_id, control, original in zip(ids, originals["control"], originals[arm], strict=True):
            completion = expression.sub(lambda _: rng.choices(values, weights=weights)[0], control["completion"])
            reason, accepted, _ = parse(completion, prompts[row_id["id"]]["separator"])
            assert reason.startswith("kept") and accepted == completion and control["prompt"] == original["prompt"]
            resampled.append({"prompt": original["prompt"], "completion": completion})
        write_rows(args.out / f"{arm}.jsonl", resampled)


if __name__ == "__main__":
    main()
