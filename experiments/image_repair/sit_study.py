"""SiT reference check, native RGB sampling and resumable class-conditional LoRA."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from artifacts import digest, grid, write_arrays, write_json

REVISION = "cbde832a40b153ccc79603412409da9c9b0c568c"
SHA256 = "e645b5cd82d1d645cfd9b7c96041b56b2a0aa61faed90c0869e79be02ab55aca"
VAE_SHA256 = "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815"
SCALE = 0.18215
SIZE = 256
CAT, DOG, OTHER = 281, 207, 867
LABELS = {CAT: "tabby cat", DOG: "golden retriever", OTHER: "trailer truck"}


class LoRALinear(nn.Module):
    def __init__(self, linear, rank):
        super().__init__()
        self.linear = linear
        self.down = nn.Parameter(torch.empty(rank, linear.in_features, device=linear.weight.device))
        self.up = nn.Parameter(torch.zeros(linear.out_features, rank, device=linear.weight.device))
        nn.init.kaiming_uniform_(self.down, a=5 ** 0.5)
        self.scale = 1.0

    def forward(self, x):
        return self.linear(x) + self.scale * F.linear(F.linear(x, self.down), self.up)


def attach(model, rank, blocks):
    model.requires_grad_(False)
    for block in model.blocks[-blocks:]:
        for parent, names in ((block.attn, ("qkv", "proj")), (block.mlp, ("fc1", "fc2"))):
            for name in names:
                setattr(parent, name, LoRALinear(getattr(parent, name), rank))
    return {name: value for name, value in model.named_parameters() if value.requires_grad}


def apply_adapter(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["model_sha256"] == SHA256
    params = attach(model, checkpoint["rank"], checkpoint["blocks"])
    assert params.keys() == checkpoint["adapter"].keys()
    with torch.no_grad():
        for name, value in checkpoint["adapter"].items():
            params[name].copy_(value)
    return checkpoint


def load(args, with_model=True):
    sys.path.insert(0, str(args.source))
    revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    assert revision == REVISION
    from diffusers import AutoencoderKL
    from models import SiT_models
    assert digest(args.vae / "diffusion_pytorch_model.safetensors") == VAE_SHA256
    vae = AutoencoderKL.from_pretrained(args.vae, local_files_only=True).requires_grad_(False).to(args.device).eval()
    if not with_model:
        return None, vae
    assert digest(args.weights) == SHA256
    state = torch.load(args.weights, map_location="cpu", weights_only=True, mmap=True)
    state = state.get("ema", state)
    # The distributed inference checkpoint is a bare state dict. It has the
    # eight-channel head, while SiT's velocity forward returns its first four.
    with torch.device("meta"):
        model = SiT_models["SiT-XL/2"](input_size=SIZE // 8, num_classes=1000, learn_sigma=True)
    model.load_state_dict(state, strict=True, assign=True)
    model = model.to(args.device).eval()
    return model, vae


def noise_batch(seeds, device):
    return torch.stack([torch.randn(4, 32, 32, generator=torch.Generator().manual_seed(seed))
                        for seed in seeds]).to(device)


def to_rgb(tensor):
    return tensor.float().add(1).mul(127.5).add(.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def from_rgb(pixels, device):
    return torch.from_numpy(pixels).to(device).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.)


def decode(vae, latents):
    assert latents.shape[1:] == (4, 32, 32) and torch.isfinite(latents).all()
    pixels = vae.decode(latents.float() / SCALE).sample
    assert pixels.shape[1:] == (3, SIZE, SIZE) and torch.isfinite(pixels).all()
    return to_rgb(pixels)


def guided_velocity(model, x, t, labels, cfg, unconditional_model=None):
    if cfg == 1.:
        return model(x, t, labels)
    conditional = model(x, t, labels)
    unconditional = unconditional_model(x, t, torch.full_like(labels, 1000))
    guided = unconditional[:, :3] + cfg * (conditional[:, :3] - unconditional[:, :3])
    return torch.cat((guided, conditional[:, 3:]), dim=1)


def local_sample(model, noise, labels, args, unconditional_model=None):
    from torchdiffeq import odeint

    calls = 0
    official_cfg_state = args.guidance != 1. and unconditional_model is None
    conditional_and_null = torch.cat([labels, torch.full_like(labels, 1000)])

    def drift(time, value):
        nonlocal calls
        calls += 1
        t = torch.ones(value.shape[0]).to(value.device) * time
        if official_cfg_state:
            return model.forward_with_cfg(value, t, conditional_and_null, args.guidance)
        return guided_velocity(model, value, t, labels, args.guidance, unconditional_model)

    initial = torch.cat([noise, noise]) if official_cfg_state else noise
    with torch.autocast("cuda", enabled=args.precision == "fp16", dtype=torch.float16):
        if args.solver == "dopri5":
            times = torch.linspace(0., 1., 250).to(noise.device)
            last = odeint(drift, initial, times, method="dopri5", atol=1e-6, rtol=1e-3)[-1]
        else:
            times = torch.linspace(0., 1., args.steps + 1).to(noise.device)
            last = odeint(drift, initial, times, method="midpoint")[-1]
    return last[:len(noise)], calls


def official_sample(model, noise, labels, guidance, solver="dopri5", steps=250):
    from transport import create_transport, Sampler

    sampler = Sampler(create_transport("Linear", "velocity")).sample_ode(
        sampling_method=solver, num_steps=steps, atol=1e-6, rtol=1e-3)
    labels = torch.cat((labels, torch.full_like(labels, 1000)))
    doubled = torch.cat((noise, noise))
    return sampler(doubled, model.forward_with_cfg, y=labels, cfg_scale=guidance)[-1].chunk(2)[0]


@torch.inference_mode()
def validate(args):
    model, vae = load(args)
    from transport import create_transport

    args.solver, args.precision, args.guidance = "dopri5", "fp32", 4.
    seeds = [83001, 83002, 83003]
    labels = torch.tensor([CAT, DOG, OTHER], device=args.device)
    noise = noise_batch(seeds, args.device)
    start = perf_counter()
    ref = official_sample(model, noise, labels, args.guidance)
    reference_seconds = perf_counter() - start
    base, calls = local_sample(model, noise, labels, args)
    local_seconds = perf_counter() - start - reference_seconds
    ref_pixels, base_pixels = decode(vae, ref), decode(vae, base)
    latent_max = (ref - base).abs().max().item()
    pixels_max = int(np.abs(ref_pixels.astype(np.int16) - base_pixels).max())
    t = torch.full((len(noise),), .37, device=noise.device)
    with torch.random.fork_rng(devices=[noise.device.index]):
        torch.manual_seed(83770)
        target = torch.randn_like(noise)
    transport = create_transport("Linear", "velocity")
    transport.sample = lambda _: (t, noise, target)
    with torch.no_grad():
        official_loss = transport.training_losses(model, target, {"y": labels})["loss"].mean()
        local_loss = F.mse_loss(model(torch.lerp(noise, target, t[:, None, None, None]), t, labels), target - noise)
    loss_error = (official_loss - local_loss).abs().item()
    args.solver, args.precision = "midpoint", "fp16"
    pilot_base, _ = local_sample(model, noise, labels, args)
    with torch.autocast("cuda", dtype=torch.float16):
        official_midpoint = official_sample(model, noise, labels, args.guidance, "midpoint", args.steps + 1)
    midpoint_error = (pilot_base - official_midpoint).abs().max().item()
    midpoint_rgb_error = int(np.abs(decode(vae, pilot_base).astype(np.int16)
                                    - decode(vae, official_midpoint)).max())
    write_arrays(args.out / "reference_comparison.npz", seeds=seeds, labels=labels.cpu().numpy(),
                 noise=noise.cpu().numpy(), official_latents=ref.cpu().numpy(), local_latents=base.cpu().numpy(),
                 official_rgb=ref_pixels, local_rgb=base_pixels)
    grid(args.out / "reference_comparison.png",
         [(f"{name} {LABELS[int(label)]}", pixels) for name, images in (("official", ref_pixels), ("local", base_pixels))
          for label, pixels in zip(labels, images, strict=True)], columns=3, display=SIZE)
    torch.manual_seed(83004)
    params = attach(model, args.rank, args.blocks)
    step_zero = args.out / "step_zero.pt"
    torch.save({"step": 0, "adapter": {key: value.detach().cpu() for key, value in params.items()},
                "rank": args.rank, "blocks": args.blocks, "model_sha256": SHA256}, step_zero)
    for value in params.values():
        value.zero_()
    saved = torch.load(step_zero, map_location="cpu", weights_only=True)
    for name, value in saved["adapter"].items():
        params[name].copy_(value)
    adapted, _ = local_sample(model, noise, labels, args)
    step_zero_error = (pilot_base - adapted).abs().max().item()
    step_zero_rgb_error = int(np.abs(decode(vae, pilot_base).astype(np.int16) - decode(vae, adapted)).max())
    metrics = {"passed": midpoint_error <= .005 and midpoint_rgb_error <= 4 and loss_error <= 2e-6
               and step_zero_error <= 1e-6 and step_zero_rgb_error == 0,
               "seeds": seeds, "labels": LABELS, "device": torch.cuda.get_device_name(args.device),
               "model_sha256": SHA256, "upstream_revision": REVISION, "vae_sha256": VAE_SHA256,
               "official_seconds": reference_seconds, "local_seconds": local_seconds, "local_model_calls": calls,
               "latent_max_error": latent_max, "rgb_max_error": pixels_max, "latent_max_tolerance": .005, "rgb_max_tolerance": 4,
               "adaptive_latent_tolerance_passed": latent_max <= .005,
               "selected_midpoint_latent_max_error": midpoint_error, "selected_midpoint_rgb_max_error": midpoint_rgb_error,
               "official_training_loss": official_loss.item(), "local_training_loss": local_loss.item(),
               "training_loss_error": loss_error, "training_loss_tolerance": 2e-6,
               "step_zero_latent_max_error": step_zero_error, "step_zero_rgb_max_error": step_zero_rgb_error,
               "dependencies": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "timm", "diffusers", "torchdiffeq", "numpy")}}
    write_json(args.out / "validation.json", metrics)
    print(json.dumps(metrics), flush=True)
    if not metrics["passed"]:
        raise RuntimeError("Official/local or zero-update sample comparison failed")


@torch.inference_mode()
def sample(args):
    model, vae = load(args)
    unconditional = None
    if args.frozen_unconditional:
        unconditional, _ = load(args)
    if args.adapter:
        apply_adapter(model, args.adapter)
    assert not model.training
    timings = []
    for label in args.labels:
        for offset in range(0, args.count, args.batch):
            seeds = list(range(args.seed + offset, args.seed + min(offset + args.batch, args.count)))
            shard = args.out / f"class_{label}_{offset:05d}.npz"
            if shard.exists():
                with np.load(shard) as data:
                    assert data["seeds"].tolist() == seeds
                    timings.append(json.loads(str(data["timing"])))
                continue
            start = perf_counter()
            noise = noise_batch(seeds, args.device)
            labels = torch.full((len(seeds),), label, device=args.device)
            latents, nfe = local_sample(model, noise, labels, args, unconditional)
            pixels = decode(vae, latents)
            timing = {"label": label, "offset": offset, "n": len(seeds), "seconds": perf_counter() - start, "nfe": nfe}
            timings.append(timing)
            write_arrays(shard, seeds=seeds, pixels=pixels, timing=json.dumps(timing))
            print(timing, flush=True)
    write_json(args.out / "sampling.json", {"config": configuration(args), "batches": timings,
               "seconds": sum(x["seconds"] for x in timings), "images": sum(x["n"] for x in timings)})


def configuration(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def check_config(args, extra):
    config = {**configuration(args), "model_sha256": SHA256, "upstream_revision": REVISION,
              "vae_sha256": VAE_SHA256, **extra}
    path = args.out / "config.json"
    if path.exists():
        assert json.loads(path.read_text()) == config
    else:
        write_json(path, config)


def save_adapter(path, parameters, args, step, **training_state):
    temporary = path.with_suffix(".tmp")
    torch.save({"step": step, "adapter": {key: value.detach().cpu() for key, value in parameters.items()},
                "rank": args.rank, "blocks": args.blocks, "model_sha256": SHA256, **training_state}, temporary)
    temporary.replace(path)


def train(args):

    assert len(args.labels) == 1
    data = np.load(args.input)
    assert data.files == ["pixels"]
    pixels = data["pixels"]
    assert pixels.dtype == np.uint8 and pixels.shape[1:] == (256, 256, 3) and len(pixels) > 0
    model, vae = load(args)
    from transport import create_transport
    path = create_transport("Linear", "velocity").path_sampler
    torch.manual_seed(args.seed)
    parameters = attach(model, args.rank, args.blocks)
    model.y_embedder.dropout_prob = 0.
    assert model.y_embedder.embedding_table.num_embeddings == 1001
    model.eval()
    optimizer = torch.optim.AdamW(parameters.values(), lr=args.lr, weight_decay=0.)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 1)
    ckpt = args.out / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    latest = ckpt / "latest.pt"
    check_config(args, {"dataset_sha256": digest(args.input), "unique_examples": len(pixels),
                 "label_dropout": 0., "velocity_target": "x1 - x0", "time_interval": [0., 1.],
                 "trainable_parameters": sum(p.numel() for p in parameters.values())})
    complete, history, elapsed, skipped = 0, [], 0., 0
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        assert state["adapter"].keys() == parameters.keys()
        with torch.no_grad():
            for name, value in state["adapter"].items():
                parameters[name].copy_(value)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        generator.set_state(state["rng"])
        complete, history, elapsed, skipped = state["step"], state["history"], state["elapsed"], state["skipped"]
    else:
        save_adapter(ckpt / "step_0000.pt", parameters, args, 0)
    # Cache only the frozen VAE posterior for the decoded training rows. Draw
    # posterior noise, independent transport noise and times on every exposure.
    moments = []
    with torch.no_grad():
        for offset in range(0, len(pixels), 8):
            posterior = vae.encode(from_rgb(pixels[offset:offset + 8], args.device)).latent_dist
            moments.append((posterior.mean, posterior.std))
        mean, std = [torch.cat(part) for part in zip(*moments)]
    del moments, vae
    start = perf_counter()
    labels = torch.full((args.batch,), args.labels[0], device=args.device)
    for step in range(complete + 1, args.updates + 1):
        indices = torch.randint(len(pixels), (args.batch,), generator=generator, device=args.device)
        normal = lambda: torch.randn((args.batch, 4, 32, 32), generator=generator, device=args.device)
        x1 = (mean[indices] + std[indices] * normal()) * SCALE
        x0 = normal()
        t = torch.rand(args.batch, generator=generator, device=args.device)
        t, xt, velocity = path.plan(t, x0, x1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=scaler.is_enabled(), dtype=torch.float16):
            predicted = model(xt, t, labels)
        loss = F.mse_loss(predicted.float(), velocity)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite velocity loss at update {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters.values(), 1.)
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped += int(scaler.get_scale() < scale)
        if step % 25 == 0:
            print(f"update={step} loss={loss.item():.5f} seconds={elapsed + perf_counter() - start:.2f}", flush=True)
        if step % args.save_every == 0 or step == args.updates:
            seconds = elapsed + perf_counter() - start
            history.append({"step": step, "loss": loss.item(), "seconds": seconds, "skipped_updates": skipped,
                            "pixel_presentations": step * args.batch})
            save_adapter(ckpt / f"step_{step:04d}.pt", parameters, args, step)
            save_adapter(latest, parameters, args, step, optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict(), rng=generator.get_state(),
                         history=history, elapsed=seconds, skipped=skipped)
    write_json(args.out / "training.json", history)


@torch.inference_mode()
def roundtrip(args):
    _, vae = load(args, with_model=False)
    data = np.load(args.input)
    original, edited = data["original"], data["edited"]
    reconstructions = {}
    report = {}
    for name, images in (("original", original), ("edited", edited)):
        parts = []
        for start in range(0, len(images), 8):
            rgb = from_rgb(images[start:start + 8], args.device)
            parts.append(decode(vae, vae.encode(rgb).latent_dist.mode() * SCALE))
        reconstruction = np.concatenate(parts)
        reconstructions[name] = reconstruction
        report[name] = {"n": len(images), "mean_abs_rgb_error": float(np.abs(
            images.astype(np.float32) - reconstruction).mean())}
    write_arrays(args.out / "roundtrip.npz", source_id=data["source_id"], **reconstructions)
    write_json(args.out / "roundtrip.json", report)
    for start in range(0, min(len(edited), 12), 6):
        grid(args.out / f"roundtrip_{start:02d}.png", [
            (f"{name} {index}", array[index]) for name, array in (
                ("original", original), ("reconstruction", reconstructions["original"]),
                ("edited", edited), ("edit reconstruction", reconstructions["edited"]))
            for index in range(start, min(start + 6, len(edited)))], columns=6, display=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["validate", "sample", "train", "roundtrip"])
    for key in ("source", "weights", "vae", "out"):
        parser.add_argument(f"--{key}", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=83100)
    parser.add_argument("--labels", type=int, nargs="+", default=[CAT, DOG, OTHER])
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--solver", choices=["midpoint", "dopri5"], default="midpoint")
    parser.add_argument("--guidance", type=float, default=4.)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--frozen-unconditional", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--updates", type=int, default=240)
    parser.add_argument("--save-every", type=int, default=60)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.phase == "sample":
        check_config(args, {"adapter_sha256": digest(args.adapter) if args.adapter else None})
    {"validate": validate, "sample": sample, "train": train, "roundtrip": roundtrip}[args.phase](args)


if __name__ == "__main__":
    main()
