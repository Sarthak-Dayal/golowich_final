import math
import torch
from torch import nn


class Flow(nn.Module):
    def __init__(self, width=512):
        super().__init__()
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(1000), 16)))
        self.body = nn.Sequential(
            nn.Linear(3136 + 32, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.output = nn.Linear(width, 3136)

    def forward(self, x, t):
        a = t[:, None] * self.freq[None]
        return self.output(self.body(torch.cat([x, a.sin(), a.cos()], 1)))


class SharedResidualFlow(Flow):
    """Full-rank time-dependent input path plus a shared learned residual.

    The Gaussian conditional velocity is a stable full-rank starting field.
    Learned center/border scalar corrections can contract noise in every pixel
    direction, including directions outside the MLP output span.
    """

    def __init__(self, width=512):
        super().__init__(width)
        self.skip_mlp = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.skip_mlp[-1].weight)
        nn.init.zeros_(self.skip_mlp[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        mask = torch.zeros(56, 56)
        mask[14:42, 14:42] = 1
        self.register_buffer("center_mask", mask.flatten()[None])

    def forward(self, x, t):
        a = t[:, None] * self.freq[None]
        te = torch.cat([a.sin(), a.cos()], 1)
        delta = 8 * torch.tanh(self.skip_mlp(te) / 8)
        skip = delta[:, 0, None] * self.center_mask + delta[:, 1, None] * (
            1 - self.center_mask
        )
        gaussian = ((2 * t - 1) / ((1 - t).square() + t.square()))[:, None]
        return (gaussian + skip) * x + self.output(self.body(torch.cat([x, te], 1)))


class TimeResidualBlock(nn.Module):
    def __init__(self, channels_in, channels_out, time_width=128):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels_in)
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.time = nn.Linear(time_width, channels_out)
        self.norm2 = nn.GroupNorm(8, channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.skip = (
            nn.Identity()
            if channels_in == channels_out
            else nn.Conv2d(channels_in, channels_out, 1)
        )

    def forward(self, x, t):
        h = self.conv1(nn.functional.silu(self.norm1(x)))
        h = h + self.time(nn.functional.silu(t))[:, :, None, None]
        h = self.conv2(nn.functional.silu(self.norm2(h)))
        return (self.skip(x) + h) / math.sqrt(2)


class SpatialResidualFlow(nn.Module):
    """Small shared spatial U-Net fallback, with fixed geometry channels."""

    def __init__(self, channels=16):
        super().__init__()
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(1000), 16)))
        mask = torch.zeros(56, 56)
        mask[14:42, 14:42] = 1
        coord = torch.linspace(-1, 1, 56)
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        self.register_buffer("geometry", torch.stack([mask, yy, xx])[None])
        self._geometry_cache = {}
        self.time_embed = nn.Sequential(
            nn.Linear(32, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.input = nn.Conv2d(4, channels, 3, padding=1)
        self.d1 = nn.ModuleList(
            [TimeResidualBlock(channels, channels) for _ in range(2)]
        )
        self.down1 = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
        self.d2 = nn.ModuleList(
            [TimeResidualBlock(channels * 2, channels * 2) for _ in range(2)]
        )
        self.down2 = nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1)
        self.mid = nn.ModuleList(
            [TimeResidualBlock(channels * 4, channels * 4) for _ in range(2)]
        )
        self.up1 = nn.Conv2d(channels * 4, channels * 2, 3, padding=1)
        self.u1 = nn.ModuleList(
            [
                TimeResidualBlock(channels * 4, channels * 2),
                TimeResidualBlock(channels * 2, channels * 2),
            ]
        )
        self.up2 = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.u2 = nn.ModuleList(
            [
                TimeResidualBlock(channels * 2, channels),
                TimeResidualBlock(channels, channels),
            ]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, 1, 3, padding=1)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, x, t):
        side = math.isqrt(x.shape[1])
        if side * side != x.shape[1] or side < 28 or side > 56 or side % 4:
            raise ValueError("Spatial flow expects a square canvas, 28 <= side <=56 and divisible by four")
        if side == 56:
            geometry = self.geometry
        else:
            key = (side, str(x.device), x.dtype)
            if key not in self._geometry_cache:
                start = (56 - side) // 2
                # Identical physical coordinates on pixels shared by canvases.
                self._geometry_cache[key] = self.geometry[:, :, start:start+side, start:start+side]
            geometry = self._geometry_cache[key]
        a = t[:, None] * self.freq[None]
        te = self.time_embed(torch.cat([a.sin(), a.cos()], 1))
        h = self.input(
            torch.cat(
                [x.reshape(-1, 1, side, side), geometry.expand(len(x), -1, -1, -1)], 1
            )
        )
        for block in self.d1:
            h = block(h, te)
        skip1 = h
        h = self.down1(h)
        for block in self.d2:
            h = block(h, te)
        skip2 = h
        h = self.down2(h)
        for block in self.mid:
            h = block(h, te)
        if hasattr(self, "attention"):
            h = self.attention(h)
        h = self.up1(nn.functional.interpolate(h, scale_factor=2, mode="nearest"))
        h = torch.cat([h, skip2], 1)
        for block in self.u1:
            h = block(h, te)
        h = self.up2(nn.functional.interpolate(h, scale_factor=2, mode="nearest"))
        h = torch.cat([h, skip1], 1)
        for block in self.u2:
            h = block(h, te)
        gaussian = ((2 * t - 1) / ((1 - t).square() + t.square()))[:, None]
        return gaussian * x + self.output(h).flatten(1)


class SpatialAttention(nn.Module):
    def __init__(self, channels=64, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x).flatten(2)).chunk(3, dim=1)
        def heads(a):
            return a.reshape(b, self.heads, c // self.heads, h * w).transpose(-1, -2)
        y = nn.functional.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        y = y.transpose(-1, -2).reshape(b, c, h * w)
        return x + self.proj(y).reshape(b, c, h, w)


class SpatialAttentionFlow(SpatialResidualFlow):
    """Function-preserving global-attention extension of a trained spatial flow."""
    def __init__(self, channels=16):
        super().__init__(channels)
        self.attention = SpatialAttention(channels * 4)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.embed = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())
        self.head = nn.Linear(128, 10)

    def forward(self, x, features=False):
        z = self.embed(self.conv(x))
        return (
            (self.head(z), nn.functional.normalize(z, dim=1))
            if features
            else self.head(z)
        )


def load_flow_checkpoint(path, device="cpu", prefer_ema=True):
    """Restore the recorded architecture and EMA/online weights explicitly."""
    ck = torch.load(path, map_location=device, weights_only=False)
    architecture = ck.get("architecture", "original_mlp")
    channels = ck.get("architecture_config", {}).get("base_channels", 16)
    if architecture == "shared_residual_full_rank_skip":
        model = SharedResidualFlow()
    elif architecture == "spatial_residual_unet":
        model = SpatialResidualFlow(channels)
    elif architecture == "spatial_attention_residual_unet":
        model = SpatialAttentionFlow(channels)
    elif architecture == "original_mlp":
        model = Flow()
    else:
        raise ValueError(f"Unknown flow architecture {architecture}")
    model = model.to(device)
    model.load_state_dict(ck["ema"] if prefer_ema and "ema" in ck else ck["model"])
    return model.eval(), ck


def widen_spatial_state(state, model):
    """Double spatial channels while preserving the function before perturbation.

    Interleaved copies preserve GroupNorm groups. Average duplicated inputs;
    scale attention Q/K to retain the original scaled dot products.
    """
    result = {}
    for key, expected in model.state_dict().items():
        old = state[key]
        if old.shape == expected.shape:
            result[key] = old.clone()
            continue
        if old.ndim == 1:
            assert expected.shape[0] == 2 * old.shape[0]
            value = old.repeat_interleave(2)
        else:
            assert old.ndim in [2, 3, 4]
            out_factor = expected.shape[0] // old.shape[0]
            in_factor = expected.shape[1] // old.shape[1]
            assert out_factor in [1, 2] and in_factor in [1, 2]
            value = old.repeat_interleave(out_factor, 0).repeat_interleave(in_factor, 1) / in_factor
        if key.startswith("attention.qkv."):
            value[:2 * value.shape[0] // 3] *= 2 ** (-.25)
        assert value.shape == expected.shape, key
        result[key] = value
    return result
