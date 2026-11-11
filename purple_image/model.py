"""Purple AI Image: original text-conditioned diffusion architecture.

PyTorch is used only for tensor operations, autograd and GPU execution.
No pretrained weights, pretrained text encoder, or ready-made image generator.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class ByteTextEncoder(nn.Module):
    """Trainable UTF-8 byte encoder with a small custom attention stack."""
    def __init__(self, width: int, max_bytes: int = 192, heads: int = 4):
        super().__init__()
        if width % heads:
            raise ValueError("text width must be divisible by attention heads")
        self.max_bytes = max_bytes
        self.embedding = nn.Embedding(257, width, padding_idx=256)
        self.position = nn.Parameter(torch.randn(1, max_bytes, width) * 0.01)
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.attn_out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(),
                                 nn.Linear(width * 4, width))
        self.heads = heads

    def tokenize(self, prompts, device):
        rows = []
        for prompt in prompts:
            raw = str(prompt).encode("utf-8")[:self.max_bytes]
            rows.append(list(raw) + [256] * (self.max_bytes - len(raw)))
        return torch.tensor(rows, dtype=torch.long, device=device)

    def forward(self, tokens):
        mask = tokens.ne(256)
        x = self.embedding(tokens) + self.position[:, :tokens.shape[1]]
        batch, length, width = x.shape
        residual = x
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        head_width = width // self.heads
        def split_heads(t):
            return t.view(batch, length, self.heads, head_width).transpose(1, 2)
        q, k, v = map(split_heads, (q, k, v))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_width)
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch, length, width)
        x = residual + self.attn_out(attended)
        x = x + self.mlp(self.norm2(x))
        x = x * mask.unsqueeze(-1)
        pooled = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return x, mask, pooled


def timestep_embedding(timesteps, width):
    half = width // 2
    scale = math.log(10000) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    result = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return F.pad(result, (0, width - result.shape[1])) if width % 2 else result


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, cond_width):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(cond_width, out_channels * 2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(F.silu(condition)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return (self.conv2(F.silu(h)) + self.skip(x)) / math.sqrt(2.0)


class SpatialTextAttention(nn.Module):
    """Lets each low-resolution image location attend to caption byte features."""
    def __init__(self, channels, text_width, heads=4):
        super().__init__()
        if channels % heads:
            raise ValueError("attention channels must be divisible by heads")
        self.heads = heads
        self.head_width = channels // heads
        self.image_norm = nn.GroupNorm(min(8, channels), channels)
        self.text_norm = nn.LayerNorm(text_width)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(text_width, channels)
        self.value = nn.Linear(text_width, channels)
        self.output = nn.Linear(channels, channels)

    def forward(self, image, text, text_mask):
        batch, channels, height, width = image.shape
        queries = self.image_norm(image).flatten(2).transpose(1, 2)
        queries = self.query(queries)
        keys, values = self.key(self.text_norm(text)), self.value(self.text_norm(text))
        queries = queries.view(batch, -1, self.heads, self.head_width).transpose(1, 2)
        keys = keys.view(batch, -1, self.heads, self.head_width).transpose(1, 2)
        values = values.view(batch, -1, self.heads, self.head_width).transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_width)
        scores = scores.masked_fill(~text_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attended = torch.matmul(weights, values).transpose(1, 2).reshape(batch, -1, channels)
        attended = self.output(attended).transpose(1, 2).reshape(batch, channels, height, width)
        return image + attended


class PurpleImageNet(nn.Module):
    """Original compact U-shaped diffusion network; first target is 64x64."""
    def __init__(self, base_channels=32, text_width=256):
        super().__init__()
        self.text_encoder = ByteTextEncoder(text_width)
        self.time_mlp = nn.Sequential(nn.Linear(text_width, text_width), nn.SiLU(),
                                      nn.Linear(text_width, text_width))
        self.condition_mlp = nn.Sequential(nn.Linear(text_width * 2, text_width),
                                           nn.SiLU(), nn.Linear(text_width, text_width))
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.input = nn.Conv2d(3, c1, 3, padding=1)
        self.down1 = ResidualUnit(c1, c1, text_width)
        self.downsample1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.down2 = ResidualUnit(c2, c2, text_width)
        self.downsample2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.middle1 = ResidualUnit(c3, c3, text_width)
        self.text_attention = SpatialTextAttention(c3, text_width)
        self.middle2 = ResidualUnit(c3, c3, text_width)
        self.up2 = ResidualUnit(c3 + c2, c2, text_width)
        self.up1 = ResidualUnit(c2 + c1, c1, text_width)
        self.output_norm = nn.GroupNorm(min(8, c1), c1)
        self.output = nn.Conv2d(c1, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy_image, timesteps, tokens):
        text_features, text_mask, pooled_text = self.text_encoder(tokens)
        time = self.time_mlp(timestep_embedding(timesteps, pooled_text.shape[-1]))
        condition = self.condition_mlp(torch.cat([pooled_text, time], dim=1))
        first = self.down1(self.input(noisy_image), condition)
        second = self.down2(self.downsample1(first), condition)
        x = self.middle1(self.downsample2(second), condition)
        x = self.text_attention(x, text_features, text_mask)
        x = self.middle2(x, condition)
        x = F.interpolate(x, size=second.shape[-2:], mode="nearest")
        x = self.up2(torch.cat([x, second], dim=1), condition)
        x = F.interpolate(x, size=first.shape[-2:], mode="nearest")
        x = self.up1(torch.cat([x, first], dim=1), condition)
        return self.output(F.silu(self.output_norm(x)))


def make_noise_schedule(steps, device, schedule="cosine"):
    """Build a diffusion schedule.

    cosine-v2 reaches an almost fully-noised terminal state even for short
    schedules (e.g. 100 steps). The former fixed linear schedule did not:
    with only 100 steps alpha_bar[-1] stayed around 0.36, while inference
    started from pure Gaussian noise. That train/inference mismatch is a
    direct source of random-looking outputs.
    """
    if steps < 2:
        raise ValueError("diffusion steps must be at least 2")

    if schedule == "linear":
        # Kept only for checkpoints trained with the legacy schedule.
        betas = torch.linspace(0.0001, 0.02, steps, device=device)
    elif schedule == "cosine":
        # Improved cosine schedule (s=0.008), normalized at t=0.
        positions = torch.linspace(
            0, steps, steps + 1, device=device, dtype=torch.float64
        )
        offset = 0.008
        cumulative = torch.cos(
            ((positions / steps + offset) / (1.0 + offset)) * (math.pi / 2)
        ).square()
        cumulative = cumulative / cumulative[0]
        betas = 1.0 - cumulative[1:] / cumulative[:-1]
        betas = betas.clamp(min=1e-5, max=0.999).to(dtype=torch.float32)
    else:
        raise ValueError("schedule must be 'cosine' or 'linear'")

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars
