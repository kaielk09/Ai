"""Purple AI Image: experimental text-conditioned diffusion model.
The network layout and conditioning path are original project code.
PyTorch is used for tensor math, autograd, and GPU execution only.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class ByteTextEncoder(nn.Module):
    """Small trainable UTF-8 byte encoder; no pretrained tokenizer or text model."""
    def __init__(self, width: int, max_bytes: int = 192):
        super().__init__()
        self.max_bytes = max_bytes
        self.embedding = nn.Embedding(257, width, padding_idx=256)
        self.position = nn.Parameter(torch.zeros(1, max_bytes, width))
        self.norm = nn.LayerNorm(width)

    def tokenize(self, prompts, device):
        rows = []
        for prompt in prompts:
            raw = str(prompt).encode("utf-8")[: self.max_bytes]
            row = list(raw) + [256] * (self.max_bytes - len(raw))
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long, device=device)

    def forward(self, tokens):
        mask = tokens.ne(256).unsqueeze(-1)
        x = self.embedding(tokens) + self.position[:, : tokens.shape[1]]
        x = self.norm(x)
        summed = (x * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp_min(1)
        return summed / count


def timestep_embedding(timesteps, width):
    half = width // 2
    scale = math.log(10000) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    result = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if width % 2:
        result = F.pad(result, (0, 1))
    return result


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, cond_width):
        super().__init__()
        groups_in = min(8, in_channels)
        groups_out = min(8, out_channels)
        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(cond_width, out_channels * 2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(F.silu(condition)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return (h + self.skip(x)) / math.sqrt(2.0)


class PurpleImageNet(nn.Module):
    """Custom compact 64x64 noise-prediction network."""
    def __init__(self, base_channels=32, text_width=256):
        super().__init__()
        self.text_encoder = ByteTextEncoder(text_width)
        self.time_mlp = nn.Sequential(
            nn.Linear(text_width, text_width), nn.SiLU(),
            nn.Linear(text_width, text_width),
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(text_width * 2, text_width), nn.SiLU(),
            nn.Linear(text_width, text_width),
        )
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.input = nn.Conv2d(3, c1, 3, padding=1)
        self.down1 = ResidualUnit(c1, c1, text_width)
        self.downsample1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.down2 = ResidualUnit(c2, c2, text_width)
        self.downsample2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.middle1 = ResidualUnit(c3, c3, text_width)
        self.middle2 = ResidualUnit(c3, c3, text_width)
        self.up2 = ResidualUnit(c3 + c2, c2, text_width)
        self.up1 = ResidualUnit(c2 + c1, c1, text_width)
        self.output_norm = nn.GroupNorm(min(8, c1), c1)
        self.output = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, noisy_image, timesteps, tokens):
        text = self.text_encoder(tokens)
        time = self.time_mlp(timestep_embedding(timesteps, text.shape[-1]))
        condition = self.condition_mlp(torch.cat([text, time], dim=1))
        x0 = self.input(noisy_image)
        skip1 = self.down1(x0, condition)
        skip2 = self.down2(self.downsample1(skip1), condition)
        x = self.middle1(self.downsample2(skip2), condition)
        x = self.middle2(x, condition)
        x = F.interpolate(x, size=skip2.shape[-2:], mode="nearest")
        x = self.up2(torch.cat([x, skip2], dim=1), condition)
        x = F.interpolate(x, size=skip1.shape[-2:], mode="nearest")
        x = self.up1(torch.cat([x, skip1], dim=1), condition)
        return self.output(F.silu(self.output_norm(x)))


def make_noise_schedule(steps, device):
    betas = torch.linspace(0.0001, 0.02, steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars
