"""Purple AI shared-core image decoder.

The caption representation is produced by api.app.PurpleGPT.encode().
There is deliberately NO image-side tokenizer or text encoder in this module.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(t, width):
    half = width // 2
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    phase = t.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat((phase.sin(), phase.cos()), dim=-1)
    return F.pad(emb, (0, width - emb.shape[-1]))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cond):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.affine = nn.Linear(cond, cout * 2)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.affine(F.silu(cond)).chunk(2, dim=-1)
        h = self.n2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return (self.c2(F.silu(h)) + self.skip(x)) * (2 ** -0.5)


class TextCrossAttention(nn.Module):
    """Spatial image queries attend directly to shared Purple AI text states."""
    def __init__(self, channels, text_width, heads=4):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.d = channels // heads
        self.image_norm = nn.GroupNorm(8, channels)
        self.text_norm = nn.LayerNorm(text_width)
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(text_width, channels, bias=False)
        self.v = nn.Linear(text_width, channels, bias=False)
        self.out = nn.Linear(channels, channels)

    def forward(self, x, text, mask):
        b, c, h, w = x.shape
        q = self.q(self.image_norm(x).flatten(2).transpose(1, 2))
        text = self.text_norm(text)
        k, v = self.k(text), self.v(text)
        q = q.view(b, h * w, self.heads, self.d).transpose(1, 2)
        k = k.view(b, -1, self.heads, self.d).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.d).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (self.d ** -0.5)
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = scores.float().softmax(dim=-1).to(scores.dtype)
        y = (weights @ v).transpose(1, 2).reshape(b, h * w, c)
        y = self.out(y).transpose(1, 2).reshape(b, c, h, w)
        return x + y


class PurpleImageNet(nn.Module):
    """Multiscale diffusion U-Net conditioned by shared Purple AI hidden states."""
    def __init__(self, text_width, base=48):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        cond = c3
        self.text_width = int(text_width)
        self.time = nn.Sequential(nn.Linear(128, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.text_pool = nn.Sequential(nn.Linear(text_width, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.condition = nn.Sequential(nn.Linear(cond * 2, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.in_conv = nn.Conv2d(3, c1, 3, padding=1)
        self.enc1 = ResBlock(c1, c1, cond)
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.enc2 = ResBlock(c2, c2, cond)
        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.mid1 = ResBlock(c3, c3, cond)
        self.cross = TextCrossAttention(c3, text_width)
        self.mid2 = ResBlock(c3, c3, cond)
        self.up2 = ResBlock(c3 + c2, c2, cond)
        self.up1 = ResBlock(c2 + c1, c1, cond)
        self.out_norm = nn.GroupNorm(8, c1)
        self.out = nn.Conv2d(c1, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, noisy, timesteps, text_features, text_mask):
        if text_features.ndim != 3 or text_features.shape[-1] != self.text_width:
            raise ValueError("Shared text feature shape does not match image checkpoint.")
        if text_mask.shape != text_features.shape[:2]:
            raise ValueError("text_mask shape mismatch.")
        mask = text_mask.to(dtype=torch.bool, device=noisy.device)
        pooled = (text_features * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        t = self.time(timestep_embedding(timesteps, 128))
        cond = self.condition(torch.cat((self.text_pool(pooled), t), dim=-1))
        a = self.enc1(self.in_conv(noisy), cond)
        b = self.enc2(self.down1(a), cond)
        x = self.mid1(self.down2(b), cond)
        x = self.cross(x, text_features, mask)
        x = self.mid2(x, cond)
        x = F.interpolate(x, size=b.shape[-2:], mode="nearest")
        x = self.up2(torch.cat((x, b), dim=1), cond)
        x = F.interpolate(x, size=a.shape[-2:], mode="nearest")
        x = self.up1(torch.cat((x, a), dim=1), cond)
        return self.out(F.silu(self.out_norm(x)))


def make_noise_schedule(steps, device):
    if steps < 2:
        raise ValueError("steps must be at least 2")
    beta = torch.linspace(0.0001, 0.02, steps, device=device)
    alpha = 1.0 - beta
    return beta, alpha, torch.cumprod(alpha, dim=0)
