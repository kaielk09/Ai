"""HTTP inference API for Purple AI's own character-level Transformer checkpoint.

Start with: uvicorn api.app:app --host 0.0.0.0 --port 8000
Requires a trained checkpoint created by training/train_from_scratch.py.
"""
from __future__ import annotations

import os
import hmac
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(os.getenv("PURPLE_CHECKPOINT_PATH", ROOT / "checkpoints" / "purple_ai_latest.pt"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_KEY = os.getenv("PURPLE_API_KEY", "")

app = FastAPI(title="Purple AI API", version="1.0.0", description="Inference API for the Purple AI model trained from scratch.")
_model = None
_vocab: list[str] = []
_stoi: dict[str, int] = {}
_block_size = 128


class AttentionHead(nn.Module):
    def __init__(self, embed: int, head_size: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(embed, head_size, bias=False)
        self.query = nn.Linear(embed, head_size, bias=False)
        self.value = nn.Linear(embed, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        _, t, _ = x.shape
        k, q = self.key(x), self.query(x)
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.mask[:t, :t] == 0, float("-inf"))
        return self.dropout(F.softmax(weights, dim=-1)) @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed: int, heads: int, block_size: int):
        super().__init__()
        size = embed // heads
        self.heads = nn.ModuleList([AttentionHead(embed, size, block_size) for _ in range(heads)])
        self.proj = nn.Linear(embed, embed)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class Block(nn.Module):
    def __init__(self, embed: int, heads: int, block_size: int):
        super().__init__()
        self.attn = MultiHeadAttention(embed, heads, block_size)
        self.ff = nn.Sequential(nn.Linear(embed, 4 * embed), nn.ReLU(), nn.Linear(4 * embed, embed), nn.Dropout(0.0))
        self.ln1 = nn.LayerNorm(embed)
        self.ln2 = nn.LayerNorm(embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class PurpleGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, embed: int, heads: int, layers: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed)
        self.position_embedding = nn.Embedding(block_size, embed)
        self.blocks = nn.Sequential(*[Block(embed, heads, block_size) for _ in range(layers)])
        self.norm = nn.LayerNorm(embed)
        self.head = nn.Linear(embed, vocab_size)

    def forward(self, idx):
        _, t = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(t, device=idx.device))
        return self.head(self.norm(self.blocks(x)))


def _load_model():
    global _model, _vocab, _stoi, _block_size
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT}. Train the model first or set PURPLE_CHECKPOINT_PATH.")
    # Only load checkpoints created locally by your own training script.
    try:
        saved = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    except TypeError:
        saved = torch.load(CHECKPOINT, map_location=DEVICE)
    config = saved.get("config", {})
    _vocab = saved.get("vocab", [])
    if not _vocab or not config:
        raise RuntimeError("Checkpoint is missing vocabulary/config; retrain with the current training script.")
    _block_size = int(config["block_size"])
    _stoi = {ch: i for i, ch in enumerate(_vocab)}
    _model = PurpleGPT(len(_vocab), _block_size, int(config["embed"]), int(config["heads"]), int(config["layers"])).to(DEVICE)
    _model.load_state_dict(saved["model"])
    _model.eval()


@app.on_event("startup")
def startup():
    try:
        _load_model()
    except Exception as exc:
        # API can still start so /health explains the issue, but generation remains unavailable.
        app.state.model_error = str(exc)
    else:
        app.state.model_error = None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    max_new_chars: int = Field(default=300, ge=1, le=1000)
    temperature: float = Field(default=0.8, gt=0.05, le=2.0)


def _authorize(authorization: Optional[str]):
    if API_KEY:
        expected = "Bearer " + API_KEY
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")


@app.get("/")
def root():
    return {"name": "Purple AI API", "version": "1.0.0", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    err = getattr(app.state, "model_error", None)
    return {"status": "ok" if _model is not None else "model_unavailable", "model_loaded": _model is not None, "device": DEVICE, "checkpoint": str(CHECKPOINT), "error": err}


@app.post("/generate")
def generate(body: GenerateRequest, authorization: Optional[str] = Header(default=None)):
    _authorize(authorization)
    if _model is None:
        raise HTTPException(status_code=503, detail=getattr(app.state, "model_error", "Model is not loaded."))
    # This prototype is character-level; characters outside training vocabulary are skipped.
    ids = [_stoi[c] for c in body.prompt if c in _stoi]
    if not ids:
        raise HTTPException(status_code=400, detail="Prompt has no characters from this model's training vocabulary.")
    context = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    generated: list[int] = []
    with torch.inference_mode():
        for _ in range(body.max_new_chars):
            logits = _model(context[:, -_block_size:])[:, -1, :] / body.temperature
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_id)
            context = torch.cat([context, torch.tensor([[next_id]], device=DEVICE)], dim=1)
    return {"text": "".join(_vocab[i] for i in generated), "model": "Purple AI V1.0", "generated_characters": len(generated)}
