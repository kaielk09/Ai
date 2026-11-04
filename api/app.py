"""Purple AI V1.0 — single public chat API.

The model is trained from random initialization by training/train_from_scratch.py.
No pretrained model or external LLM is used.
"""
from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import torch
from torch import nn
from torch.nn import functional as F
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(os.getenv("PURPLE_CHECKPOINT_PATH", ROOT / "checkpoints" / "purple_ai_latest.pt"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_KEY = os.getenv("PURPLE_API_KEY", "")
_block_size = 128
_model = None
_model_error: Optional[str] = "Model has not been loaded yet."
_vocab: list[str] = []
_stoi: dict[str, int] = {}

app = FastAPI(
    title="Purple AI V1.0",
    version="1.0.0",
    description="One chat endpoint for applications, games, websites, bots and terminal clients.",
)


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
        return self.dropout(self.proj(torch.cat([head(x) for head in self.heads], dim=-1)))


class Block(nn.Module):
    def __init__(self, embed: int, heads: int, block_size: int):
        super().__init__()
        self.attn = MultiHeadAttention(embed, heads, block_size)
        self.ff = nn.Sequential(
            nn.Linear(embed, 4 * embed), nn.ReLU(),
            nn.Linear(4 * embed, embed), nn.Dropout(0.0),
        )
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
        positions = torch.arange(t, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        return self.head(self.norm(self.blocks(x)))


def _load_model():
    global _model, _vocab, _stoi, _block_size, _model_error
    if not CHECKPOINT.is_file():
        raise RuntimeError(
            f"Brak checkpointu: {CHECKPOINT}. Najpierw uruchom trening i zapisz wagi modelu."
        )
    try:
        saved = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    except TypeError:
        saved = torch.load(CHECKPOINT, map_location=DEVICE)

    if not isinstance(saved, dict):
        raise RuntimeError("Checkpoint ma nieprawidłowy format.")
    config = saved.get("config", {})
    vocab = saved.get("vocab", [])
    state_dict = saved.get("model")
    required = {"block_size", "embed", "heads", "layers"}
    if not vocab or not isinstance(config, dict) or not required.issubset(config) or not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint nie zawiera poprawnych wag, słownika i konfiguracji.")

    block_size = int(config["block_size"])
    model = PurpleGPT(
        len(vocab), block_size, int(config["embed"]),
        int(config["heads"]), int(config["layers"]),
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    _model, _vocab = model, vocab
    _stoi = {char: index for index, char in enumerate(vocab)}
    _block_size = block_size
    _model_error = None


@app.on_event("startup")
def startup():
    global _model_error
    try:
        _load_model()
    except Exception as exc:
        _model_error = str(exc)
        print(f"Purple AI model unavailable: {_model_error}")


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    web_search: bool = False
    max_new_chars: int = Field(default=300, ge=1, le=1000)
    temperature: float = Field(default=0.8, gt=0.05, le=2.0)


def _authorize(authorization: Optional[str]):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API key is not configured on the server.")
    expected = "Bearer " + API_KEY
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


def _web_search(query: str, max_results: int = 5) -> list[dict]:
    search_key = os.getenv("SEARCH_API_KEY", "").strip()
    if not search_key:
        raise HTTPException(status_code=503, detail="Wyszukiwanie nie jest skonfigurowane na serwerze.")
    payload = json.dumps({
        "api_key": search_key,
        "query": query[:300],
        "topic": "general",
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }).encode("utf-8")
    request = Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail="Wyszukiwanie internetowe chwilowo nie działa.") from exc

    results = []
    for item in data.get("results", []):
        if isinstance(item, dict) and item.get("url"):
            results.append({
                "title": str(item.get("title") or "Wynik")[:200],
                "url": str(item["url"])[:1500],
                "snippet": str(item.get("content") or "")[:700],
            })
    return results


def _format_prompt(messages: list[ChatMessage], search_results: list[dict]) -> str:
    # Keep the public interface conversation-based. This small prototype still sees
    # characters, not semantic tokens, and therefore only the tail fits its context.
    parts = ["Purple AI conversation:\n"]
    for message in messages:
        parts.append(f"{message.role.upper()}: {message.content}\n")
    if search_results:
        parts.append("WEB SEARCH RESULTS (reference material):\n")
        for item in search_results:
            parts.append(f"{item['title']}\n{item['snippet']}\n{item['url']}\n")
    parts.append("ASSISTANT:")
    return "".join(parts)


def _generate(prompt: str, max_new_chars: int, temperature: float) -> str:
    if _model is None:
        detail = "Model nie jest załadowany."
        if _model_error:
            detail += " " + _model_error
        raise HTTPException(status_code=503, detail=detail)

    ids = [_stoi[char] for char in prompt if char in _stoi]
    if not ids:
        raise HTTPException(status_code=400, detail="Wiadomość nie zawiera znaków znanych modelowi.")
    context = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    generated: list[int] = []
    with torch.inference_mode():
        for _ in range(max_new_chars):
            logits = _model(context[:, -_block_size:])[:, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probabilities, num_samples=1).item())
            generated.append(next_id)
            context = torch.cat(
                [context, torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)],
                dim=1,
            )
    return "".join(_vocab[index] for index in generated)


@app.get("/")
def root():
    return {
        "name": "Purple AI V1.0",
        "api": "POST /v1/chat",
        "health": "/health",
        "model_loaded": _model is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "ready" if _model is not None else "model_unavailable",
        "model_loaded": _model is not None,
        "device": DEVICE,
        "checkpoint_exists": CHECKPOINT.is_file(),
        "web_search_configured": bool(os.getenv("SEARCH_API_KEY")),
        "model_error": _model_error if _model is None else None,
    }


@app.post("/v1/chat")
def chat(body: ChatRequest, authorization: Optional[str] = Header(default=None)):
    """The single public conversation endpoint used by all Purple AI clients."""
    _authorize(authorization)
    search_results: list[dict] = []
    if body.web_search:
        last_user = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
        if not last_user:
            raise HTTPException(status_code=400, detail="Do wyszukiwania potrzebna jest wiadomość użytkownika.")
        search_results = _web_search(last_user)

    prompt = _format_prompt(body.messages, search_results)
    answer = _generate(prompt, body.max_new_chars, body.temperature)
    return {
        "id": "purple-chat",
        "model": "purple-ai-v1.0",
        "answer": answer,
        "sources": search_results,
        "usage": {"generated_characters": len(answer)},
    }
