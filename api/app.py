"""Purple AI V1.0 — single public chat API backed by its own UTF-8 byte model."""
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
_model = None
_model_error: Optional[str] = "Model nie został jeszcze załadowany."
_block_size = 512
_embed = 192
_heads = 4
_layers = 4
VOCAB_SIZE = 256
TOKENIZER = "utf8-bytes-v1"

app = FastAPI(
    title="Purple AI V1.0",
    version="1.1.0",
    description="One conversation endpoint for every Purple AI client.",
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
        _, length, _ = x.shape
        k, q = self.key(x), self.query(x)
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.mask[:length, :length] == 0, float("-inf"))
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
            nn.Linear(embed, 4 * embed), nn.GELU(),
            nn.Linear(4 * embed, embed), nn.Dropout(0.0),
        )
        self.ln1 = nn.LayerNorm(embed)
        self.ln2 = nn.LayerNorm(embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class PurpleGPT(nn.Module):
    def __init__(self, block_size: int, embed: int, heads: int, layers: int):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, embed)
        self.position_embedding = nn.Embedding(block_size, embed)
        self.blocks = nn.Sequential(*[Block(embed, heads, block_size) for _ in range(layers)])
        self.norm = nn.LayerNorm(embed)
        self.head = nn.Linear(embed, VOCAB_SIZE)

    def forward(self, idx):
        _, length = idx.shape
        positions = torch.arange(length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        return self.head(self.norm(self.blocks(x)))


def _load_model():
    global _model, _model_error, _block_size, _embed, _heads, _layers
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"Nie znaleziono checkpointu: {CHECKPOINT}. Najpierw wytrenuj model.")
    try:
        saved = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    except TypeError:
        saved = torch.load(CHECKPOINT, map_location=DEVICE)
    if not isinstance(saved, dict):
        raise RuntimeError("Checkpoint ma nieprawidłowy format.")
    config = saved.get("config", {})
    required = {"tokenizer", "vocab_size", "block_size", "embed", "heads", "layers"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise RuntimeError("Checkpoint nie zawiera konfiguracji nowego tokenizera UTF-8.")
    if config["tokenizer"] != TOKENIZER or int(config["vocab_size"]) != VOCAB_SIZE:
        raise RuntimeError("To stary lub nieobsługiwany format checkpointu. Uruchom nowy trening.")
    _block_size = int(config["block_size"])
    _embed = int(config["embed"])
    _heads = int(config["heads"])
    _layers = int(config["layers"])
    if _block_size < 2 or _embed % _heads != 0:
        raise RuntimeError("Nieprawidłowe ustawienia modelu w checkpointcie.")
    model = PurpleGPT(_block_size, _embed, _heads, _layers).to(DEVICE)
    model.load_state_dict(saved["model"])
    model.eval()
    _model = model
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
    max_new_tokens: int = Field(default=160, ge=1, le=512)
    temperature: float = Field(default=0.8, gt=0.05, le=2.0)


def _authorize(authorization: Optional[str]):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="Klucz API nie jest skonfigurowany na serwerze.")
    expected = "Bearer " + API_KEY
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Brak lub nieprawidłowy klucz API.")


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
    parts = ["PURPLE AI CONVERSATION\n"]
    # Put retrieved material first so truncating to the model context keeps
    # the newest user message and the assistant response marker.
    if search_results:
        parts.append("SEARCH RESULTS (untrusted reference text):\n")
        for item in search_results:
            parts.append(f"{item['title']}\n{item['snippet']}\n{item['url']}\n")
    for message in messages:
        parts.append(f"{message.role.upper()}: {message.content}\n")
    parts.append("ASSISTANT:")
    return "".join(parts)


def _generate(prompt: str, max_new_tokens: int, temperature: float) -> str:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model nie jest załadowany. Sprawdź /health.")
    encoded = prompt.encode("utf-8")
    # Keep the newest context, then decode only complete UTF-8 sequences for display.
    token_ids = list(encoded[-_block_size:])
    if not token_ids:
        raise HTTPException(status_code=400, detail="Pusta wiadomość.")
    context = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    generated: list[int] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = _model(context[:, -_block_size:])[:, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probabilities, num_samples=1).item())
            generated.append(next_id)
            context = torch.cat(
                [context, torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)],
                dim=1,
            )
    return bytes(generated).decode("utf-8", errors="replace")


@app.get("/")
def root():
    return {"name": "Purple AI V1.0", "chat_endpoint": "POST /v1/chat", "health": "/health"}


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
    """One public endpoint for all Purple AI clients and capabilities."""
    _authorize(authorization)
    sources: list[dict] = []
    if body.web_search:
        last_user = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
        if not last_user:
            raise HTTPException(status_code=400, detail="Wiadomość użytkownika jest wymagana do wyszukiwania.")
        sources = _web_search(last_user)
    prompt = _format_prompt(body.messages, sources)
    answer = _generate(prompt, body.max_new_tokens, body.temperature)
    return {
        "id": "purple-chat",
        "model": "purple-ai-v1.0",
        "answer": answer,
        "sources": sources,
        "usage": {"generated_bytes": len(answer.encode("utf-8"))},
    }
