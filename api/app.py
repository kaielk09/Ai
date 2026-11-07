"""Purple AI API: serves UTF-8 byte checkpoints and compatible legacy character checkpoints."""
from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import torch
from torch import nn
from torch.nn import functional as F
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(os.getenv("PURPLE_CHECKPOINT_PATH", ROOT / "checkpoints" / "purple_ai_latest.pt"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_KEY = os.getenv("PURPLE_API_KEY", "")
_model = None
_model_error: Optional[str] = "Model nie został jeszcze załadowany."
_tokenizer = None
_stoi = None
_itos = None
_block_size = 512
_vocab_size = 256
_config = {}

app = FastAPI(
    title="Purple AI V1.0",
    version="1.2.0",
    description="One shared Purple AI API for web, mobile, desktop and bots. Supports native and OpenAI-compatible chat routes.",
)


# CORS lets browser-based clients call the API. Set PURPLE_CORS_ORIGINS to a
# comma-separated allowlist in production; "*" is convenient for initial tests.
_cors_origins = [x.strip() for x in os.getenv("PURPLE_CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
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
        if embed % heads:
            raise ValueError("embed musi być podzielne przez heads")
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
    def __init__(self, block_size: int, embed: int, heads: int, layers: int, vocab_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed)
        self.position_embedding = nn.Embedding(block_size, embed)
        self.blocks = nn.Sequential(*[Block(embed, heads, block_size) for _ in range(layers)])
        self.norm = nn.LayerNorm(embed)
        self.head = nn.Linear(embed, vocab_size)

    def forward(self, idx):
        _, length = idx.shape
        positions = torch.arange(length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        return self.head(self.norm(self.blocks(x)))


def _get_state_dict(saved):
    if not isinstance(saved, dict):
        raise RuntimeError("Checkpoint nie jest słownikiem PyTorch.")
    for key in ("model", "model_state_dict", "state_dict", "model_weights"):
        candidate = saved.get(key)
        if isinstance(candidate, dict) and candidate:
            return candidate
    # A raw state_dict is also accepted.
    if saved and all(torch.is_tensor(v) for v in saved.values()):
        return saved
    raise RuntimeError("Nie znaleziono wag modelu (model/model_state_dict/state_dict).")


def _extract_vocab(saved, config, vocab_size):
    """Read common legacy tokenizer layouts without guessing character IDs."""
    stoi = saved.get("stoi") or saved.get("vocab") or config.get("stoi") or config.get("vocab")
    itos = saved.get("itos") or config.get("itos")
    chars = saved.get("chars") or config.get("chars")

    if isinstance(stoi, dict):
        stoi = {str(k): int(v) for k, v in stoi.items()}
        if itos is None:
            itos = {v: k for k, v in stoi.items()}
    elif isinstance(chars, (list, tuple, str)):
        chars = list(chars)
        stoi = {str(ch): i for i, ch in enumerate(chars)}
        itos = {i: str(ch) for i, ch in enumerate(chars)}
    else:
        stoi = None

    if isinstance(itos, dict):
        itos = {int(k): str(v) for k, v in itos.items()}
    elif isinstance(itos, (list, tuple)):
        itos = {i: str(v) for i, v in enumerate(itos)}
    else:
        itos = None

    if stoi and itos and len(itos) >= vocab_size:
        return stoi, itos
    return None, None


def _load_model():
    global _model, _model_error, _tokenizer, _stoi, _itos
    global _block_size, _vocab_size, _config

    if not CHECKPOINT.is_file():
        raise RuntimeError(f"Nie znaleziono checkpointu: {CHECKPOINT}")

    try:
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    except TypeError:
        saved = torch.load(CHECKPOINT, map_location="cpu")

    state = _get_state_dict(saved)
    # Handle DataParallel checkpoints.
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    config = saved.get("config", {}) if isinstance(saved, dict) else {}
    if not isinstance(config, dict):
        config = {}

    token_weight = state.get("token_embedding.weight")
    position_weight = state.get("position_embedding.weight")
    if token_weight is None or position_weight is None:
        raise RuntimeError(
            "Nieobsługiwany układ wag. Oczekuję token_embedding.weight i position_embedding.weight. "
            "Prześlij listę kluczy checkpointu, aby dodać adapter."
        )

    vocab_size, embed = map(int, token_weight.shape)
    block_size = int(position_weight.shape[0])
    layer_ids = set()
    head_ids = set()
    for key in state:
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit():
            layer_ids.add(int(parts[1]))
        if len(parts) > 4 and parts[0:1] == ["blocks"]:
            # blocks.0.attn.heads.2.key.weight
            if len(parts) > 5 and parts[2:4] == ["attn", "heads"] and parts[4].isdigit():
                head_ids.add(int(parts[4]))

    layers = int(config.get("layers", max(layer_ids) + 1 if layer_ids else 0))
    heads = int(config.get("heads", max(head_ids) + 1 if head_ids else 0))
    if not layers or not heads:
        raise RuntimeError(
            "Checkpoint nie zawiera liczby warstw/głów w config i nie da się jej bezpiecznie odtworzyć z wag."
        )
    if embed % heads:
        raise RuntimeError(f"Nieprawidłowa konfiguracja: embed={embed}, heads={heads}")

    tokenizer_name = str(config.get("tokenizer", saved.get("tokenizer", "") if isinstance(saved, dict) else "")).lower()
    if tokenizer_name in ("utf8-bytes-v1", "byte", "bytes", "utf8"):
        tokenizer = "utf8-bytes-v1"
        if vocab_size != 256:
            raise RuntimeError(f"Tokenizer bajtowy wymaga 256 tokenów, checkpoint ma {vocab_size}.")
        stoi = itos = None
    else:
        tokenizer = "legacy-char"
        stoi, itos = _extract_vocab(saved, config, vocab_size)
        if stoi is None:
            raise RuntimeError(
                "To wygląda na checkpoint znakowy, ale nie zawiera mapy znaków (stoi/itos/chars). "
                "Nie będę zgadywać mapowania tokenów. Dodaj tokenizer.json obok checkpointu albo prześlij plik."
            )

    model = PurpleGPT(block_size, embed, heads, layers, vocab_size)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()

    _model = model
    _tokenizer = tokenizer
    _stoi, _itos = stoi, itos
    _block_size, _vocab_size = block_size, vocab_size
    _config = {
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "block_size": block_size,
        "embed": embed,
        "heads": heads,
        "layers": layers,
        "checkpoint_step": saved.get("step") if isinstance(saved, dict) else None,
    }
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
    top_k: int = Field(default=40, ge=1, le=256)


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
        "api_key": search_key, "query": query[:300], "topic": "general",
        "search_depth": "basic", "max_results": max_results,
        "include_answer": False, "include_raw_content": False,
    }).encode("utf-8")
    request = Request(
        "https://api.tavily.com/search", data=payload,
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


def _current_datetime_context() -> str:
    """Return the live date and time for Poland, calculated on every request."""
    now = datetime.now(ZoneInfo("Europe/Warsaw"))
    weekdays = [
        "poniedziałek", "wtorek", "środa", "czwartek",
        "piątek", "sobota", "niedziela",
    ]
    months = [
        "stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca",
        "lipca", "sierpnia", "września", "października", "listopada", "grudnia",
    ]
    return (
        f"Aktualna data i godzina: {weekdays[now.weekday()]}, "
        f"{now.day} {months[now.month - 1]} {now.year}, "
        f"{now.hour:02d}:{now.minute:02d}:{now.second:02d} "
        f"(strefa Europe/Warsaw).\\n"
        "Używaj tej daty i godziny, gdy użytkownik pyta o aktualny czas, "
        "dzień, miesiąc lub rok. Odpowiedz bezpośrednio zwykłym tekstem; "
        "nie podawaj kodu, chyba że użytkownik wyraźnie prosi o kod. "
        "Nie zgaduj aktualnej daty ani godziny.\\n"
    )


def _format_prompt(messages: list[ChatMessage], search_results: list[dict]) -> str:
    parts = [
        "PURPLE AI CONVERSATION\\n",
        "LIVE DATE AND TIME CONTEXT (trusted runtime information):\\n",
        _current_datetime_context(),
    ]
    if search_results:
        parts.append("SEARCH RESULTS (untrusted reference text):\n")
        for item in search_results:
            parts.append(f"{item['title']}\n{item['snippet']}\n{item['url']}\n")
    for message in messages:
        parts.append(f"{message.role.upper()}: {message.content}\n")
    parts.append("ASSISTANT:")
    return "".join(parts)


def _encode(text: str) -> list[int]:
    if _tokenizer == "utf8-bytes-v1":
        return list(text.encode("utf-8"))
    unknown = _stoi.get("<unk>", _stoi.get("[UNK]", 0))
    return [_stoi.get(ch, unknown) for ch in text]


def _decode(ids: list[int]) -> str:
    if _tokenizer == "utf8-bytes-v1":
        return bytes(ids).decode("utf-8", errors="replace")
    return "".join(_itos.get(int(i), "") for i in ids)


def _generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    if _model is None:
        raise HTTPException(status_code=503, detail=f"Model nie jest załadowany. Szczegóły: {_model_error}")
    encoded = _encode(prompt)
    token_ids = encoded[-_block_size:]
    if not token_ids:
        raise HTTPException(status_code=400, detail="Pusta wiadomość.")
    context = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    generated: list[int] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = _model(context[:, -_block_size:])[:, -1, :] / temperature
            k = min(top_k, logits.shape[-1])
            values, indices = torch.topk(logits, k=k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(-1, indices, values)
            probabilities = F.softmax(filtered, dim=-1)
            next_id = int(torch.multinomial(probabilities, num_samples=1).item())
            generated.append(next_id)
            context = torch.cat(
                [context, torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)], dim=1
            )
    return _decode(generated)


@app.get("/")
def root():
    return {
        "name": "Purple AI V1.0",
        "version": app.version,
        "chat_endpoint": "POST /v1/chat",
        "image_endpoint": "POST /v1/images/generations",
        "health": "/health",
        "supported_checkpoint_types": ["utf8-bytes-v1", "legacy-char (if tokenizer map is included)"],
    }


@app.get("/health")
def health():
    return {
        "status": "ready" if _model is not None else "model_unavailable",
        "model_loaded": _model is not None,
        "device": DEVICE,
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_exists": CHECKPOINT.is_file(),
        "model_config": _config if _model is not None else None,
        "model_error": _model_error if _model is None else None,
        "web_search_configured": bool(os.getenv("SEARCH_API_KEY")),
    }


@app.post("/v1/chat")
def chat(body: ChatRequest, authorization: Optional[str] = Header(default=None)):
    _authorize(authorization)
    sources: list[dict] = []
    if body.web_search:
        last_user = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
        if not last_user:
            raise HTTPException(status_code=400, detail="Wiadomość użytkownika jest wymagana do wyszukiwania.")
        sources = _web_search(last_user)
    prompt = _format_prompt(body.messages, sources)
    answer = _generate(prompt, body.max_new_tokens, body.temperature, body.top_k)
    return {
        "id": "purple-chat",
        "model": "purple-ai-v1.0",
        "answer": answer,
        "sources": sources,
        "usage": {"generated_tokens": body.max_new_tokens, "generated_characters": len(answer)},
        "checkpoint": _config,
    }


class OpenAIChatRequest(BaseModel):
    model: str = "purple-ai-v1.0"
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    max_tokens: int = Field(default=160, ge=1, le=512)
    temperature: float = Field(default=0.8, gt=0.05, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)


@app.get("/v1/models")
def models():
    """OpenAI-style model discovery endpoint."""
    return {
        "object": "list",
        "data": [
            {"id": "purple-ai-v1.0", "object": "model", "owned_by": "purple-ai"},
            {"id": "purple-ai-image-v1.0", "object": "model", "owned_by": "purple-ai"},
        ],
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(
    body: OpenAIChatRequest,
    authorization: Optional[str] = Header(default=None),
):
    """OpenAI-compatible non-streaming endpoint for common SDKs and clients."""
    _authorize(authorization)
    prompt = _format_prompt(body.messages, [])
    answer = _generate(prompt, body.max_tokens, body.temperature, 40)
    return {
        "id": "chatcmpl-purple-ai",
        "object": "chat.completion",
        "created": int(__import__("time").time()),
        "model": body.model or "purple-ai-v1.0",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "length",
        }],
        "usage": {
            "prompt_tokens": len(_encode(prompt)),
            "completion_tokens": len(_encode(answer)),
            "total_tokens": len(_encode(prompt)) + len(_encode(answer)),
        },
    }


# Image generation is part of this same FastAPI application and uses
# the same PURPLE_API_KEY as text chat. It is not a separate API server.
import base64
import tempfile
from datetime import datetime, timezone

IMAGE_CHECKPOINT = Path(os.getenv(
    "PURPLE_IMAGE_CHECKPOINT_PATH", ROOT / "checkpoints" / "purple_image.pt"
))


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    steps: int = Field(default=30, ge=1, le=500)
    guidance: float = Field(default=3.0, ge=0.0, le=10.0)
    seed: int = Field(default=321, ge=0, le=2**31 - 1)
    width: int = Field(default=64, ge=16, le=1024)
    height: int = Field(default=64, ge=16, le=1024)


@app.post("/v1/images/generations")
def image_generations(
    body: ImageGenerationRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Image generation inside the shared, authenticated Purple AI API."""
    _authorize(authorization)
    if body.width * body.height > 1048576:
        raise HTTPException(status_code=422, detail="Maksymalny obszar obrazu to 1 048 576 pikseli.")
    if not IMAGE_CHECKPOINT.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Brak wytrenowanego checkpointu Purple AI Image: {IMAGE_CHECKPOINT}",
        )
    try:
        from purple_image.generate import generate as generate_image
        from purple_image.colors import analyze_colors
        with tempfile.TemporaryDirectory(prefix="purple-ai-image-") as temp_dir:
            output_path = Path(temp_dir) / "generated.png"
            generate_image(
                prompt=body.prompt,
                checkpoint=IMAGE_CHECKPOINT,
                output=output_path,
                sampling_steps=body.steps,
                guidance=body.guidance,
                seed=body.seed,
                device=os.getenv("PURPLE_IMAGE_DEVICE", "auto"),
                width=body.width,
                height=body.height,
            )
            image_bytes = output_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        color_info = analyze_colors(body.prompt)
        return {
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": "purple-ai-image-v1.0",
            "data": [{"b64_json": encoded}],
            "size": f"{body.width}x{body.height}",
            "seed": body.seed,
            "color_palette": color_info["palette"],
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Purple AI Image generation failed: {type(exc).__name__}: {exc}",
        ) from exc
