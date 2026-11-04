"""
Purple AI V1.0 — własny, bajtowy model językowy trenowany od losowych wag.
Tokenizer UTF-8: każdy bajt ma własny identyfikator 0..255.
Dane: training/data.txt (lokalnie; nie commituj prywatnych danych).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("PURPLE_DATA", ROOT / "training" / "data.txt"))
OUT_DIR = Path(os.getenv("PURPLE_CHECKPOINT_DIR", ROOT / "checkpoints"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = int(os.getenv("PURPLE_BATCH_SIZE", "16"))
BLOCK_SIZE = int(os.getenv("PURPLE_BLOCK_SIZE", "512"))
STEPS = int(os.getenv("PURPLE_STEPS", "5000"))
EVAL_EVERY = max(1, int(os.getenv("PURPLE_EVAL_EVERY", "100")))
LR = float(os.getenv("PURPLE_LR", "0.0003"))
EMBED = int(os.getenv("PURPLE_EMBED", "192"))
HEADS = int(os.getenv("PURPLE_HEADS", "4"))
LAYERS = int(os.getenv("PURPLE_LAYERS", "4"))
DROPOUT = float(os.getenv("PURPLE_DROPOUT", "0.1"))
VOCAB_SIZE = 256
TOKENIZER = "utf8-bytes-v1"
CKPT = OUT_DIR / "purple_ai_latest.pt"


class AttentionHead(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(EMBED, head_size, bias=False)
        self.query = nn.Linear(EMBED, head_size, bias=False)
        self.value = nn.Linear(EMBED, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        _, t, _ = x.shape
        k, q = self.key(x), self.query(x)
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.mask[:t, :t] == 0, float("-inf"))
        return self.dropout(F.softmax(weights, dim=-1)) @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        size = EMBED // HEADS
        self.heads = nn.ModuleList([AttentionHead(size) for _ in range(HEADS)])
        self.proj = nn.Linear(EMBED, EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([head(x) for head in self.heads], dim=-1)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ff = nn.Sequential(
            nn.Linear(EMBED, 4 * EMBED), nn.GELU(),
            nn.Linear(4 * EMBED, EMBED), nn.Dropout(DROPOUT),
        )
        self.ln1 = nn.LayerNorm(EMBED)
        self.ln2 = nn.LayerNorm(EMBED)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class PurpleGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED)
        self.blocks = nn.Sequential(*[Block() for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(EMBED)
        self.head = nn.Linear(EMBED, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        _, length = idx.shape
        if length > BLOCK_SIZE:
            raise ValueError(f"Wejście ma {length} tokenów, limit to {BLOCK_SIZE}.")
        positions = torch.arange(length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        logits = self.head(self.norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        return logits, loss


def config():
    return {
        "tokenizer": TOKENIZER,
        "vocab_size": VOCAB_SIZE,
        "block_size": BLOCK_SIZE,
        "embed": EMBED,
        "heads": HEADS,
        "layers": LAYERS,
    }


def main():
    if BLOCK_SIZE < 32:
        raise SystemExit("PURPLE_BLOCK_SIZE musi wynosić co najmniej 32.")
    if EMBED % HEADS != 0:
        raise SystemExit("PURPLE_EMBED musi być podzielne przez PURPLE_HEADS.")
    if BATCH_SIZE < 1 or STEPS < 1:
        raise SystemExit("PURPLE_BATCH_SIZE i PURPLE_STEPS muszą być dodatnie.")
    if not DATA_PATH.is_file():
        raise SystemExit(f"Brak pliku {DATA_PATH}. Utwórz training/data.txt z legalnymi danymi.")
    raw_text = DATA_PATH.read_text(encoding="utf-8")
    encoded = torch.tensor(list(raw_text.encode("utf-8")), dtype=torch.long)
    if len(encoded) < BLOCK_SIZE + 2:
        raise SystemExit(
            f"Dane są za krótkie: potrzeba co najmniej {BLOCK_SIZE + 2} bajtów UTF-8."
        )

    split = min(max(BLOCK_SIZE + 1, int(len(encoded) * 0.9)), len(encoded) - 1)
    train_data = encoded[:split]
    val_data = encoded[split:]
    if len(val_data) <= BLOCK_SIZE:
        val_data = train_data

    def batch(source):
        if len(source) <= BLOCK_SIZE:
            raise RuntimeError("Za mało danych do utworzenia partii treningowej.")
        starts = torch.randint(0, len(source) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([source[i:i + BLOCK_SIZE] for i in starts]).to(DEVICE)
        y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in starts]).to(DEVICE)
        return x, y

    model = PurpleGPT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    start_step = 0
    expected_config = config()

    if CKPT.is_file():
        try:
            saved = torch.load(CKPT, map_location=DEVICE, weights_only=True)
            if saved.get("config") == expected_config:
                model.load_state_dict(saved["model"])
                if "optimizer" in saved:
                    optimizer.load_state_dict(saved["optimizer"])
                start_step = int(saved.get("step", 0))
                print(f"Wznawiam zgodny trening od kroku {start_step}.")
            else:
                print("Checkpoint ma inny format lub ustawienia — rozpoczynam nowy trening.")
        except Exception as exc:
            print(f"Nie można wznowić checkpointu ({exc}); rozpoczynam nowy trening.")

    if start_step >= STEPS:
        print(f"Checkpoint ma już {start_step} kroków. Ustaw PURPLE_STEPS na wartość większą.")
        return

    model.train()
    for step in range(start_step, STEPS):
        xb, yb = batch(train_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = batch(val_data)
                _, val_loss = model(vx, vy)
            print(
                f"krok {step + 1}/{STEPS} | "
                f"loss={loss.item():.4f} | val_loss={val_loss.item():.4f} | device={DEVICE}"
            )
            torch.save({
                "step": step + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": expected_config,
                "loss": float(loss.item()),
                "validation_loss": float(val_loss.item()),
            }, CKPT)
            (OUT_DIR / "tokenizer.json").write_text(
                json.dumps({"type": TOKENIZER, "vocab_size": VOCAB_SIZE}),
                encoding="utf-8",
            )
            print(f"Checkpoint zapisany: {CKPT}")
            model.train()

    print("Trening zakończony. Wagi i konfiguracja są zapisane w checkpointcie.")


if __name__ == "__main__":
    main()
