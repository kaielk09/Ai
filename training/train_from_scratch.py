"""
Purple AI — minimalny model znakowy trenowany od zera.
To prototyp edukacyjny, nie odpowiednik dużego asystenta.
Wymaga: pip install torch
Dane: training/data.txt (zwykły tekst, lokalnie; nie commituj go).
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
BLOCK_SIZE = int(os.getenv("PURPLE_BLOCK_SIZE", "128"))
STEPS = int(os.getenv("PURPLE_STEPS", "2000"))
EVAL_EVERY = int(os.getenv("PURPLE_EVAL_EVERY", "100"))
LR = float(os.getenv("PURPLE_LR", "0.0003"))
EMBED = int(os.getenv("PURPLE_EMBED", "128"))
HEADS = int(os.getenv("PURPLE_HEADS", "4"))
LAYERS = int(os.getenv("PURPLE_LAYERS", "4"))
DROPOUT = 0.1
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
        weights = self.dropout(F.softmax(weights, dim=-1))
        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        size = EMBED // HEADS
        self.heads = nn.ModuleList([AttentionHead(size) for _ in range(HEADS)])
        self.proj = nn.Linear(EMBED, EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ff = nn.Sequential(
            nn.Linear(EMBED, 4 * EMBED), nn.ReLU(),
            nn.Linear(4 * EMBED, EMBED), nn.Dropout(DROPOUT)
        )
        self.ln1 = nn.LayerNorm(EMBED)
        self.ln2 = nn.LayerNorm(EMBED)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class PurpleGPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBED)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED)
        self.blocks = nn.Sequential(*[Block() for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(EMBED)
        self.head = nn.Linear(EMBED, vocab_size)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(
            torch.arange(t, device=idx.device)
        )
        logits = self.head(self.norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def main():
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Brak pliku {DATA_PATH}. Utwórz training/data.txt i dodaj własny tekst "
            "do nauki. Nie dodawaj prywatnych danych."
        )
    text = DATA_PATH.read_text(encoding="utf-8")
    if len(text) < BLOCK_SIZE + 2:
        raise SystemExit(f"Dane są za krótkie. Dodaj co najmniej {BLOCK_SIZE + 2} znaków.")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = max(BLOCK_SIZE + 1, int(len(encoded) * 0.9))
    train_data = encoded[:split]
    val_data = encoded[split:] if len(encoded) - split > BLOCK_SIZE else encoded[:split]

    def batch(source):
        ix = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
        x = torch.stack([source[i:i + BLOCK_SIZE] for i in ix]).to(DEVICE)
        y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in ix]).to(DEVICE)
        return x, y

    model = PurpleGPT(len(chars)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    start_step = 0

    if CKPT.exists():
        try:
            saved = torch.load(CKPT, map_location=DEVICE, weights_only=False)
            if saved.get("vocab") == chars and saved.get("config") == {
                "block_size": BLOCK_SIZE, "embed": EMBED, "heads": HEADS, "layers": LAYERS
            }:
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                start_step = int(saved.get("step", 0))
                print(f"Wznawiam trening od kroku {start_step}.")
            else:
                print("Checkpoint nie pasuje do danych/ustawień — zaczynam nowy trening.")
        except Exception as exc:
            print(f"Nie udało się odczytać checkpointu ({exc}); zaczynam od zera.")

    config = {"block_size": BLOCK_SIZE, "embed": EMBED, "heads": HEADS, "layers": LAYERS}
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
            print(f"krok {step + 1}/{STEPS} | loss={loss.item():.4f} | val_loss={val_loss.item():.4f}")
            model.train()
            torch.save({
                "step": step + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "vocab": chars,
                "config": config,
                "loss": float(loss.item()),
            }, CKPT)
            (OUT_DIR / "vocab.json").write_text(
                json.dumps(chars, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Checkpoint zapisany: {CKPT}")

    model.eval()
    prompt = text[:min(40, len(text))]
    context = torch.tensor([[stoi[c] for c in prompt]], device=DEVICE)
    with torch.no_grad():
        for _ in range(300):
            context_cond = context[:, -BLOCK_SIZE:]
            logits, _ = model(context_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_id), dim=1)
    generated = "".join(itos[int(i)] for i in context[0].tolist())
    print("\n--- Przykładowy tekst (prototyp) ---\n", generated)


if __name__ == "__main__":
    main()
