"""Train Purple AI's image head while preserving the shared text backbone.

Set PURPLE_CHECKPOINT_PATH to the existing Purple AI chat checkpoint.
The text backbone is loaded from that checkpoint and frozen; only the image
generation layers are optimized. This prevents damaging chat/programming skills.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader

from purple_image.model import PurpleImageNet, make_noise_schedule
import api.app as purple_api


class PairedImages(Dataset):
    def __init__(self, root, size):
        self.root = Path(root)
        self.size = int(size)
        manifest = self.root / "captions.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Brak {manifest}. Wymagany format JSONL: image + caption.")
        self.rows = []
        for n, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rel = Path(row["image"])
                caption = str(row["caption"]).strip()
                if rel.is_absolute() or ".." in rel.parts or not caption:
                    raise ValueError("nieprawidłowa ścieżka lub pusty opis")
                path = self.root / rel
                if path.is_file():
                    self.rows.append((path, caption))
                else:
                    print(f"Pominięto brakujący obraz: {path}")
            except Exception as exc:
                print(f"Pominięto wiersz {n}: {exc}")
        if len(self.rows) < 2:
            raise ValueError("Potrzebne są co najmniej 2 pary obraz-opis.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, caption = self.rows[i]
        with Image.open(path) as im:
            im = ImageOps.fit(im.convert("RGB"), (self.size, self.size), method=Image.Resampling.LANCZOS)
            if random.random() < 0.5:
                im = ImageOps.mirror(im)
            arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), caption


class EMA:
    def __init__(self, model):
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.n = 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        decay = min(0.999, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if self.shadow[k].is_floating_point():
                self.shadow[k].lerp_(v.detach(), 1 - decay)
            else:
                self.shadow[k].copy_(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="purple_image/data")
    p.add_argument("--output", default="checkpoints/purple_image.pt")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--diffusion-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.size < 16 or args.size > 512 or args.size % 4:
        raise ValueError("--size musi być wielokrotnością 4 w zakresie 16..512.")
    if args.epochs < 1 or args.batch_size != 1:
        raise ValueError("Ten bezpieczny skrypt wymaga epochs>=1 i batch-size=1.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the exact text backbone used by the API. Never initialize a second encoder.
    purple_api._load_model()
    text_model = purple_api._model
    if text_model is None:
        raise RuntimeError(f"Nie udało się załadować wspólnego modelu tekstowego: {purple_api._model_error}")
    text_model.eval()
    for parameter in text_model.parameters():
        parameter.requires_grad_(False)
    text_width = int(purple_api._config["embed"])

    dataset = PairedImages(args.data, args.size)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    model = PurpleImageNet(text_width=text_width, base=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = EMA(model)
    start_epoch = 0
    global_step = 0
    output = Path(args.output)
    if args.resume:
        if not output.is_file():
            raise FileNotFoundError(f"Nie znaleziono checkpointu do wznowienia: {output}")
        ckpt = torch.load(output, map_location=device, weights_only=True)
        if ckpt.get("architecture") != "purple-shared-core-v2" or int(ckpt["text_width"]) != text_width:
            raise RuntimeError("Checkpoint obrazu nie pasuje do architektury lub wymiaru tekstowego.")
        model.load_state_dict(ckpt["model"])
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        ema.n = int(ckpt.get("ema_updates", 0))
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))

    _, _, alpha_bar = make_noise_schedule(args.diffusion_steps, device)
    print(f"Purple AI shared image head | device={device} | pairs={len(dataset)} | size={args.size}")
    print(f"Shared text backbone frozen: width={text_width}; chat/code weights will not be updated.")
    model.train()
    for epoch in range(start_epoch, args.epochs):
        losses = []
        for image, caption in loader:
            image = image.to(device)
            ids = purple_api._encode(caption[0])[-purple_api._block_size:]
            if not ids:
                continue
            token_ids = torch.tensor([ids], dtype=torch.long, device=purple_api.DEVICE)
            with torch.inference_mode():
                features = text_model.encode(token_ids).to(device)
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
            t = torch.randint(0, args.diffusion_steps, (1,), device=device)
            noise = torch.randn_like(image)
            a = alpha_bar[t].view(1, 1, 1, 1)
            noisy = a.sqrt() * image + (1 - a).sqrt() * noise
            optimizer.zero_grad(set_to_none=True)
            predicted = model(noisy, t, features, mask)
            loss = F.mse_loss(predicted.float(), noise.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            losses.append(float(loss.detach()))
            global_step += 1
            if global_step % 20 == 0:
                print(f"epoch={epoch+1}/{args.epochs} step={global_step} loss={losses[-1]:.5f}")
        if not losses:
            raise RuntimeError("Brak poprawnych kroków treningowych.")
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        torch.save({
            "architecture": "purple-shared-core-v2",
            "model": model.state_dict(),
            "ema": ema.shadow,
            "ema_updates": ema.n,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "size": args.size,
            "width": args.size,
            "height": args.size,
            "diffusion_steps": args.diffusion_steps,
            "text_width": text_width,
            "base_channels": args.base_channels,
            "text_checkpoint": str(purple_api.CHECKPOINT),
        }, tmp)
        tmp.replace(output)
        print(f"Epoch {epoch+1}: mean_loss={sum(losses)/len(losses):.5f}")
        print(f"Checkpoint zapisany: {output}")


if __name__ == "__main__":
    main()
