"""Train Purple AI Image from paired images and Polish/English captions."""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model import PurpleImageNet, make_noise_schedule


class CaptionImageDataset(Dataset):
    def __init__(self, root, size):
        self.root = Path(root)
        self.size = size
        index = self.root / "captions.jsonl"
        if not index.exists():
            raise FileNotFoundError(f"Missing {index}. See README.md for format.")
        self.items = []
        for line in index.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            image_path = self.root / item["image"]
            if image_path.is_file() and str(item.get("caption", "")).strip():
                self.items.append((image_path, str(item["caption"])))
        if not self.items:
            raise ValueError("No valid image/caption pairs found.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, caption = self.items[index]
        with Image.open(path) as source:
            image = source.convert("RGB").resize((self.size, self.size), Image.Resampling.LANCZOS)
        # Convert image bytes to a normalized CHW tensor without torchvision.
        import numpy as np
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="checkpoints/purple_image.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if args.size != 64:
        raise ValueError("This first prototype is designed for --size 64.")
    random.seed(321)
    torch.manual_seed(321)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CaptionImageDataset(args.data, args.size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))
    model = PurpleImageNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    betas, alphas, alpha_bars = make_noise_schedule(args.steps, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print(f"Purple AI Image | device={device} | pairs={len(dataset)} | size={args.size}")
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for step, (clean, captions) in enumerate(loader, start=1):
            clean = clean.to(device, non_blocking=True)
            tokens = model.text_encoder.tokenize(captions, device)
            # Classifier-free conditioning dropout is trained into the same model.
            if random.random() < 0.10:
                tokens = model.text_encoder.tokenize([""] * len(captions), device)
            t = torch.randint(0, args.steps, (clean.shape[0],), device=device)
            noise = torch.randn_like(clean)
            a_bar = alpha_bars[t].view(-1, 1, 1, 1)
            noisy = a_bar.sqrt() * clean + (1.0 - a_bar).sqrt() * noise
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                prediction = model(noisy, t, tokens)
                loss = F.mse_loss(prediction.float(), noise.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            if step % 25 == 0:
                print(f"epoch {epoch + 1}/{args.epochs} step {step}/{len(loader)} loss={loss.item():.5f}")
        print(f"epoch {epoch + 1} average_loss={total / max(len(loader), 1):.5f}")
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "size": args.size,
                    "steps": args.steps, "base_channels": 32}, target)
        print(f"Checkpoint saved: {target}")


if __name__ == "__main__":
    main()
