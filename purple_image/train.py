"""Train the original Purple AI Image diffusion model from random initialization."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader, random_split

from model import PurpleImageNet, make_noise_schedule


class CaptionImageDataset(Dataset):
    def __init__(self, root, size):
        self.root, self.size = Path(root), size
        if isinstance(size, int):
            self.width = self.height = size
        else:
            self.width, self.height = size
        index = self.root / "captions.jsonl"
        if not index.is_file():
            raise FileNotFoundError(f"Missing {index}. Create it using the README format.")
        self.items = []
        for line_no, line in enumerate(index.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                relative = Path(item["image"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("image path must stay inside the data directory")
                image_path = self.root / relative
                caption = str(item["caption"]).strip()
                if not caption:
                    raise ValueError("caption is empty")
                if image_path.is_file():
                    self.items.append((image_path, caption))
                else:
                    print(f"WARNING: line {line_no}: missing image {image_path}")
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                print(f"WARNING: skipping captions.jsonl line {line_no}: {exc}")
        if len(self.items) < 2:
            raise ValueError("At least two valid image/caption pairs are required.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, caption = self.items[index]
        with Image.open(path) as source:
            image = ImageOps.fit(source.convert("RGB"), (self.width, self.height),
                                 method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        # Mild augmentation helps a tiny dataset; captions should describe content,
        # not depend on left/right orientation.
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous(), caption


class ModelEMA:
    """A separate smoothed copy of model weights for more stable sampling."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        # Warm up EMA: fixed 0.999 from step 1 can leave the sampled weights
        # almost identical to random initialization on short runs.
        effective_decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if shadow.is_floating_point():
                shadow.lerp_(value.detach(), 1.0 - effective_decay)
            else:
                shadow.copy_(value)

    def state_dict(self):
        return self.shadow


def save_checkpoint(path, model, ema, optimizer, scaler, epoch, step, args):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save({
        "format_version": 2,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "ema_updates": ema.num_updates,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "size": [args.height, args.width],
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "base_channels": 32,
        "text_width": 256,
    }, temporary)
    temporary.replace(target)


@torch.no_grad()
def evaluate(model, loader, alpha_bars, steps, device):
    model.eval()
    losses = []
    for clean, captions in loader:
        clean = clean.to(device, non_blocking=True)
        tokens = model.text_encoder.tokenize(captions, device)
        t = torch.randint(0, steps, (clean.shape[0],), device=device)
        noise = torch.randn_like(clean)
        a_bar = alpha_bars[t].view(-1, 1, 1, 1)
        noisy = a_bar.sqrt() * clean + (1.0 - a_bar).sqrt() * noise
        prediction = model(noisy, t, tokens)
        losses.append(F.mse_loss(prediction.float(), noise.float()).item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="checkpoints/purple_image.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--size", default="64", help="N or WIDTHxHEIGHT, e.g. 124 or 512x512")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--validation", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()
    try:
        if "x" in str(args.size).lower():
            w, h = str(args.size).lower().split("x", 1)
            args.width, args.height = int(w), int(h)
        else:
            args.width = args.height = int(args.size)
    except (TypeError, ValueError):
        raise ValueError("--size must be N or WIDTHxHEIGHT.")
    if not (16 <= args.width <= 1024 and 16 <= args.height <= 1024):
        raise ValueError("Dimensions must be between 16 and 1024.")
    if args.width * args.height > 1048576:
        raise ValueError("Maximum image area is 1,048,576 pixels.")
    if args.epochs < 1 or args.batch_size < 1 or not 0 <= args.validation < 0.5:
        raise ValueError("Invalid epochs, batch size, or validation fraction.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CaptionImageDataset(args.data, (args.width, args.height))
    val_count = max(1, int(len(dataset) * args.validation)) if args.validation else 0
    train_count = len(dataset) - val_count
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_count, val_count], generator=generator)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda")) if val_count else None

    model = PurpleImageNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                                  weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ema = ModelEMA(model)
    start_epoch, global_step = 0, 0
    output_path = Path(args.output)
    if args.resume:
        if not output_path.is_file():
            raise FileNotFoundError(f"Cannot resume: {output_path} does not exist.")
        saved = torch.load(output_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
        if "scaler" in saved:
            scaler.load_state_dict(saved["scaler"])
        if "ema" in saved:
            ema.shadow = {k: v.to(device) for k, v in saved["ema"].items()}
            ema.num_updates = int(saved.get("ema_updates", saved.get("global_step", 0)))
        start_epoch = int(saved.get("epoch", -1)) + 1
        global_step = int(saved.get("global_step", 0))
        print(f"Resuming at epoch {start_epoch + 1}, global step {global_step}")

    _, _, alpha_bars = make_noise_schedule(args.steps, device)
    print(f"Purple AI Image | device={device} | train={train_count} | val={val_count} | {args.width}x{args.height}")
    model.train()
    for epoch in range(start_epoch, args.epochs):
        running = 0.0
        for batch_index, (clean, captions) in enumerate(loader, 1):
            clean = clean.to(device, non_blocking=True)
            tokens = model.text_encoder.tokenize(captions, device)
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
            ema.update(model)
            running += loss.item()
            global_step += 1
            if batch_index % 25 == 0:
                print(f"epoch {epoch+1}/{args.epochs} batch {batch_index}/{len(loader)} "
                      f"loss={loss.item():.5f} step={global_step}")
        average = running / max(1, len(loader))
        validation = evaluate(model, val_loader, alpha_bars, args.steps, device) if val_loader else None
        print(f"epoch {epoch+1}: train_loss={average:.5f}" +
              (f" val_loss={validation:.5f}" if validation is not None else ""))
        save_checkpoint(output_path, model, ema, optimizer, scaler, epoch, global_step, args)
        print(f"Checkpoint saved: {output_path}")


if __name__ == "__main__":
    main()
