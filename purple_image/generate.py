"""Purple AI Image inference with color hints and flexible canvas sizes."""
import argparse
from pathlib import Path
import torch
from PIL import Image
try:
    from .model import PurpleImageNet, make_noise_schedule
    from .colors import analyze_colors
except ImportError:
    from model import PurpleImageNet, make_noise_schedule
    from colors import analyze_colors

@torch.inference_mode()
def generate(prompt, checkpoint, output, sampling_steps=100, guidance=3.0, seed=321,
             device="auto", width=None, height=None):
    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    if guidance < 0:
        raise ValueError("Guidance must be >= 0.")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    choice = str(device).lower()
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice not in ("cpu", "cuda") or (choice == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device is unavailable or invalid.")
    device = torch.device(choice)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(saved, dict):
        raise ValueError("Invalid checkpoint.")
    model = PurpleImageNet(int(saved.get("base_channels", 32)), int(saved.get("text_width", 256))).to(device)
    weights = saved["ema"] if "ema" in saved else saved.get("model")
    if weights is None:
        raise ValueError("Checkpoint has no model weights.")
    model.load_state_dict(weights, strict=True)
    model.eval()
    total_steps = int(saved.get("steps", 500))
    old_size = saved.get("size", 64)
    if isinstance(old_size, (list, tuple)) and len(old_size) == 2:
        old_h, old_w = map(int, old_size)
    else:
        old_h = old_w = int(old_size)
    width = int(width or saved.get("width", old_w))
    height = int(height or saved.get("height", old_h))
    if not (16 <= width <= 1024 and 16 <= height <= 1024) or width * height > 1048576:
        raise ValueError("Width and height must be 16..1024 and total area <= 1,048,576 pixels.")
    if not 1 <= sampling_steps <= total_steps:
        raise ValueError(f"sampling_steps must be between 1 and {total_steps}.")
    color_info = analyze_colors(prompt)
    # ByteTextEncoder keeps only the first 192 UTF-8 bytes. Put a compact color
    # hint before the user prompt so color guidance is not silently truncated.
    color_hint = ("Colors: " + ", ".join(color_info["colors"][:4]) + ". ") if color_info["colors"] else ""
    model_prompt = color_hint + prompt
    rng = torch.Generator(device=device).manual_seed(int(seed))
    _, _, alpha_bars = make_noise_schedule(total_steps, device)
    tokens = model.text_encoder.tokenize([model_prompt], device)
    empty = model.text_encoder.tokenize([""], device)
    image = torch.randn((1, 3, height, width), device=device, generator=rng)
    sequence = torch.linspace(total_steps - 1, 0, sampling_steps, device=device).long().unique_consecutive()
    for i, step in enumerate(sequence):
        t = torch.full((1,), int(step.item()), device=device, dtype=torch.long)
        cond = model(image, t, tokens)
        if guidance:
            uncond = model(image, t, empty)
            noise = uncond + guidance * (cond - uncond)
        else:
            noise = cond
        alpha = alpha_bars[step]
        prev = alpha_bars[sequence[i + 1]] if i + 1 < len(sequence) else torch.ones((), device=device)
        clean = ((image - torch.sqrt(1 - alpha) * noise) / torch.sqrt(alpha)).clamp(-1, 1)
        image = torch.sqrt(prev) * clean + torch.sqrt(1 - prev) * noise
    pixels = ((image[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(output, format="PNG")
    return str(output)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--checkpoint", default="checkpoints/purple_image.pt")
    p.add_argument("--output", default="generated/purple.png")
    p.add_argument("--sampling-steps", type=int, default=100)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--size", help="Square size or WIDTHxHEIGHT, e.g. 256 or 512x384")
    p.add_argument("--width", type=int, help="Output width (overrides --size width)")
    p.add_argument("--height", type=int, help="Output height (overrides --size height)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    size_width = size_height = None
    if a.size:
        if "x" in a.size.lower():
            parts = a.size.lower().split("x", 1)
            if not all(part.isdigit() for part in parts):
                p.error("--size must be N or WIDTHxHEIGHT, e.g. 256 or 512x384")
            size_width, size_height = map(int, parts)
        elif a.size.isdigit():
            size_width = size_height = int(a.size)
        else:
            p.error("--size must be N or WIDTHxHEIGHT, e.g. 256 or 512x384")
    width = a.width if a.width is not None else size_width
    height = a.height if a.height is not None else size_height
    print("Saved:", generate(a.prompt, a.checkpoint, a.output, a.sampling_steps,
                             a.guidance, a.seed, a.device, width, height))

if __name__ == "__main__":
    main()
