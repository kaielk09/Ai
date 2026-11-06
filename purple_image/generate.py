"""Generate a PNG with the locally trained Purple AI Image checkpoint."""
import argparse
from pathlib import Path

import torch
from PIL import Image

# Support both: python generate.py and import purple_image.generate
try:
    from .model import PurpleImageNet, make_noise_schedule
except ImportError:
    from model import PurpleImageNet, make_noise_schedule


@torch.inference_mode()
def generate(prompt, checkpoint, output, sampling_steps=100, guidance=3.0, seed=321):
    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    if guidance < 0:
        raise ValueError("Guidance must be zero or greater.")

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Image checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(saved, dict):
        raise ValueError("Invalid Purple AI Image checkpoint.")

    model = PurpleImageNet(
        base_channels=int(saved.get("base_channels", 32)),
        text_width=int(saved.get("text_width", 256)),
    ).to(device)

    # Avoid dict.get(..., saved["model"]): Python evaluates that default
    # eagerly, even when the EMA entry exists.
    weights = saved["ema"] if "ema" in saved else saved.get("model")
    if weights is None:
        raise ValueError("Checkpoint does not contain model or EMA weights.")
    model.load_state_dict(weights, strict=True)
    model.eval()

    total_steps = int(saved.get("steps", 500))
    size = int(saved.get("size", 64))
    if size != 64:
        raise ValueError(f"Unsupported checkpoint image size: {size}; expected 64.")
    if not 1 <= sampling_steps <= total_steps:
        raise ValueError(f"sampling_steps must be between 1 and {total_steps}.")

    _, _, alpha_bars = make_noise_schedule(total_steps, device)
    tokens = model.text_encoder.tokenize([prompt], device)
    empty = model.text_encoder.tokenize([""], device)
    image = torch.randn((1, 3, size, size), device=device, generator=generator)
    sequence = torch.linspace(
        total_steps - 1, 0, sampling_steps, device=device
    ).long().unique_consecutive()

    for index, step in enumerate(sequence):
        t = torch.full((1,), int(step.item()), device=device, dtype=torch.long)
        conditional = model(image, t, tokens)

        if guidance == 0:
            predicted_noise = conditional
        else:
            unconditional = model(image, t, empty)
            predicted_noise = unconditional + guidance * (conditional - unconditional)

        alpha_bar = alpha_bars[step]
        previous_alpha_bar = (
            alpha_bars[sequence[index + 1]]
            if index + 1 < len(sequence)
            else torch.ones((), device=device)
        )

        # Deterministic DDIM update with bounded clean-image estimate.
        predicted_clean = (
            image - torch.sqrt(1.0 - alpha_bar) * predicted_noise
        ) / torch.sqrt(alpha_bar)
        predicted_clean = predicted_clean.clamp(-1.0, 1.0)
        image = (
            torch.sqrt(previous_alpha_bar) * predicted_clean
            + torch.sqrt(1.0 - previous_alpha_bar) * predicted_noise
        )

    pixels = ((image[0].clamp(-1, 1) + 1) * 127.5).round().byte()
    pixels = pixels.permute(1, 2, 0).cpu().numpy()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(output, format="PNG")
    return str(output)


def main():
    parser = argparse.ArgumentParser(description="Purple AI Image — local text-to-image model")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/purple_image.pt")
    parser.add_argument("--output", default="generated/purple.png")
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()
    print("Saved:", generate(
        args.prompt, args.checkpoint, args.output,
        args.sampling_steps, args.guidance, args.seed
    ))


if __name__ == "__main__":
    main()
