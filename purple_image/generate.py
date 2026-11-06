"""Generate a 64x64 PNG with a locally trained Purple AI Image checkpoint."""
import argparse
from pathlib import Path

import torch
from PIL import Image

from model import PurpleImageNet, make_noise_schedule


@torch.inference_mode()
def generate(prompt, checkpoint, output, sampling_steps=100, guidance=3.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = PurpleImageNet(base_channels=saved.get("base_channels", 32)).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    total_steps = int(saved["steps"])
    if not 1 <= sampling_steps <= total_steps:
        raise ValueError(f"sampling_steps must be between 1 and {total_steps}")
    betas, alphas, alpha_bars = make_noise_schedule(total_steps, device)
    tokens = model.text_encoder.tokenize([prompt], device)
    empty = model.text_encoder.tokenize([""], device)
    image = torch.randn(1, 3, int(saved["size"]), int(saved["size"]), device=device)
    sequence = torch.linspace(total_steps - 1, 0, sampling_steps, device=device).long().unique_consecutive()

    for index, step in enumerate(sequence):
        t = torch.full((1,), int(step.item()), device=device, dtype=torch.long)
        conditional = model(image, t, tokens)
        unconditional = model(image, t, empty)
        predicted_noise = unconditional + guidance * (conditional - unconditional)
        alpha = alphas[step]
        alpha_bar = alpha_bars[step]
        beta = betas[step]
        image = (image - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
        if index + 1 < len(sequence):
            image = image + torch.sqrt(beta) * torch.randn_like(image)

    image = ((image[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(output, format="PNG")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/purple_image.pt")
    parser.add_argument("--output", default="generated/purple.png")
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--guidance", type=float, default=3.0)
    args = parser.parse_args()
    print("Saved:", generate(args.prompt, args.checkpoint, args.output,
                            args.sampling_steps, args.guidance))


if __name__ == "__main__":
    main()
