"""Purple AI Image inference: DDIM sampling, CFG negative prompts and safe sizing."""
import argparse
import secrets
from pathlib import Path

import torch
from PIL import Image

try:
    from .model import PurpleImageNet, make_noise_schedule
    from .colors import analyze_colors
except ImportError:
    from model import PurpleImageNet, make_noise_schedule
    from colors import analyze_colors


def _ddim_step(image, predicted_noise, alpha_now, alpha_previous, eta, generator):
    """One mathematically correct DDIM step; eta=0 is deterministic."""
    clean = (image - torch.sqrt(1.0 - alpha_now) * predicted_noise) / torch.sqrt(alpha_now)
    clean = clean.clamp(-1.0, 1.0)
    sigma = eta * torch.sqrt(
        ((1.0 - alpha_previous) / (1.0 - alpha_now).clamp_min(1e-12))
        * (1.0 - alpha_now / alpha_previous.clamp_min(1e-12))
    )
    direction_scale = torch.sqrt((1.0 - alpha_previous - sigma.square()).clamp_min(0.0))
    noise = torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=generator)
    return torch.sqrt(alpha_previous) * clean + direction_scale * predicted_noise + sigma * noise


@torch.inference_mode()
def generate(prompt, checkpoint, output, sampling_steps=100, guidance=3.0, seed=None,
             device="auto", width=None, height=None, negative_prompt="", eta=0.0):
    prompt = str(prompt).strip()
    negative_prompt = str(negative_prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    if guidance < 0:
        raise ValueError("Guidance must be >= 0.")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be between 0 and 1.")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    choice = str(device).lower()
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice not in ("cpu", "cuda") or (choice == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device is unavailable or invalid.")
    target = torch.device(choice)
    saved = torch.load(checkpoint, map_location=target, weights_only=False)
    if not isinstance(saved, dict):
        raise ValueError("Invalid checkpoint.")
    model = PurpleImageNet(int(saved.get("base_channels", 32)),
                           int(saved.get("text_width", 256))).to(target)
    weights = saved.get("ema", saved.get("model"))
    if weights is None:
        raise ValueError("Checkpoint has no model weights.")
    model.load_state_dict(weights, strict=True)
    model.eval()

    total_steps = int(saved.get("steps", 500))
    schedule_name = saved.get("noise_schedule", "linear-v1")
    if schedule_name == "cosine-v2":
        schedule = "cosine"
    elif schedule_name in ("linear-v1", None):
        schedule = "linear"
    else:
        raise ValueError(f"Unsupported checkpoint noise schedule: {schedule_name}")
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
    color_hint = ("Colors: " + ", ".join(color_info["colors"][:4]) + ". ") if color_info["colors"] else ""
    model_prompt = color_hint + prompt
    # None means a fresh random image each call; pass an integer to reproduce a result.
    if seed is None:
        seed = secrets.randbits(63)
    seed = int(seed)
    print(f"Generation seed: {seed}")
    rng = torch.Generator(device=target).manual_seed(seed)
    _, _, alpha_bars = make_noise_schedule(total_steps, target, schedule=schedule)

    trained_steps = int(saved.get("global_step", 0))
    dataset_size = int(saved.get("dataset_size", 0))
    if trained_steps < 10000 or dataset_size < 1000:
        print("WARNING: checkpoint is undertrained for text-to-image "
              f"(updates={trained_steps}, image-caption pairs={dataset_size}). "
              "Output may be noisy or incoherent.")

    tokens = model.text_encoder.tokenize([model_prompt], target)
    # Classifier-free guidance: use the negative prompt as the unconditioned branch.
    negative_tokens = model.text_encoder.tokenize([negative_prompt], target)
    image = torch.randn((1, 3, height, width), device=target, generator=rng)
    sequence = torch.linspace(total_steps - 1, 0, sampling_steps, device=target).long().unique_consecutive()

    for index, step in enumerate(sequence):
        timestep = torch.full((1,), int(step.item()), device=target, dtype=torch.long)
        conditional_noise = model(image, timestep, tokens)
        if guidance:
            negative_noise = model(image, timestep, negative_tokens)
            predicted_noise = negative_noise + guidance * (conditional_noise - negative_noise)
        else:
            predicted_noise = conditional_noise
        alpha_now = alpha_bars[step]
        alpha_previous = (alpha_bars[sequence[index + 1]]
                          if index + 1 < len(sequence)
                          else torch.ones((), device=target))
        image = _ddim_step(image, predicted_noise, alpha_now, alpha_previous, eta, rng)

    pixels = ((image[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(output, format="PNG")
    return str(output)


def main():
    parser = argparse.ArgumentParser(description="Generate an image with the local Purple AI model.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="", help="Content or traits to discourage.")
    parser.add_argument("--checkpoint", default="checkpoints/purple_image.pt")
    parser.add_argument("--output", default="generated/purple.png")
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity: 0=repeatable, 1=more random.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional fixed seed. Omit it for a different image each run.")
    parser.add_argument("--size", help="Square size or WIDTHxHEIGHT, e.g. 256 or 512x384")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    size_width = size_height = None
    if args.size:
        if "x" in args.size.lower():
            parts = args.size.lower().split("x", 1)
            if not all(part.isdigit() for part in parts):
                parser.error("--size must be N or WIDTHxHEIGHT, e.g. 256 or 512x384")
            size_width, size_height = map(int, parts)
        elif args.size.isdigit():
            size_width = size_height = int(args.size)
        else:
            parser.error("--size must be N or WIDTHxHEIGHT, e.g. 256 or 512x384")
    width = args.width if args.width is not None else size_width
    height = args.height if args.height is not None else size_height
    result = generate(args.prompt, args.checkpoint, args.output, args.sampling_steps,
                      args.guidance, args.seed, args.device, width, height,
                      args.negative_prompt, args.eta)
    print("Saved:", result)


if __name__ == "__main__":
    main()
