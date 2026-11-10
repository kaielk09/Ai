"""Sampling for the shared-core Purple AI image model."""
import argparse
from pathlib import Path
import torch
from PIL import Image

try:
    from .model import PurpleImageNet, make_noise_schedule
except ImportError:
    from model import PurpleImageNet, make_noise_schedule


@torch.inference_mode()
def generate(text_features, text_mask, checkpoint, output, sampling_steps=50,
             guidance=3.0, seed=321, device="auto", width=None, height=None):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Image checkpoint not found: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("architecture") != "purple-shared-core-v2":
        raise ValueError("Nieobsługiwany checkpoint. Wymagany nowy checkpoint shared-core v2.")
    choice = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if choice not in ("cpu", "cuda") or (choice == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Wybrane urządzenie jest niedostępne.")
    dev = torch.device(choice)
    width = int(width or saved.get("width", 64))
    height = int(height or saved.get("height", 64))
    if not (16 <= width <= 1024 and 16 <= height <= 1024) or width * height > 1048576:
        raise ValueError("Wymiary muszą mieć 16..1024 px, a powierzchnia maks. 1 048 576 px.")
    total_steps = int(saved.get("diffusion_steps", 500))
    if not 1 <= int(sampling_steps) <= total_steps:
        raise ValueError(f"sampling_steps musi być w zakresie 1..{total_steps}.")
    if float(guidance) < 0:
        raise ValueError("guidance musi być >= 0.")
    model = PurpleImageNet(int(saved["text_width"]), int(saved.get("base_channels", 48))).to(dev)
    model.load_state_dict(saved["ema"] if "ema" in saved else saved["model"], strict=True)
    model.eval()
    text_features = text_features.to(dev)
    text_mask = text_mask.to(device=dev, dtype=torch.bool)
    if text_features.shape[-1] != model.text_width:
        raise ValueError(
            f"Niezgodny wymiar wspólnego modelu tekstowego: checkpoint obrazu oczekuje "
            f"{model.text_width}, API dostarczyło {text_features.shape[-1]}."
        )
    rng = torch.Generator(device=dev).manual_seed(int(seed))
    _, _, alpha_bar = make_noise_schedule(total_steps, dev)
    image = torch.randn((1, 3, height, width), generator=rng, device=dev)
    # Classifier-free guidance uses zeroed shared features, not a second text encoder.
    empty_features = torch.zeros_like(text_features)
    sequence = torch.linspace(total_steps - 1, 0, int(sampling_steps), device=dev).long().unique_consecutive()
    for i, step in enumerate(sequence):
        t = torch.full((1,), int(step.item()), device=dev, dtype=torch.long)
        conditional = model(image, t, text_features, text_mask)
        if guidance:
            unconditional = model(image, t, empty_features, text_mask)
            predicted_noise = unconditional + float(guidance) * (conditional - unconditional)
        else:
            predicted_noise = conditional
        a = alpha_bar[step]
        previous = alpha_bar[sequence[i + 1]] if i + 1 < len(sequence) else torch.ones((), device=dev)
        clean = ((image - (1 - a).sqrt() * predicted_noise) / a.sqrt()).clamp(-1, 1)
        image = previous.sqrt() * clean + (1 - previous).sqrt() * predicted_noise
    pixels = ((image[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, "RGB").save(output, format="PNG")
    return str(output)


def main():
    parser = argparse.ArgumentParser(description="Use API-provided shared text features to generate an image.")
    parser.error("Uruchamiaj generowanie przez API Purple AI, które przekazuje ukryte reprezentacje wspólnego modelu tekstowego.")


if __name__ == "__main__":
    main()
