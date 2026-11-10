"""Smoke tests for output shapes, gradients and prompt color parsing."""
import torch
from model import PurpleImageNet, make_noise_schedule
from colors import analyze_colors


def main():
    torch.manual_seed(321)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PurpleImageNet().to(device)
    for height, width in ((64, 64), (124, 124), (96, 128)):
        image = torch.randn(1, 3, height, width, device=device)
        steps = torch.tensor([10], device=device)
        tokens = model.text_encoder.tokenize(["Ciemny fioletowy smok"], device)
        prediction = model(image, steps, tokens)
        assert prediction.shape == image.shape
        assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert model.output.weight.grad is not None

    # A short cosine schedule must end near pure noise. This guards against
    # the old 100-step linear schedule whose terminal alpha_bar was too high.
    _, _, alpha_bars = make_noise_schedule(100, device, schedule="cosine")
    assert alpha_bars[-1].item() < 1e-3, alpha_bars[-1].item()
    _, _, legacy_bars = make_noise_schedule(100, device, schedule="linear")
    assert legacy_bars[-1].item() > 0.1, legacy_bars[-1].item()

    # Arbitrary RGB/HEX values and CSS named colors must be recognized.
    palette = analyze_colors("smok w kolorze #33AAFF z akcentem rgb(12, 34, 56)")
    assert "#33AAFF" in palette["colors"]
    assert "#0C2238" in palette["colors"]
    css = analyze_colors("cornflowerblue rebeccapurple")
    assert len(css["colors"]) == 2
    polish = analyze_colors("ciemny fioletowy smok")
    assert polish["colors"] and polish["palette"]
    automatic = analyze_colors("ciemny obraz z ładnymi odcieniami")
    assert automatic["palette"] and automatic["conditioning"]
    print("OK: model shapes, gradients, arbitrary RGB/HEX, CSS names and Polish aliases")


if __name__ == "__main__":
    main()
