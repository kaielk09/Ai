"""Smoke tests for output shapes, gradients and color palette parsing."""
import torch
from model import PurpleImageNet
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
    palette = analyze_colors("ciemny fioletowy smok z #33AAFF akcentami")
    assert palette["colors"] and palette["palette"]
    print("OK: 64x64, 124x124, 128x96 and color parser")

if __name__ == "__main__":
    main()