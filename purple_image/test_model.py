"""Smoke tests for the shared-core image network (no dataset/checkpoint needed)."""
import torch
from purple_image.model import PurpleImageNet
from purple_image.colors import analyze_colors


def main():
    torch.manual_seed(321)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_width = 64
    model = PurpleImageNet(text_width=text_width, base=32).to(device)
    for height, width in ((64, 64), (124, 124), (96, 128)):
        image = torch.randn(1, 3, height, width, device=device)
        timesteps = torch.tensor([10], device=device)
        # Represents hidden states returned by the shared Purple AI text backbone.
        shared_text = torch.randn(1, 24, text_width, device=device)
        text_mask = torch.ones(1, 24, dtype=torch.bool, device=device)
        prediction = model(image, timesteps, shared_text, text_mask)
        assert prediction.shape == image.shape
        assert torch.isfinite(prediction).all()
        prediction.square().mean().backward()
        assert model.out.weight.grad is not None
        model.zero_grad(set_to_none=True)

    palette = analyze_colors("smok w kolorze #33AAFF oraz rgb(12, 34, 56)")
    assert "#33AAFF" in palette["colors"]
    assert "#0C2238" in palette["colors"]
    print("OK: shared hidden-state conditioning, shapes, gradients and color parsing")


if __name__ == "__main__":
    main()
