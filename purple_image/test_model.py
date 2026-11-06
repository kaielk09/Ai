"""Quick architecture and gradient smoke test; no dataset or pretrained weights needed."""
import torch
from model import PurpleImageNet


def main():
    torch.manual_seed(321)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PurpleImageNet().to(device)
    image = torch.randn(2, 3, 64, 64, device=device)
    steps = torch.tensor([0, 499], device=device)
    tokens = model.text_encoder.tokenize(
        ["Fioletowy smok nad zamkiem", "A red car on a road"], device
    )
    prediction = model(image, steps, tokens)
    assert prediction.shape == image.shape, (prediction.shape, image.shape)
    loss = prediction.square().mean()
    loss.backward()
    assert torch.isfinite(loss), loss
    assert model.output.weight.grad is not None
    print(f"OK | device={device} | output={tuple(prediction.shape)} | loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
