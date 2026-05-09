import os
import torch
from torchvision.utils import save_image


def denormalize(x):
    """
    Convert image from [-1, 1] to [0, 1].
    """

    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def save_reconstructions(model, x, epoch, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    x = x.to(device)
    mu, _ = model.encode(x)
    x_recon = model.decode(mu)

    x = denormalize(x)
    x_recon = denormalize(x_recon)

    comparison = torch.cat([x[:8], x_recon[:8]], dim=0)

    save_path = os.path.join(save_dir, f"recon_epoch_{epoch:03d}.png")
    save_image(comparison, save_path, nrow=8)


@torch.no_grad()
def save_samples(model, epoch, save_dir, device, num_samples=16):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    samples = model.sample(num_samples=num_samples, device=device)
    samples = denormalize(samples)

    save_path = os.path.join(save_dir, f"samples_epoch_{epoch:03d}.png")
    save_image(samples, save_path, nrow=4)
