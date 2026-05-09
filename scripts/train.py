import os
import sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LinearAttnBetaVAE
from datasets.anime_face import AnimeFaceDataset
from losses.vae_loss import beta_vae_loss, get_beta
from utils.seed import set_seed
from utils.visualization import save_reconstructions, save_samples


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    beta,
    recon_loss_type,
    kl_reduction,
):
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_mu_abs = 0.0
    total_logvar_mean = 0.0
    total_active_units = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
        leave=True,
    )

    for x in progress_bar:
        x = x.to(device)

        x_recon, mu, logvar = model(x)

        loss, loss_dict = beta_vae_loss(
            x_recon=x_recon,
            x=x,
            mu=mu,
            logvar=logvar,
            beta=beta,
            recon_loss_type=recon_loss_type,
            kl_reduction=kl_reduction,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss_dict["loss"].item()
        total_recon += loss_dict["recon_loss"].item()
        total_kl += loss_dict["kl_loss"].item()
        total_mu_abs += loss_dict["mu_abs"].item()
        total_logvar_mean += loss_dict["logvar_mean"].item()
        total_active_units += loss_dict["active_units"].item()

        progress_bar.set_postfix({
            "loss": f"{loss_dict['loss'].item():.4f}",
            "recon": f"{loss_dict['recon_loss'].item():.4f}",
            "kl": f"{loss_dict['kl_loss'].item():.4f}",
            "au": f"{loss_dict['active_units'].item():.0f}",
            "beta": f"{beta:.4f}",
        })

    num_batches = len(dataloader)

    return {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon / num_batches,
        "kl_loss": total_kl / num_batches,
        "beta": beta,
        "mu_abs": total_mu_abs / num_batches,
        "logvar_mean": total_logvar_mean / num_batches,
        "active_units": total_active_units / num_batches,
    }


def save_checkpoint(model, optimizer, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    save_path = os.path.join(save_dir, f"vae_epoch_{epoch:03d}.pt")
    torch.save(checkpoint, save_path)


def main():
    # -------------------------
    # Config
    # -------------------------
    seed = 42

    image_size = 64
    image_channels = 3
    latent_dim = 128
    base_channels = 64

    batch_size = 64
    num_workers = 4
    epochs = 100
    lr = 2e-4

    recon_loss_type = "l1"

    kl_reduction = "mean_dim"
    beta_max = 0.1
    warmup_epochs = 30

    sample_every = 5
    checkpoint_every = 10

    output_dir = "outputs"
    sample_dir = os.path.join(output_dir, "samples")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    # -------------------------
    # Setup
    # -------------------------
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # Dataset
    # -------------------------
    dataset = AnimeFaceDataset(
        data_path="/home/haoqian/Data/GenAI/AniLinearVAE/Data/data",
        image_size=image_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    fixed_batch = next(iter(dataloader)).to(device)

    # -------------------------
    # Model
    # -------------------------
    model = LinearAttnBetaVAE(
        image_channels=image_channels,
        base_channels=base_channels,
        latent_dim=latent_dim,
        output_activation="tanh",
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(1, epochs + 1):
        beta = get_beta(
            epoch=epoch,
            beta_max=beta_max,
            warmup_epochs=warmup_epochs,
        )

        logs = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            beta=beta,
            recon_loss_type=recon_loss_type,
            kl_reduction=kl_reduction,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={logs['loss']:.4f} | "
            f"recon={logs['recon_loss']:.4f} | "
            f"kl={logs['kl_loss']:.4f} | "
            f"beta={logs['beta']:.4f} | "
            f"mu_abs={logs['mu_abs']:.4f} | "
            f"logvar={logs['logvar_mean']:.4f} | "
            f"active={logs['active_units']:.1f}/{latent_dim}"
        )

        if epoch % sample_every == 0:
            save_reconstructions(
                model=model,
                x=fixed_batch,
                epoch=epoch,
                save_dir=sample_dir,
                device=device,
            )

            save_samples(
                model=model,
                epoch=epoch,
                save_dir=sample_dir,
                device=device,
                num_samples=16,
            )

        if epoch % checkpoint_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                save_dir=checkpoint_dir,
            )


if __name__ == "__main__":
    main()
