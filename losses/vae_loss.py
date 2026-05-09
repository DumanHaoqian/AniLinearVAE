import torch
import torch.nn.functional as F


def reconstruction_loss(x_recon, x, loss_type="l1"):
    """
    x_recon: [B, 3, 64, 64]
    x:       [B, 3, 64, 64]
    """

    if loss_type == "l1":
        return F.l1_loss(x_recon, x, reduction="mean")

    elif loss_type == "mse":
        return F.mse_loss(x_recon, x, reduction="mean")

    elif loss_type == "bce":
        return F.binary_cross_entropy(x_recon, x, reduction="mean")

    else:
        raise ValueError(f"Unsupported reconstruction loss: {loss_type}")


def kl_divergence(mu, logvar, reduction="mean_dim"):
    """
    KL(q(z|x) || p(z))

    mu:     [B, latent_dim]
    logvar: [B, latent_dim]

    Return:
        KL loss. By default this is averaged over batch and latent
        dimensions so it is on a scale closer to the mean pixel
        reconstruction loss.
    """

    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    if reduction == "mean_dim":
        kl = kl_per_dim.mean()
    elif reduction == "sum_dim":
        kl = kl_per_dim.sum(dim=1).mean()
    else:
        raise ValueError(f"Unsupported KL reduction: {reduction}")

    return kl


def posterior_stats(mu, logvar, active_threshold=0.01):
    with torch.no_grad():
        mu_var = mu.var(dim=0)
        active_units = (mu_var > active_threshold).float().sum()

        return {
            "mu_abs": mu.abs().mean(),
            "logvar_mean": logvar.mean(),
            "active_units": active_units,
        }


def get_beta(epoch, beta_max=0.1, warmup_epochs=30):
    """
    Linear KL warm-up.

    epoch starts from 1.
    """

    if warmup_epochs <= 0:
        return beta_max

    beta = beta_max * min(epoch / warmup_epochs, 1.0)

    return beta


def beta_vae_loss(
    x_recon,
    x,
    mu,
    logvar,
    beta=1.0,
    recon_loss_type="l1",
    kl_reduction="mean_dim",
):
    """
    Total beta-VAE loss:

        loss = recon_loss + beta * kl_loss
    """

    recon = reconstruction_loss(
        x_recon,
        x,
        loss_type=recon_loss_type,
    )

    kl = kl_divergence(mu, logvar, reduction=kl_reduction)
    stats = posterior_stats(mu, logvar)

    loss = recon + beta * kl

    loss_dict = {
        "loss": loss,
        "recon_loss": recon,
        "kl_loss": kl,
        "beta": beta,
        **stats,
    }

    return loss, loss_dict
