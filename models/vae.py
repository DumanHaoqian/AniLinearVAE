import torch
import torch.nn as nn

from .blocks import DownBlock, UpBlock, ConvBlock
from .attention import LinearAttention2d


class LinearAttnBetaVAE(nn.Module):
    """
    Linear-Attention Beta-VAE for 64x64 anime faces.

    Input:
        x: [B, 3, 64, 64]

    Encoder:
        3   -> 64   | 64x64 -> 32x32
        64  -> 128  | 32x32 -> 16x16
        Linear Attention at 16x16
        128 -> 256  | 16x16 -> 8x8
        Linear Attention at 8x8
        256 -> 512  | 8x8 -> 4x4

    Latent:
        mu:     [B, latent_dim]
        logvar: [B, latent_dim]

    Decoder:
        latent_dim -> 512x4x4
        512 -> 256 | 4x4 -> 8x8
        Linear Attention at 8x8
        256 -> 128 | 8x8 -> 16x16
        Linear Attention at 16x16
        128 -> 64  | 16x16 -> 32x32
        64  -> 64  | 32x32 -> 64x64
        64  -> 3   | output image
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 128,
        attention_heads: int = 4,
        attention_dim_head: int = 32,
        output_activation: str = "tanh",
    ):
        super().__init__()

        self.image_channels = image_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.output_activation = output_activation

        c1 = base_channels          # 64
        c2 = base_channels * 2      # 128
        c3 = base_channels * 4      # 256
        c4 = base_channels * 8      # 512

        # -------------------------
        # Encoder
        # -------------------------
        self.enc1 = DownBlock(image_channels, c1)  # 64 -> 32
        self.enc2 = DownBlock(c1, c2)              # 32 -> 16
        self.attn16_enc = LinearAttention2d(
            c2,
            heads=attention_heads,
            dim_head=attention_dim_head,
        )

        self.enc3 = DownBlock(c2, c3)              # 16 -> 8
        self.attn8_enc = LinearAttention2d(
            c3,
            heads=attention_heads,
            dim_head=attention_dim_head,
        )

        self.enc4 = DownBlock(c3, c4)              # 8 -> 4

        self.flatten_dim = c4 * 4 * 4

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # -------------------------
        # Decoder
        # -------------------------
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        self.dec1 = UpBlock(c4, c3)                # 4 -> 8
        self.attn8_dec = LinearAttention2d(
            c3,
            heads=attention_heads,
            dim_head=attention_dim_head,
        )

        self.dec2 = UpBlock(c3, c2)                # 8 -> 16
        self.attn16_dec = LinearAttention2d(
            c2,
            heads=attention_heads,
            dim_head=attention_dim_head,
        )

        self.dec3 = UpBlock(c2, c1)                # 16 -> 32
        self.dec4 = UpBlock(c1, c1)                # 32 -> 64

        self.out_conv = nn.Conv2d(
            c1,
            image_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def encode(self, x: torch.Tensor):
        """
        x:
            [B, 3, 64, 64]

        return:
            mu:     [B, latent_dim]
            logvar: [B, latent_dim]
        """

        h = self.enc1(x)          # [B, 64, 32, 32]
        h = self.enc2(h)          # [B, 128, 16, 16]
        h = self.attn16_enc(h)

        h = self.enc3(h)          # [B, 256, 8, 8]
        h = self.attn8_enc(h)

        h = self.enc4(h)          # [B, 512, 4, 4]

        h = torch.flatten(h, start_dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        z = mu + std * eps

        std = exp(0.5 * logvar)
        eps ~ N(0, I)
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor):
        """
        z:
            [B, latent_dim]

        return:
            x_recon: [B, 3, 64, 64]
        """

        h = self.fc_decode(z)
        h = h.view(z.size(0), self.base_channels * 8, 4, 4)

        h = self.dec1(h)          # [B, 256, 8, 8]
        h = self.attn8_dec(h)

        h = self.dec2(h)          # [B, 128, 16, 16]
        h = self.attn16_dec(h)

        h = self.dec3(h)          # [B, 64, 32, 32]
        h = self.dec4(h)          # [B, 64, 64, 64]

        x_recon = self.out_conv(h)

        if self.output_activation == "tanh":
            x_recon = torch.tanh(x_recon)
        elif self.output_activation == "sigmoid":
            x_recon = torch.sigmoid(x_recon)
        elif self.output_activation == "none":
            pass
        else:
            raise ValueError(
                f"Unsupported output_activation: {self.output_activation}"
            )

        return x_recon

    def forward(self, x: torch.Tensor):
        """
        return:
            x_recon, mu, logvar
        """

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        return x_recon, mu, logvar

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device):
        """
        Randomly sample z from standard normal distribution.
        """

        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)

        return samples