import torch
import torch.nn as nn

class LinearAttention2d(nn.Module):
    """
    Linear Attention for 2D feature maps.
    Input:
        x: [B, C, H, W]
    Output:
        out: [B, C, H, W]
    Core idea:
        Avoid computing QK^T with size [N, N].
        Instead compute K^T V first, then multiply by Q.
    """
    def __init__(self, channels: int, heads: int = 4, dim_head: int = 32):
        super().__init__()

        self.channels = channels
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Conv2d(
            channels,
            self.inner_dim * 3,
            kernel_size=1,
            bias=False,
        )

        self.to_out = nn.Conv2d(
            self.inner_dim,
            channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # [B, heads * dim_head, H, W]
        # -> [B, heads, dim_head, H * W]
        q = q.view(b, self.heads, self.dim_head, n)
        k = k.view(b, self.heads, self.dim_head, n)
        v = v.view(b, self.heads, self.dim_head, n)

        # Common linear attention normalization
        q = q.softmax(dim=2) * self.scale      # softmax over channel dimension
        k = k.softmax(dim=3)                   # softmax over spatial tokens

        # context: [B, heads, dim_head, dim_head]
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        # out: [B, heads, dim_head, H * W]
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)

        out = out.contiguous().view(b, self.inner_dim, h, w)
        out = self.to_out(out)

        return x + out