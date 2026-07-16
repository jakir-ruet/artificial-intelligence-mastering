# The feed-forward network processes each token position independently after attention.
import torch
import torch.nn as nn

from config import config


class FeedForwardNetwork(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        feed_forward_dim: int,
        dropout: float
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        if feed_forward_dim <= 0:
            raise ValueError(
                "feed_forward_dim must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be between 0.0 and 1.0."
            )

        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim,
                feed_forward_dim
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                feed_forward_dim,
                embedding_dim
            ),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "Expected x with shape "
                "[batch_size, sequence_length, embedding_dim]."
            )

        return self.network(x)


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 8

    feed_forward = FeedForwardNetwork(
        embedding_dim=config.embedding_dim,
        feed_forward_dim=config.feed_forward_dim,
        dropout=config.dropout
    ).to(config.device)

    x = torch.randn(
        batch_size,
        sequence_length,
        config.embedding_dim,
        device=config.device
    )

    output = feed_forward(x)

    print("Device:", output.device)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    assert output.shape == (
        batch_size,
        sequence_length,
        config.embedding_dim
    )

    print("Feed-forward test passed.")

# Expected:

# Device: mps:0
# Input shape: torch.Size([2, 8, 128])
# Output shape: torch.Size([2, 8, 128])
# Feed-forward test passed.
# What it does

# With your configuration:

# 128
#  ↓
# Linear
#  ↓
# 512
#  ↓
# GELU
#  ↓
# Linear
#  ↓
# 128

# Shape flow:

# Input
# [B, T, 128]
#       ↓
# Linear expansion
# [B, T, 512]
#       ↓
# GELU
# [B, T, 512]
#       ↓
# Linear projection
# [B, T, 128]

# The same network is applied independently to every token position:

# Token 1 → FFN
# Token 2 → FFN
# Token 3 → FFN
# ...

# Attention mixes information between tokens. The feed-forward network transforms the resulting representation within each token position.
