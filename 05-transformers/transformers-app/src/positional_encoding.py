# A Transformer processes tokens in parallel, so positional encoding adds sequence-order information.

import torch
import torch.nn as nn

from config import config


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        context_length: int,
        embedding_dim: int
    ) -> None:
        super().__init__()

        if context_length <= 0:
            raise ValueError(
                "context_length must be greater than zero."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        self.context_length = context_length

        self.position_embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embedding_dim
        )

    def forward(
        self,
        token_embeddings: torch.Tensor
    ) -> torch.Tensor:

        if token_embeddings.ndim != 3:
            raise ValueError(
                "Expected token embeddings with shape "
                "[batch_size, sequence_length, embedding_dim]."
            )

        batch_size, sequence_length, _ = (
            token_embeddings.shape
        )

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.context_length}."
            )

        positions = torch.arange(
            sequence_length,
            device=token_embeddings.device
        )

        positional_embeddings = (
            self.position_embedding(positions)
        )

        return (
            token_embeddings
            + positional_embeddings.unsqueeze(0)
        )


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 8

    positional_encoding = PositionalEncoding(
        context_length=config.context_length,
        embedding_dim=config.embedding_dim
    ).to(config.device)

    token_embeddings = torch.randn(
        batch_size,
        sequence_length,
        config.embedding_dim,
        device=config.device
    )

    output = positional_encoding(
        token_embeddings
    )

    print("Device:", output.device)
    print(
        "Input shape:",
        token_embeddings.shape
    )
    print(
        "Output shape:",
        output.shape
    )

    assert output.shape == (
        batch_size,
        sequence_length,
        config.embedding_dim
    )

# Expected:

# Input shape: torch.Size([2, 8, 128])
# Output shape: torch.Size([2, 8, 128])

# The shape stays unchanged:

# Token Embeddings
# [2, 8, 128]
#         +
# Position Embeddings
# [1, 8, 128]
#         ↓
# Position-Aware Embeddings
# [2, 8, 128]

# unsqueeze(0) creates a batch dimension so the same position vectors are broadcast across every sequence in the batch.
