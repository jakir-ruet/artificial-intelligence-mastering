# This file converts integer token IDs into dense vectors.
import torch
import torch.nn as nn

from config import config


class TokenEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.embedding(token_ids)


if __name__ == "__main__":
    vocab_size = 40
    batch_size = 2
    sequence_length = 8

    model = TokenEmbedding(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim
    ).to(config.device)

    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
        dtype=torch.long,
        device=config.device
    )

    embeddings = model(token_ids)

    print("Device:", embeddings.device)
    print("Token IDs shape:", token_ids.shape)
    print("Embedding shape:", embeddings.shape)

    assert embeddings.shape == (
        batch_size,
        sequence_length,
        config.embedding_dim
    )

# Expected shape:

# Token IDs shape: torch.Size([2, 8])
# Embedding shape: torch.Size([2, 8, 128])

# Meaning:

# 2 sequences
# ×
# 8 tokens per sequence
# ×
# 128 values per token

# The transformation is:

# Token IDs
# [2, 8]
#    ↓
# nn.Embedding
#    ↓
# Token Vectors
# [2, 8, 128]

# Important: nn.Embedding is a trainable lookup table. Its weights begin randomly and are learned during training through backpropagation.
