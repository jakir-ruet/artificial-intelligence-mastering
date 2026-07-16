# This implements causal scaled dot-product self-attention from scratch.
import math

import torch
import torch.nn as nn

from config import config


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        context_length: int,
        dropout: float
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        if context_length <= 0:
            raise ValueError(
                "context_length must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be between 0.0 and 1.0."
            )

        self.embedding_dim = embedding_dim
        self.context_length = context_length

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        # Lower-triangular causal mask:
        # token t can attend only to positions <= t.
        causal_mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
                dtype=torch.bool
            )
        )

        self.register_buffer(
            "causal_mask",
            causal_mask
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if x.ndim != 3:
            raise ValueError(
                "Expected x with shape "
                "[batch_size, sequence_length, embedding_dim]."
            )

        batch_size, sequence_length, embedding_dim = (
            x.shape
        )

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, got {embedding_dim}."
            )

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.context_length}."
            )

        # [B, T, C]
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # [B, T, C] @ [B, C, T]
        # -> [B, T, T]
        attention_scores = (
            queries
            @ keys.transpose(-2, -1)
        )

        attention_scores = (
            attention_scores
            / math.sqrt(self.embedding_dim)
        )

        # [T, T] -> broadcast over batch dimension.
        mask = self.causal_mask[
            :sequence_length,
            :sequence_length
        ]

        attention_scores = (
            attention_scores.masked_fill(
                ~mask,
                float("-inf")
            )
        )

        # Normalize across key positions.
        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = (
            self.attention_dropout(
                attention_weights
            )
        )

        # [B, T, T] @ [B, T, C]
        # -> [B, T, C]
        context = attention_weights @ values

        output = self.output_projection(
            context
        )

        return output, attention_weights


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 8

    attention = CausalSelfAttention(
        embedding_dim=config.embedding_dim,
        context_length=config.context_length,
        dropout=config.dropout
    ).to(config.device)

    x = torch.randn(
        batch_size,
        sequence_length,
        config.embedding_dim,
        device=config.device
    )

    output, weights = attention(x)

    print("Device:", output.device)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", weights.shape)

    print("\nFirst attention matrix:")
    print(weights[0])

    assert output.shape == (
        batch_size,
        sequence_length,
        config.embedding_dim
    )

    assert weights.shape == (
        batch_size,
        sequence_length,
        sequence_length
    )

    # Future positions must have zero attention.
    future_positions = torch.triu(
        weights[0],
        diagonal=1
    )

    assert torch.allclose(
        future_positions,
        torch.zeros_like(future_positions),
        atol=1e-6
    )

    print("\nCausal-mask test passed.")

# Expected shapes:

# Input shape: torch.Size([2, 8, 128])
# Output shape: torch.Size([2, 8, 128])
# Attention weights shape: torch.Size([2, 8, 8])
# Causal-mask test passed.
# What this code does
# Input X
#   ↓
# Linear projections
#   ↓
# Q, K, V
#   ↓
# QKᵀ / √d
#   ↓
# Causal mask
#   ↓
# Softmax
#   ↓
# Attention weights
#   ↓
# Weights × V
#   ↓
# Context vectors
#   ↓
# Output projection

# The causal mask prevents a token from attending to future tokens:

# Token 1 → Token 1 only
# Token 2 → Tokens 1–2
# Token 3 → Tokens 1–3
# Token 4 → Tokens 1–4

# masked_fill supports a broadcastable boolean mask, and transpose(-2, -1) swaps the final two tensor dimensions for the Query–Key matrix multiplication.
