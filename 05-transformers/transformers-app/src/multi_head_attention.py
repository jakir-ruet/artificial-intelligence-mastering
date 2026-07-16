# This implementation splits the embedding dimension across multiple attention heads.

# With your configuration:

# embedding_dim = 128
# num_heads     = 4
# head_dim      = 128 / 4 = 32

import math

import torch
import torch.nn as nn

from config import config


class MultiHeadCausalSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be greater than zero."
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
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
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.context_length = context_length

        # A single projection produces Q, K, and V together.
        self.qkv_projection = nn.Linear(
            embedding_dim,
            3 * embedding_dim,
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

        self.output_dropout = nn.Dropout(
            dropout
        )

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

        # Input:
        # [B, T, C]
        #
        # Projected QKV:
        # [B, T, 3C]
        qkv = self.qkv_projection(x)

        # Split the final dimension into:
        # Q, K, V — each [B, T, C]
        queries, keys, values = qkv.chunk(
            chunks=3,
            dim=-1
        )

        # Reshape:
        # [B, T, C]
        # ->
        # [B, T, H, D]
        #
        # H = number of heads
        # D = head dimension
        queries = queries.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        keys = keys.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        values = values.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        # Transpose:
        # [B, T, H, D]
        # ->
        # [B, H, T, D]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Attention scores:
        #
        # [B, H, T, D]
        # @
        # [B, H, D, T]
        # ->
        # [B, H, T, T]
        attention_scores = (
            queries
            @ keys.transpose(-2, -1)
        )

        attention_scores = (
            attention_scores
            / math.sqrt(self.head_dim)
        )

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length
        ]

        # Reshape mask for broadcasting:
        # [T, T] -> [1, 1, T, T]
        mask = mask.unsqueeze(0).unsqueeze(0)

        attention_scores = (
            attention_scores.masked_fill(
                ~mask,
                float("-inf")
            )
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = (
            self.attention_dropout(
                attention_weights
            )
        )

        # Weighted values:
        #
        # [B, H, T, T]
        # @
        # [B, H, T, D]
        # ->
        # [B, H, T, D]
        context = attention_weights @ values

        # Restore token-major layout:
        #
        # [B, H, T, D]
        # ->
        # [B, T, H, D]
        context = context.transpose(1, 2)

        # Merge all heads:
        #
        # [B, T, H, D]
        # ->
        # [B, T, C]
        context = context.contiguous().reshape(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        output = self.output_projection(
            context
        )

        output = self.output_dropout(
            output
        )

        return output, attention_weights


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 8

    attention = MultiHeadCausalSelfAttention(
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        context_length=config.context_length,
        dropout=config.dropout
    ).to(config.device)

    x = torch.randn(
        batch_size,
        sequence_length,
        config.embedding_dim,
        device=config.device
    )

    output, attention_weights = attention(x)

    print("Device:", output.device)
    print("Head dimension:", attention.head_dim)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print(
        "Attention weights shape:",
        attention_weights.shape
    )

    assert attention.head_dim == (
        config.embedding_dim
        // config.num_heads
    )

    assert output.shape == (
        batch_size,
        sequence_length,
        config.embedding_dim
    )

    assert attention_weights.shape == (
        batch_size,
        config.num_heads,
        sequence_length,
        sequence_length
    )

    # Verify causal masking for every head.
    future_attention = torch.triu(
        attention_weights,
        diagonal=1
    )

    assert torch.allclose(
        future_attention,
        torch.zeros_like(future_attention),
        atol=1e-6
    )

    print("Multi-head causal-mask test passed.")


# Expected:

# Device: mps:0
# Head dimension: 32
# Input shape: torch.Size([2, 8, 128])
# Output shape: torch.Size([2, 8, 128])
# Attention weights shape: torch.Size([2, 4, 8, 8])
# Multi-head causal-mask test passed.
# Shape flow
# Input
# [B, T, 128]
#       ↓
# QKV Projection
# [B, T, 384]
#       ↓
# Split Q, K, V
# 3 × [B, T, 128]
#       ↓
# Split into 4 heads
# 3 × [B, 4, T, 32]
#       ↓
# Attention per head
# [B, 4, T, T]
#       ↓
# Weighted values
# [B, 4, T, 32]
#       ↓
# Concatenate heads
# [B, T, 128]
#       ↓
# Output projection
# [B, T, 128]

# The important improvement over the previous file is:

# Single-head attention
# → one 128-dimensional attention view

# Multi-head attention
# → four parallel 32-dimensional attention views
