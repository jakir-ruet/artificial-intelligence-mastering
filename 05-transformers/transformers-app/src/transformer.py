# This file assembles the complete decoder-only Transformer language model:

# Token IDs
#    ↓
# Token Embeddings
#    +
# Positional Embeddings
#    ↓
# Transformer Blocks × N
#    ↓
# Final LayerNorm
#    ↓
# Vocabulary Projection
#    ↓
# Next-token Logits

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from embedding import TokenEmbedding
from positional_encoding import PositionalEncoding
from transformer_block import TransformerBlock


class MiniTransformer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        feed_forward_dim: int,
        dropout: float
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be greater than zero."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than zero."
            )

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )

        self.positional_encoding = PositionalEncoding(
            context_length=context_length,
            embedding_dim=embedding_dim
        )

        self.embedding_dropout = nn.Dropout(
            dropout
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                context_length=context_length,
                feed_forward_dim=feed_forward_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            embedding_dim
        )

        self.language_model_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False
        )

        self.apply(
            self._initialize_weights
        )

    @staticmethod
    def _initialize_weights(
        module: nn.Module
    ) -> None:

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor]
    ]:

        if token_ids.ndim != 2:
            raise ValueError(
                "Expected token_ids with shape "
                "[batch_size, sequence_length]."
            )

        batch_size, sequence_length = (
            token_ids.shape
        )

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.context_length}."
            )

        if token_ids.dtype != torch.long:
            raise TypeError(
                "token_ids must use torch.long dtype."
            )

        # [B, T] -> [B, T, C]
        x = self.token_embedding(
            token_ids
        )

        # Add learned position information.
        x = self.positional_encoding(
            x
        )

        x = self.embedding_dropout(
            x
        )

        for block in self.blocks:
            x, _ = block(x)

        x = self.final_norm(
            x
        )

        # [B, T, C] -> [B, T, V]
        logits = self.language_model_head(
            x
        )

        loss = None

        if targets is not None:

            if targets.shape != token_ids.shape:
                raise ValueError(
                    "targets must have the same shape "
                    "as token_ids."
                )

            if targets.dtype != torch.long:
                raise TypeError(
                    "targets must use torch.long dtype."
                )

            loss = F.cross_entropy(
                logits.reshape(
                    batch_size * sequence_length,
                    self.vocab_size
                ),
                targets.reshape(
                    batch_size * sequence_length
                )
            )

        return logits, loss

    def count_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


if __name__ == "__main__":
    vocab_size = 50
    batch_size = 2
    sequence_length = 16

    model = MiniTransformer(
        vocab_size=vocab_size,
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        feed_forward_dim=config.feed_forward_dim,
        dropout=config.dropout
    ).to(config.device)

    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            sequence_length
        ),
        dtype=torch.long,
        device=config.device
    )

    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            sequence_length
        ),
        dtype=torch.long,
        device=config.device
    )

    logits, loss = model(
        token_ids,
        targets
    )

    print("Device:", logits.device)
    print("Input shape:", token_ids.shape)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
    print(
        "Trainable parameters:",
        f"{model.count_parameters():,}"
    )

    assert logits.shape == (
        batch_size,
        sequence_length,
        vocab_size
    )

    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()

    has_gradients = any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    assert has_gradients

    print("Transformer model test passed.")
    print("Gradient-flow test passed.")


# Expected structure of the output:

# Device: mps:0
# Input shape: torch.Size([2, 16])
# Logits shape: torch.Size([2, 16, 50])
# Loss: ...
# Trainable parameters: ...
# Transformer model test passed.
# Gradient-flow test passed.
# Understanding the output

# Input:

# [B, T]
# =
# [2, 16]

# After token and positional embeddings:

# [B, T, C]
# =
# [2, 16, 128]

# After the Transformer blocks:

# [2, 16, 128]

# After vocabulary projection:

# [B, T, V]
# =
# [2, 16, 50]

# For every token position, the model produces one score for every vocabulary token:

# Position 1 → 50 next-token scores
# Position 2 → 50 next-token scores
# ...
# Position 16 → 50 next-token scores
# Why Cross-Entropy?

# The target for each input token is the next token:

# Input:
# t₁  t₂  t₃  t₄

# Target:
# t₂  t₃  t₄  t₅

# The model produces raw vocabulary logits:

# Vocabulary logits
#       ↓
# Cross-Entropy Loss
#       ↓
# Compare predicted token with target token

# Do not apply Softmax before F.cross_entropy(). Cross-entropy handles the required normalization internally.
