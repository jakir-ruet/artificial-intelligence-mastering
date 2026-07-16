import torch
import torch.nn as nn

from config import config
from feed_forward import FeedForwardNetwork
from multi_head_attention import MultiHeadCausalSelfAttention


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        feed_forward_dim: int,
        dropout: float
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        self.embedding_dim = embedding_dim

        self.attention_norm = nn.LayerNorm(
            normalized_shape=embedding_dim
        )

        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout
        )

        self.feed_forward_norm = nn.LayerNorm(
            normalized_shape=embedding_dim
        )

        self.feed_forward = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout
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

        if x.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, got {x.size(-1)}."
            )

        # Pre-LayerNorm attention block
        normalized_x = self.attention_norm(x)

        attention_output, attention_weights = (
            self.attention(normalized_x)
        )

        # First residual connection
        x = x + attention_output

        # Pre-LayerNorm feed-forward block
        normalized_x = self.feed_forward_norm(x)

        feed_forward_output = self.feed_forward(
            normalized_x
        )

        # Second residual connection
        x = x + feed_forward_output

        return x, attention_weights


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 8

    block = TransformerBlock(
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        context_length=config.context_length,
        feed_forward_dim=config.feed_forward_dim,
        dropout=config.dropout
    ).to(config.device)

    x = torch.randn(
        batch_size,
        sequence_length,
        config.embedding_dim,
        device=config.device
    )

    output, attention_weights = block(x)

    print("Device:", output.device)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print(
        "Attention weights shape:",
        attention_weights.shape
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

    # Verify that gradients pass through the block.
    loss = output.mean()
    loss.backward()

    has_gradients = any(
        parameter.grad is not None
        for parameter in block.parameters()
    )

    assert has_gradients

    print("Transformer block test passed.")
    print("Gradient-flow test passed.")
