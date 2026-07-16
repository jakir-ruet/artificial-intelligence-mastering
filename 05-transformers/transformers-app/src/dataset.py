from typing import Tuple

import torch
from torch.utils.data import Dataset

from config import config
from tokenizer import CharacterTokenizer


class TextSequenceDataset(Dataset):

    def __init__(
        self,
        text: str,
        tokenizer: CharacterTokenizer,
        context_length: int
    ) -> None:
        if not text:
            raise ValueError("Text cannot be empty.")

        if context_length <= 0:
            raise ValueError(
                "Context length must be greater than zero."
            )

        self.tokenizer = tokenizer
        self.context_length = context_length

        token_ids = tokenizer.encode(text)

        if len(token_ids) <= context_length:
            raise ValueError(
                "Training text must contain more tokens "
                "than the configured context length."
            )

        self.tokens = torch.tensor(
            token_ids,
            dtype=torch.long
        )

    def __len__(self) -> int:
        return (
            len(self.tokens)
            - self.context_length
        )

    def __getitem__(
        self,
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        input_tokens = self.tokens[
            index:index + self.context_length
        ]

        target_tokens = self.tokens[
            index + 1:index + self.context_length + 1
        ]

        return input_tokens, target_tokens


if __name__ == "__main__":
    training_text = config.data_path.read_text(
        encoding="utf-8"
    )

    tokenizer = CharacterTokenizer.from_text(
        training_text
    )

    test_context_length = min(
        16,
        config.context_length
    )

    dataset = TextSequenceDataset(
        text=training_text,
        tokenizer=tokenizer,
        context_length=test_context_length
    )

    input_tokens, target_tokens = dataset[0]

    print("Dataset samples:", len(dataset))
    print("Input shape:", input_tokens.shape)
    print("Target shape:", target_tokens.shape)

    print("\nInput token IDs:")
    print(input_tokens)

    print("\nTarget token IDs:")
    print(target_tokens)

    print("\nDecoded input:")
    print(tokenizer.decode(input_tokens.tolist()))

    print("\nDecoded target:")
    print(tokenizer.decode(target_tokens.tolist()))

# What you should observe

# The target is the input shifted one character to the left:

# Input:
# Artificial intel

# Target:
# rtificial intell

# ------------------------

# This teaches the Transformer:

# Given previous tokens
# ↓
# Predict the next token
