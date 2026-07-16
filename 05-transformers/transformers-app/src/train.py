# Load text
# → Build tokenizer
# → Build next-token dataset
# → Create DataLoader
# → Create MiniTransformer
# → Train with AdamW
# → Clip gradients
# → Save checkpoint

# PyTorch’s Dataset and DataLoader separate sample preparation from batching, shuffling, and iteration. Saving model and optimizer state_dict() values provides a flexible checkpoint that can later resume training or support inference.

import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import config
from dataset import TextSequenceDataset
from tokenizer import CharacterTokenizer
from transformer import MiniTransformer


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_text(data_path: Path) -> str:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {data_path}"
        )

    text = data_path.read_text(
        encoding="utf-8"
    )

    if not text.strip():
        raise ValueError(
            f"Training file is empty: {data_path}"
        )

    return text


def create_data_loader(
    training_text: str,
    tokenizer: CharacterTokenizer
) -> DataLoader:

    dataset = TextSequenceDataset(
        text=training_text,
        tokenizer=tokenizer,
        context_length=config.context_length
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    if len(data_loader) == 0:
        raise ValueError(
            "DataLoader contains no batches. "
            "Check the training text, context length, "
            "and batch size."
        )

    return data_loader


def create_model(
    vocab_size: int
) -> MiniTransformer:

    model = MiniTransformer(
        vocab_size=vocab_size,
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        feed_forward_dim=config.feed_forward_dim,
        dropout=config.dropout
    )

    return model.to(config.device)


def save_checkpoint(
    model: MiniTransformer,
    optimizer: AdamW,
    tokenizer: CharacterTokenizer,
    epoch: int,
    loss: float
) -> None:

    config.model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "loss": loss,

        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "tokenizer_stoi": tokenizer.stoi,

        "model_config": {
            "vocab_size": tokenizer.vocab_size,
            "context_length": config.context_length,
            "embedding_dim": config.embedding_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "feed_forward_dim": config.feed_forward_dim,
            "dropout": config.dropout
        }
    }

    torch.save(
        checkpoint,
        config.model_path
    )


def train() -> None:
    set_random_seed(
        config.random_seed
    )

    training_text = load_training_text(
        config.data_path
    )

    tokenizer = CharacterTokenizer.from_text(
        training_text
    )

    train_loader = create_data_loader(
        training_text=training_text,
        tokenizer=tokenizer
    )

    model = create_model(
        vocab_size=tokenizer.vocab_size
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )

    print("Training Configuration")
    print("----------------------")
    print("Device             :", config.device)
    print("Training characters:", len(training_text))
    print("Vocabulary size    :", tokenizer.vocab_size)
    print("Dataset samples    :", len(train_loader.dataset))
    print("Batches per epoch  :", len(train_loader))
    print("Batch size         :", config.batch_size)
    print("Context length     :", config.context_length)
    print("Embedding dimension:", config.embedding_dim)
    print("Attention heads    :", config.num_heads)
    print("Transformer layers :", config.num_layers)
    print("Trainable parameters:",
          f"{model.count_parameters():,}")
    print()

    best_loss = float("inf")

    training_start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()

        total_loss = 0.0
        total_batches = 0

        epoch_start = time.perf_counter()

        for input_tokens, target_tokens in train_loader:
            input_tokens = input_tokens.to(
                config.device
            )

            target_tokens = target_tokens.to(
                config.device
            )

            # 1. Clear gradients from previous step
            optimizer.zero_grad(
                set_to_none=True
            )

            # 2. Forward propagation
            _, loss = model(
                token_ids=input_tokens,
                targets=target_tokens
            )

            if loss is None:
                raise RuntimeError(
                    "Model did not return training loss."
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected: "
                    f"{loss.item()}"
                )

            # 3. Backpropagation
            loss.backward()

            # 4. Limit excessively large gradients
            clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            # 5. Update parameters
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        average_loss = (
            total_loss / total_batches
        )

        epoch_seconds = (
            time.perf_counter() - epoch_start
        )

        print(
            f"Epoch [{epoch:03d}/{config.epochs:03d}] "
            f"Loss: {average_loss:.4f} "
            f"Time: {epoch_seconds:.2f}s"
        )

        if average_loss < best_loss:
            best_loss = average_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                epoch=epoch,
                loss=average_loss
            )

            print(
                "  Best checkpoint saved:",
                config.model_path
            )

    total_seconds = (
        time.perf_counter() - training_start
    )

    print()
    print("Training completed.")
    print("Best loss :", f"{best_loss:.4f}")
    print("Model path:", config.model_path)
    print("Total time:", f"{total_seconds:.2f}s")


if __name__ == "__main__":
    train()
