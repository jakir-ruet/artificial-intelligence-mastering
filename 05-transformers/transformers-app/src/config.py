from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TransformerConfig:
    # Project paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_path: Path = project_root / "data" / "training_text.txt"
    model_path: Path = project_root / "models" / "mini_transformer.pth"

    # Dataset
    context_length: int = 128
    batch_size: int = 32

    # Transformer architecture
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    feed_forward_dim: int = 512
    dropout: float = 0.1

    # Training
    learning_rate: float = 3e-4
    epochs: int = 20
    random_seed: int = 42

    @property
    def device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")


config = TransformerConfig()

if __name__ == "__main__":
    print("Project root:", config.project_root)
    print("Data path:", config.data_path)
    print("Model path:", config.model_path)
    print("Device:", config.device)
    print("Embedding dimension:", config.embedding_dim)
    print("Attention heads:", config.num_heads)

    assert config.embedding_dim % config.num_heads == 0
