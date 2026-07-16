# Load checkpoint
# → Rebuild tokenizer
# → Rebuild model
# → Encode prompt
# → Predict one token
# → Append token
# → Repeat
# → Decode generated text

import argparse
from pathlib import Path
from typing import Any

import torch

from config import config
from tokenizer import CharacterTokenizer
from transformer import MiniTransformer


def load_checkpoint(
    model_path: Path
) -> dict[str, Any]:

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=config.device,
        weights_only=False
    )

    required_keys = {
        "model_state_dict",
        "tokenizer_stoi",
        "model_config"
    }

    missing_keys = required_keys.difference(
        checkpoint.keys()
    )

    if missing_keys:
        raise ValueError(
            "Checkpoint is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    return checkpoint


def restore_tokenizer(
    checkpoint: dict[str, Any]
) -> CharacterTokenizer:

    stoi = checkpoint["tokenizer_stoi"]

    if not isinstance(stoi, dict) or not stoi:
        raise ValueError(
            "Invalid tokenizer vocabulary in checkpoint."
        )

    itos = {
        token_id: character
        for character, token_id in stoi.items()
    }

    return CharacterTokenizer(
        stoi=stoi,
        itos=itos
    )


def restore_model(
    checkpoint: dict[str, Any]
) -> MiniTransformer:

    model_config = checkpoint["model_config"]

    model = MiniTransformer(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        embedding_dim=model_config["embedding_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        feed_forward_dim=model_config[
            "feed_forward_dim"
        ],
        dropout=model_config["dropout"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(config.device)
    model.eval()

    return model


def apply_top_k(
    logits: torch.Tensor,
    top_k: int | None
) -> torch.Tensor:

    if top_k is None:
        return logits

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    top_k = min(
        top_k,
        logits.size(-1)
    )

    top_values, _ = torch.topk(
        logits,
        k=top_k,
        dim=-1
    )

    cutoff = top_values[:, -1].unsqueeze(-1)

    return logits.masked_fill(
        logits < cutoff,
        float("-inf")
    )


@torch.no_grad()
def generate_text(
    model: MiniTransformer,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = 20
) -> str:

    if not prompt:
        raise ValueError(
            "Prompt cannot be empty."
        )

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be greater than zero."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than zero."
        )

    prompt_token_ids = tokenizer.encode(
        prompt
    )

    generated = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long,
        device=config.device
    )

    for _ in range(max_new_tokens):

        # Keep only the latest context window.
        input_tokens = generated[
            :,
            -model.context_length:
        ]

        logits, _ = model(
            token_ids=input_tokens
        )

        # Use logits from the final sequence position.
        next_token_logits = logits[:, -1, :]

        # Temperature scaling.
        next_token_logits = (
            next_token_logits / temperature
        )

        # Restrict sampling to the most likely tokens.
        next_token_logits = apply_top_k(
            next_token_logits,
            top_k
        )

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

    generated_token_ids = (
        generated[0]
        .detach()
        .cpu()
        .tolist()
    )

    return tokenizer.decode(
        generated_token_ids
    )


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate text using the trained "
            "mini Transformer."
        )
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Artificial",
        help="Starting text for generation."
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Number of new tokens to generate."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help=(
            "Sampling temperature. "
            "Lower values are more deterministic."
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help=(
            "Sample only from the top-k tokens. "
            "Use 0 to disable top-k filtering."
        )
    )

    return parser.parse_args()


def main() -> None:

    arguments = parse_arguments()

    checkpoint = load_checkpoint(
        config.model_path
    )

    tokenizer = restore_tokenizer(
        checkpoint
    )

    model = restore_model(
        checkpoint
    )

    top_k = (
        arguments.top_k
        if arguments.top_k > 0
        else None
    )

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=arguments.prompt,
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_k=top_k
    )

    print("Generation Configuration")
    print("------------------------")
    print("Device         :", config.device)
    print("Prompt         :", arguments.prompt)
    print(
        "Max new tokens :",
        arguments.max_new_tokens
    )
    print(
        "Temperature    :",
        arguments.temperature
    )
    print("Top-k          :", top_k)
    print()

    print("Generated Text")
    print("--------------")
    print(generated_text)


if __name__ == "__main__":
    main()
