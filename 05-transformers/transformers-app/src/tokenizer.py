from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CharacterTokenizer:
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        if not text:
            raise ValueError("Training text cannot be empty.")

        vocabulary = sorted(set(text))

        stoi = {
            character: index
            for index, character in enumerate(vocabulary)
        }

        itos = {
            index: character
            for character, index in stoi.items()
        }

        return cls(
            stoi=stoi,
            itos=itos
        )

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        unknown_characters = {
            character
            for character in text
            if character not in self.stoi
        }

        if unknown_characters:
            raise ValueError(
                "Unknown characters found: "
                f"{sorted(unknown_characters)}"
            )

        return [
            self.stoi[character]
            for character in text
        ]

    def decode(self, token_ids: List[int]) -> str:
        unknown_ids = {
            token_id
            for token_id in token_ids
            if token_id not in self.itos
        }

        if unknown_ids:
            raise ValueError(
                f"Unknown token IDs found: {sorted(unknown_ids)}"
            )

        return "".join(
            self.itos[token_id]
            for token_id in token_ids
        )


if __name__ == "__main__":
    from config import config

    if not config.data_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {config.data_path}"
        )

    training_text = config.data_path.read_text(
        encoding="utf-8"
    )

    tokenizer = CharacterTokenizer.from_text(
        training_text
    )

    encoded = tokenizer.encode(training_text)
    decoded = tokenizer.decode(encoded)

    print("Vocabulary size:", tokenizer.vocab_size)
    print("Total characters:", len(training_text))
    print("Total tokens:", len(encoded))
    print("First 50 token IDs:", encoded[:50])
    print("Decoded preview:")
    print(decoded[:100])

    assert decoded == training_text
