from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharTokenizer:
    vocab: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        vocab = tuple(sorted(set(text)))
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        stoi = {ch: i for i, ch in enumerate(self.vocab)}
        return [stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)
