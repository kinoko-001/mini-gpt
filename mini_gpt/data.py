from dataclasses import dataclass
from pathlib import Path

import torch

from mini_gpt.config import DataConfig
from mini_gpt.tokenizer import CharTokenizer
from mini_gpt.training import Split


def _load_text(path: Path) -> str:
    with open(path, "r") as f:
        text = f.read()

    return text


@dataclass
class TextDataset:
    train_data: torch.Tensor
    val_data: torch.Tensor
    tokenizer: CharTokenizer
    device: torch.device

    def get_batch(
        self, split: Split, block_size: int, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns input chunks and target chunks (shifted one character forward)

        Example:

            x:
            [
                [h, e, l, l, o],
                [w, o, r, l, d]
            ]

            y:
            [
                [e, l, l, o, _],
                [o, r, l, d, _]
            ]

        """
        data_source = self.train_data if split == "train" else self.val_data

        # choose batch_size many random starting positions
        ix = torch.randint(len(data_source) - block_size, (batch_size,))

        x = torch.stack([data_source[i : i + block_size] for i in ix])
        y = torch.stack([data_source[i + 1 : i + 1 + block_size] for i in ix])

        return x.to(self.device), y.to(self.device)


def load_data(cfg: DataConfig, device: torch.device) -> TextDataset:
    text = _load_text(cfg.input_path)

    tokenizer = CharTokenizer.from_text(text)

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(cfg.train_split * len(data))
    return TextDataset(
        train_data=data[:n],
        val_data=data[n:],
        tokenizer=tokenizer,
        device=device,
    )
