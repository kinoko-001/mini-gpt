from dataclasses import asdict
from pathlib import Path
from shutil import copy2

import torch
import torch.nn as nn

from mini_gpt.config import DataConfig, ModelConfig, TrainConfig
from mini_gpt.data import load_data
from mini_gpt.metrics import CsvMetricsLogger
from mini_gpt.model import MiniGPT
from mini_gpt.tokenizer import CharTokenizer
from mini_gpt.training import Split, train

torch.manual_seed(1337)

CHECKPOINT_DIR = Path("checkpoints")


def save_checkpoint(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    model: nn.Module,
    tokenizer: CharTokenizer,
) -> None:
    # .pt = PyTorch model / checkpoint file
    # contains a model’s learned parameters, a full model object, or a training checkpoint
    path = CHECKPOINT_DIR / f"{train_cfg.run_name}.pt"
    latest_checkpoint = CHECKPOINT_DIR / "latest.pt"

    path.parent.mkdir(parents=True, exist_ok=True)

    # convention: save the min pieces needed to reconstruct the model later
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_cfg": asdict(model_cfg),
            "train_cfg": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in asdict(train_cfg).items()
            },
            "vocab": list(tokenizer.vocab),
        },
        path,
    )

    copy2(path, latest_checkpoint)

    print(f"Saved checkpoint to {path}")


def main() -> None:
    train_cfg = TrainConfig()
    data_cfg = DataConfig()
    device = torch.device(train_cfg.device)

    data = load_data(data_cfg, device=device)
    model_cfg = ModelConfig(vocab_size=data.tokenizer.vocab_size)
    model = MiniGPT(model_cfg).to(device)

    def _batch_fn(split: Split):
        return data.get_batch(split, model_cfg.block_size, train_cfg.batch_size)

    logger = CsvMetricsLogger(train_cfg.log_path)
    train(model, _batch_fn, train_cfg, model_cfg, logger)

    save_checkpoint(train_cfg, model_cfg, model, data.tokenizer)


if __name__ == "__main__":
    main()
