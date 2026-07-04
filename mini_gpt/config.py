from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    max_iters: int = 10000
    learning_rate: float = 1e-3
    batch_size: int = 4
    eval_interval: int = 250
    eval_iters: int = 100
    device: str = "mps"
    run_name: str = field(
        default_factory=lambda: dt.now().strftime("run_%Y%m%d_%H%M%S")
    )
    log_path: Path | None = None

    def __post_init__(self):
        object.__setattr__(
            self, "log_path", Path("logs") / f"{self.run_name}.csv"
        )


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 32
    embd_dim: int = 256
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.embd_dim % self.n_head != 0:
            raise ValueError("embd_dim must be divisible by n_head")


@dataclass(frozen=True)
class DataConfig:
    input_path: Path = Path("input.txt")
    train_split: float = 0.9
