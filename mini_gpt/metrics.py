import csv
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from mini_gpt.config import ModelConfig, TrainConfig

MetricValue = str | int | float | None
MetricRow = dict[str, MetricValue]

MemoryMb = float | None


def _bytes_to_mib(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        # some parameters may be frozen - this is useful for transfer learning
        # - where you only train the final layers
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def parameter_memory_mb(model: nn.Module) -> float:
    return _bytes_to_mib(
        sum(p.numel() * p.element_size() for p in model.parameters())
    )


def grad_norm(model: nn.Module) -> float:
    """Computes gradient norm - a value that summarizes "how large" all the gradients
    in the model are after backprop. I.e. all the gradient magnitudes compressed
    into one scalar.

    A very large `grad_norm` can suggest exploding gradients
    A very small `grad_norm` can suggest vanishing gradients
    """
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue

        norm = p.grad.detach().norm(2).item()
        total_sq += norm**2
    return total_sq**0.5


def perplexity(loss: float) -> float:
    return math.exp(loss)


def mps_memory_mb() -> tuple[MemoryMb, MemoryMb]:
    """Returns MPS allocated and driver-allocated memory in MiB.

    Returns:
        `(allocated_mb, driver_mb)`, or `(None, None)` when MPS is unavailable.
    """
    if not torch.backends.mps.is_available():
        return None, None

    return (
        _bytes_to_mib(torch.mps.current_allocated_memory()),
        _bytes_to_mib(torch.mps.driver_allocated_memory()),
    )


def build_metrics_row(
    *,
    model: nn.Module,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    step: int,
    losses: dict[str, float],
    tokens_seen: int,
    tokens_per_second: float,
    step_time_ms: float,
    grad_norm: float,
) -> MetricRow:
    mps_allocated_mb, mps_driver_mb = mps_memory_mb()

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": train_cfg.run_name,
        "step": step,
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "loss_gap": losses["val"] - losses["train"],
        "val_perplexity": math.exp(losses["val"]),
        "tokens_seen": tokens_seen,
        "tokens_per_second": tokens_per_second,
        "step_time_ms": step_time_ms,
        "grad_norm": grad_norm,
        "mps_allocated_mb": mps_allocated_mb,
        "mps_driver_mb": mps_driver_mb,
        "parameters": count_parameters(model),
        "parameter_memory_mb": parameter_memory_mb(model),
        "device": train_cfg.device,
        "batch_size": train_cfg.batch_size,
        "block_size": model_cfg.block_size,
        "embd_dim": model_cfg.embd_dim,
        "n_head": model_cfg.n_head,
        "n_layer": model_cfg.n_layer,
        "learning_rate": train_cfg.learning_rate,
    }


class CsvMetricsLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: list[str] | None = None

    def write(self, row: dict[str, int | float]) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row)

        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
