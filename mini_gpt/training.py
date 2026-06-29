import time
from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn
from tqdm import tqdm

from mini_gpt.config import ModelConfig, TrainConfig
from mini_gpt.metrics import (
    CsvMetricsLogger,
    build_metrics_row,
    grad_norm,
)

Split = Literal["train", "val"]
BatchFn = Callable[[Split], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    grad_norm: float
    tokens: int
    step_time_s: float


# During evaluation, we're not updating weights - we only want to measure loss
# so no need to build the gradient computation graph here
@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    get_batch: BatchFn,
    eval_iters: int,
) -> dict[Split, float]:
    """Checks the how well the model is doing on both the training data and the
    val data, without training the model during that check.

    I.e. Runs the model in eval mode over several random batches, records the loss
    for each batch, and returns the avg loss for each split.

    """
    was_training = model.training
    model.eval()

    losses: dict[Split, float] = {}

    for split in ("train", "val"):
        split_losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)

            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses)

    model.train(was_training)
    return losses


def _should_eval(step: int, eval_interval: int):
    return step == 1 or step % eval_interval == 0


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    get_batch: BatchFn,
) -> TrainStepResult:
    step_start = time.perf_counter()

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    # clear old gradients
    optimizer.zero_grad(set_to_none=True)
    # calculate grads
    loss.backward()
    g_norm = grad_norm(model)
    # update weights
    optimizer.step()

    step_time = time.perf_counter() - step_start

    return TrainStepResult(
        loss=loss.item(),
        grad_norm=g_norm,
        tokens=xb.numel(),
        step_time_s=step_time,
    )


def train(
    model: nn.Module,
    get_batch: BatchFn,
    train_config: TrainConfig,
    model_config: ModelConfig,
    metrics_logger: CsvMetricsLogger | None = None,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate
    )

    progress = tqdm(range(1, train_config.max_iters + 1), desc="Training")

    tokens_seen = 0
    tokens_since_log = 0
    secs_since_log = 0.0
    steps_since_log = 0

    for step in progress:
        result = train_step(model, optimizer, get_batch)

        tokens_seen += result.tokens
        tokens_since_log += result.tokens
        secs_since_log += result.step_time_s
        steps_since_log += 1

        if _should_eval(step, train_config.eval_interval):
            losses = estimate_loss(model, get_batch, train_config.eval_iters)

            if metrics_logger is not None:
                row = build_metrics_row(
                    model=model,
                    train_cfg=train_config,
                    model_cfg=model_config,
                    step=step,
                    losses=losses,
                    tokens_seen=tokens_seen,
                    tokens_per_second=tokens_since_log / secs_since_log,
                    step_time_ms=1000 * secs_since_log / steps_since_log,
                    grad_norm=result.grad_norm,
                )
                metrics_logger.write(row)

            progress.set_postfix(
                train=f"{losses['train']:.3f}",
                val=f"{losses['val']:.3f}",
            )

            tokens_since_log = 0
            secs_since_log = 0.0
            steps_since_log = 0

    print("Trained !")
