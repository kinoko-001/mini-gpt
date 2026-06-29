import torch

from mini_gpt.config import DataConfig, ModelConfig, TrainConfig
from mini_gpt.data import load_data
from mini_gpt.metrics import CsvMetricsLogger
from mini_gpt.model import MiniGPT
from mini_gpt.training import Split, train

torch.manual_seed(1337)


# TODO: for generating later
def save_checkpoint(): ...


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


if __name__ == "__main__":
    main()
