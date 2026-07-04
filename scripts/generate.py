import argparse
from pathlib import Path

import torch

from mini_gpt.config import ModelConfig
from mini_gpt.model import MiniGPT
from mini_gpt.tokenizer import CharTokenizer


def _choose_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/latest.pt"),
    )
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = _choose_device()

    checkpoint = torch.load(args.checkpoint, map_location=device)

    tokenizer = CharTokenizer(tuple(checkpoint["vocab"]))
    model_cfg = ModelConfig(**checkpoint["model_cfg"])

    model = MiniGPT(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    x = torch.tensor(
        [tokenizer.encode(args.prompt)],
        dtype=torch.long,
        device=device,
    )
    generated_ids = model.generate(x)
    resp = tokenizer.decode(generated_ids[0].tolist())
    print(resp)


if __name__ == "__main__":
    main()
