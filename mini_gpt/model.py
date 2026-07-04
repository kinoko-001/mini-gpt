import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_gpt.config import ModelConfig

# TODO: add dropout?


class Head(nn.Module):
    def __init__(self, cfg: ModelConfig, head_size: int):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(cfg.embd_dim, head_size)
        self.query = nn.Linear(cfg.embd_dim, head_size)
        self.value = nn.Linear(cfg.embd_dim, head_size)

        # Keep the causal mask with the module so it moves across devices.
        self.register_buffer(
            "tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )

    def forward(self, x: torch.Tensor):
        _, T, C = x.shape

        # (B, T, C) @ (C, head_size) -> (B, T, head_size)
        k: torch.Tensor = self.key(x)  # (B, T, head_size)
        q: torch.Tensor = self.query(x)

        affinities: torch.Tensor = q @ k.transpose(-2, -1)  # (B, T, T)

        # Scale before softmax to keep attention scores well-behaved.
        affinities *= self.head_size**-0.5

        # Causal masking: future positions get -inf.
        affinities = affinities.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )
        affinities = F.softmax(affinities, dim=-1)

        v = self.value(x)
        out = affinities @ v
        return out  # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(cfg, head_size) for _ in range(cfg.n_head)]
        )

        self.proj = nn.Linear(cfg.embd_dim, cfg.embd_dim)

    def forward(self, x: torch.Tensor):
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )  # (B, T , num_heads * head_size)

        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(cfg.embd_dim, 4 * cfg.embd_dim),
            nn.ReLU(),
            nn.Linear(4 * cfg.embd_dim, cfg.embd_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        head_size = cfg.embd_dim // cfg.n_head
        self.self_attention = MultiHeadAttention(cfg, head_size)
        self.ff = FeedForward(cfg)

        self.ln1 = nn.LayerNorm(cfg.embd_dim)
        self.ln2 = nn.LayerNorm(cfg.embd_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.embedding_table = nn.Embedding(cfg.vocab_size, cfg.embd_dim)
        self.positional_embedding = nn.Embedding(cfg.block_size, cfg.embd_dim)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln = nn.LayerNorm(cfg.embd_dim)
        self.lm_head = nn.Linear(cfg.embd_dim, cfg.vocab_size)

        self.cfg = cfg

    def _init_weights(self):
        pass

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape

        tok_emb = self.embedding_table(x)  # (B, T, C)

        pos = torch.arange(T, dtype=torch.long, device=x.device)
        pos_emb = self.positional_embedding(pos)  # (T, C)

        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        loss = None

        if y is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(
        self, x: torch.Tensor, max_new_tokens: int = 300
    ) -> torch.Tensor:
        # x = (B, T) tensor of IDs representing the current context
        for _ in range(max_new_tokens):
            # feed only the last `block_size` tokens, but keep the full output
            x_cond = x[:, -self.cfg.block_size :]
            logits, _ = self(x_cond)

            # take the logits from the last position
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)

            # Sample instead of always picking the highest-probability token.
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            x = torch.cat((x, next_token), dim=1)  # (B, T+1)

        return x  # (B, T + max_new_tokens)
