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

        # lower triangle of 1s
        # store the matrix on the module:
        # - it's not a learned parameter - never gets gradients or updates in training
        # but still part of a module's state:
        # - moves to GPU too
        # - saved / loaded with state_dict
        self.register_buffer(
            "tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )

    def forward(self, x: torch.Tensor):
        _, T, C = x.shape

        # (B, T, C) @ (C, head_size) -> (B, T, head_size)
        k: torch.Tensor = self.key(x)  # (B, T, head_size)
        q: torch.Tensor = self.query(x)

        # affinities / attention scores
        # so each wei[i, j] = how much (query) token i attends to (key) token j
        # if flipped the other way the meaning of the matrix will be different
        affinities: torch.Tensor = q @ k.transpose(-2, -1)  # (B, T, T)

        # scaling - keep variance near 1, so not all weight is on one token
        # -> soft-max is well-behaved, trainable
        affinities *= self.head_size**-0.5

        # causal masking - future positions get -inf
        affinities = affinities.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )
        # convert scores into probabilities / weights
        # all values along dim=-1 (key) sum to 1
        affinities = F.softmax(affinities, dim=-1)

        v = self.value(x)
        out = affinities @ v
        return out  # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    # in language, understanding a token needs to look at different things at the same itme

    # split the embeddings pace into several smaller attention heads,
    # let each head learn a different way of looking at the context,
    # then glue their results back together

    # possible for 2+ heads to learn very similar things
    # splitting into multiple heads encourages each head to specialize, but does
    # not guarantee it
    # -> sometimes can prune / remove certain attention heads with little damage
    def __init__(self, cfg: ModelConfig, head_size: int):
        super().__init__()
        # Using nn.ModuleList registers the params of every child (Head)
        # - so optimizer.step() updates those parameters
        self.heads = nn.ModuleList(
            [Head(cfg, head_size) for _ in range(cfg.n_head)]
        )

        # this projection / mixing matrix = "how much should I use each head feature?""
        self.proj = nn.Linear(cfg.embd_dim, cfg.embd_dim)

    def forward(self, x: torch.Tensor):
        # for each (batch, token_seq) slot, glue each head summary together
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )  # (B, T , num_heads * head_size)

        # but after concat, there's not interaction between heads
        # we want the model to mix information across heads
        # so we use a projection = a learned linear map where every output
        # becomes one coherent summary
        # out @ proj_matrix + b
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.net = nn.Sequential(
            # for an update:
            # temporarily expand a token into more learned feature checks,
            # keep the useful signals,
            # and compresses them back to the original embedding size
            # another explanation:
            # take the C-dimensional token vector
            # -> ask 4C different learned “questions” about it
            # -> keep only the useful activated answers
            # -> combine those answers back into a C-dimensional update
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
        # NOTE: we create a new tensor instead of modifying inplace with x=,
        # because PyTorch needs to remember certain tensor values so it can
        # compute gradients later during backprop
        # mutating a tensor with += may be overwriting a value that PyTorch needs

        # Post-LN:
        # x = ln(x + sublayer(x))
        # but for deep transformers it can be hard to train bc gradients have
        # to pass through many ln operations

        # Pre-LN here:
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        # in bigram, each token ID maps to a logit
        # but in transformer, token ID -> embedding vector -> processing -> logits
        # embedding table has shape (vocab_size, C) where C = embedding dim / channel size
        # bigger embedding dim gives them model more room to describe each
        # token and its context

        # first embedding vector = starting guess / ID for that token
        # the self-attention and later layers update it into context-aware meaning

        # add positional embeddings - position matters in meaning too
        # one vector per position -> (T, C)
        # position 0 -> vector of length C
        # position 1 -> vector of length C
        # ...

        # pytorch broadcasting makes x = tok_emb + pos_emb (B, T, C) shape

        # C = like the number of "slots" (channels) the model has to describe a token
        # for example if C = 6, cat can loosely encode things like:
        # cat → [
        #   animal-ish,
        #   noun-ish,
        #   small-pet-ish,
        #   can-be-subject-ish,
        #   related-to-dog-ish,
        #   related-to-meow-ish
        # ]

        # each row stores the learnable vector of one token
        self.embedding_table = nn.Embedding(cfg.vocab_size, cfg.embd_dim)
        # each position (in a sequence) -> learnable vector
        self.positional_embedding = nn.Embedding(cfg.block_size, cfg.embd_dim)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln = nn.LayerNorm(cfg.embd_dim)
        # produces logits
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

    def generate(
        self, x: torch.Tensor, max_new_tokens: int = 10
    ) -> torch.Tensor:
        # x = (B, T) tensor of IDs representing the current context
        for _ in range(max_new_tokens):
            # keep only the last `block_size` tokens from each sequence
            x = x[:, -self.cfg.block_size :]
            logits, loss = self(x)

            # take the logits from the last position
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)

            # sample one from the probability distribution
            # NOTE: we're do SAMPLING, not always picking the highest-probability token
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            x = torch.cat((x, next_token), dim=1)  # (B, T+1)

        return x  # (B, T + max_new_tokens)


# the model can't train on the whole text at once, we want to train in batches
# 1. how to predict the next char using only the current one as context?

# ff layer = processing the info inside each token vector

# Input IDs (B, T)
# Token embedding - "what token is this?"
# position embedding - "Where is / what position is this token in?"
# Add them - what token is this, and where is it?
# Self-attention = "which other tokens should this token look at?"
# FF - "Now that I gathered context, how should I refine my own vector?"

# self-attention head: query, key, value
# causal masking - the rule that stops the model from "looking ahead" at future
# words while it's predicting the next word
# during training, it blocks each (token) position from attending to tokens that
# come after it - done by hiding the upper-right part of the attention matrix
# - a triangle-shaped mask

# multi-head attention
