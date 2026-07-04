# Mini GPT Explained

This write-up should be able to explain this project to another student.

This project is a from-scratch tiny GPT implementation with some experiments, heavily inspired by Karpathy's tutorial. The goal is to understand transformer architecture, tensor shapes, training dynamics, memory usage, and basic profiling in PyTorch.

## Birds-eye view

The model implements next-character prediction.

We have some `input.txt` (a mini Shakespeare, from Karpathy's tutorial) as our training data. However, we want to turn this raw text into numbers that PyTorch can work with / the model can learn from.

Let's define a token = a single char. We map each token to an ID in a lookup table.
*(NOTE: GPTs most commonly use subwords as tokens)*

The model can't train on the whole text at once, so we train in batches.

At a high level:

```text
Input IDs: shape (B, T)
-> token embedding: "what token is this?"
-> position embedding: "where is / what position is this token in?"
-> add them: what token is this, and where is it?
-> self-attention: "which other tokens should this token look at?"
-> feed-forward layer: "now that I gathered context, how should I refine my own vector?"
-> logits: scores for the next token
```

## Shapes

Common shapes:

- `B` = batch size
- `T` = sequence length / block size / context length
- `C` = embedding dimension / channel size
- `V` = vocab size

Input IDs have shape (B, T).
Token and positional embeddings produce (B, T, C).
The final logits have shape (B, T, V).

## Data and batching

The model can't train on the whole text at once, so we train in batches.

Each batch contains random chunks of text:
- `x` = input characters
- `y` = the same characters shifted one step forward

So the model learns: given the current context, predict the next character.

## Token and Positional Embeddings

In a bigram model (the simplest language model where you only use the previous token to predict the next token), each token ID maps directly to logits.

In this transformer:

```text
token ID -> embedding vector -> processing -> logits
```

The embedding table has shape `(vocab_size, C)`, where C is the embedding dimension / channel size.

The first embedding vector is the starting representation for that token. Self-attention and later layers update it into a context-aware meaning.

Position matters too - the position of a token in a sequence contributes to its meaning, so we add positional embeddings:

```text
token embedding:    what token is this?
position embedding: where is this token?
sum:                what token is this, and where is it?
```

## Self-Attention

This is best explained in the original paper, *Attention is All You Need*.

A self-attention head uses query, key, and value projections.

For each token:
- query = what am I looking for?
- key = what information do I contain?
- value = what information do I pass on?

The attention scores (i.e. the "affinities" between each token) are:

```python
q @ k.T
```

Each score says how much one token should attend to another token.

We scale the scores by `head_size ** -0.5` so the values don't get too large and make the softmax later too sharp.

## Causal Masking

This is the rule that stops the model from looking ahead at future tokens while predicting the next token.

During training, it blocks each token from attending to tokens that come after it. This is done with a triangle-shaped mask (tensor) to hide the upper-right part of the attention matrix.

After masking, we use softmax to convert the scores into probabilities. The weights (probabilities) across the *key* dimension sum to 1.

## Multi-Head Attention

In language,

Multi-head attention splits the embedding space (`C`) into several smaller attention heads, lets each head learn a different way of looking at the input / context, then glues their results back together.

NOTE: This encourages heads to specialize (e.g. one head looks at spelling, another head looks at speaker / dialogue structure, ...), but does not guarantee it. Two heads can still learn similar things.

After concatenating heads, they are side-by-side and have no links to each other, so we use the projection layer *mixes* the head outputs back into one coherent token representation (like a coherent summary).

## Feed-Forward Layer

This layer processes the information inside each token vector.

One way to think about it:

```
take the C-dimensional token vector
-> ask 4C different learned "questions" about it
-> keep the useful answers
-> combine those answers back into a C-dimensional update
```

Important: the FF layer is applied independently to each token position - it doesn't look at the other token positions. Attention mixes information between tokens; in FF, each token individually refines its own vector.

## Transformer Block

Each block has:

```text
LayerNorm -> self-attention -> residual connection
LayerNorm -> feed-forward -> residual connection
```

Note this implementation uses Pre-LN:

```python
x = x + self_attention(layer_norm(x))
x = x + feed_forward(layer_norm(x))
```

Post-LN would be:

```python
x = layer_norm(x + sublayer(x))
```

Pre-LN is often easier to train in deeper transformers because gradients do not have to pass through as many layer norms on the residual path.

## Logits and Loss

After the transformer blocks, the model applies a final layer norm and `lm_head`, which is just a linear layer that produces the final logits.

The logits have shape `(B, T, V)`. For every batch item and every position, the model outputs a score for every possible next character (in the vocab).

If targets `y` are provided, we flatten logits and targets and use cross-entropy loss. The loss measures how surprised the model is by the true next character. Lower loss means the model assigned a higher probability to the correct next character.

## Generation

Generation starts with some prompt IDs.

For each new token:
1. Keep only the last `block_size` tokens as context.
2. Run the model.
3. Take logits from the last position.
4. Convert logits into probabilities with softmax.
5. Sample one next token from the probability distribution.
6. Append that token to the (input) sequence, to be used as the next "context".

NOTE: we are sampling, not always picking the highest-probability token.

## Notes / Current limitations
- This is a char-level GPT, not word-level or subword-level GPT.
- `dropout` exists in the config but is not implemented in the model yet.
- I also did some experiment for learning how tweaking certain hyperparameters would change the model and training. Experiment results are in `docs/experiments.md`; this file is for explaining how the model works.