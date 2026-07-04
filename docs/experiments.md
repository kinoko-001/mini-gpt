# Experiments

Goal: solidify my transformer understanding and intuition by testing changes to:
- context length
- width (embedding dim)
- depth (`n_layers`)
- attention head structure (`n_heads`)

on the mini character-level GPT.

## Fixed setup

Dataset: input.txt, char tokenizer, 90/10 train/val split
Training: batch=4, lr=1e-3, eval every 500 steps, 2000 steps

## Experiment summary

What changed | Setup | Main observation | Link to theory | So what
|---|---|---|---|---|
| Context | block_size=8/16/32, C=384 L=6 H=8 | Longer context helped | Context is useful for predicting the next tokens | Use block_size=32 for later experiments |
| Embedding dimension | C=128 vs 256, block_size=32, L=4, H=4 | C=256 only slightly better | Increasing C is supposed to be giving the model more capacity to represent each token, quadratically | Selecting a C will depend on compute / time budget |
| Depth (layers) | L=2/4/6, block_size=32, C=256, H=4 | L=2 actually was best | Increasing layers is supposed to increase the model's ability to learn complex hierarchical patterns, but our dataset was tiny and context length was short, L=2 may already be enough | Rerun L2 and maybe L3 (parameter count may have increased too much from L=2 to 4)? |
| Heads | H=1/2/4/8, block_size=32, C=256, L=4 | H=2/4/8 similar, H=1 did worse | Once we have a few heads, the model may already have enough attention diversity for our dataset | Pick H=2 for *this* config because it's enough and throughput is higher than H=4/8 (note the global best uses a different `H`) |

- `block=32, embd=256, layers=2, heads=4` is best logged run: lowest val loss, small final loss gap, very high throughput, plus it was actually one of the smaller models.
- for this small dataset and a 2000-step budget, extra layers did not help (NOTE: not generalizing that more layers doesn't help); L2 may be enough here or deeper models may need different training settings
- more heads are slower (much lower throughput / tokens_per_s) - bc heads are separate modules in a Py loop
- single attention head is too restrictive for the setup; but `H=2/4/8` achieved very similar training and val losses, which means additional heads beyond `H=2` did not provide meaningful extra modelling benefit. Since `H=2` also achieve higher throughput than `H=4/8`, it's the most efficient choice for *this* config

After these conclusions, we rerun 2-3 of the best, with more steps and `eval_iters`. Then pick one and evaluate it on a test set once.

### Why these configs?

The main goal of my experiments here is to *learn* and observe. So we want to maximise educational value of each experiment.

*NOTE: the baseline (`block_size=8,C=384,L=6,H=8`) is a fairly wide, deep model with a very short context window.*

Change context length first: more layers and heads are less educational until the model has enough context to attend over.

- NOTE: increasing context is not free; attention has a `T x T` matrix - this gets more expensive

Once we've tested context, we stop using `block=8` as the main setup - it's not enough to be effective. `block=32` is a nice middle ground: enough context to make attention interesting, not so much that it makes every run painful.

How many parameters are roughly in the transformer:
```
attention projections: about 4 * C^2
feedforward:           about 8 * C^2
total per block:       about 12 * C^2
```

so increasing `C` is roughly quadratic

```
C=128: 12 * 128^2  ~= 0.20M params/block
C=256: 12 * 256^2  ~= 0.79M params/block
C=384: 12 * 384^2  ~= 1.77M params/block
```

doubling layers is roughly doubling transformer block compute and parameters, but each layer gives another round of:
- attention
- mixing of features
- refining token representation

we check changes to the head structure separately bc with multi-head attention, embd_dim is split across the heads - fewer heads mean wider heads.

### Re-runs

*Ran with three different seeds, `block_size=32, embd_dim=256`*

| Config | Mean final val loss | Mean best val loss | Median toks/s | Params | Verdict |
|---|---:|---:|---:|---:|---|
| `L=2, H=4` | `1.6887 ± 0.0044` | `1.6887 ± 0.0044` | `3949` | `1.62M` | Pick |
| `L=4, H=2` | `1.7633 ± 0.0098` | `1.7368 ± 0.0067` | `3120` | `3.20M` | Not worth extra cost |
| `L=6, H=4` | `1.7665 ± 0.0049` | `1.7477 ± 0.0051` | `1451` | `4.78M` | Much slower, no gain |

*Then plotted out train vs val loss for each run - no clear signs of overfitting.*

Main observation is still: the 2-layer/4-head model performs best amongst the reruns. Potential reasons:

- the dataset (mini-Shakespeare) is small. 2 layers, 256 embedding dim may already be enough to capture many of the useful short-range patterns, especially with `block_size=32`.
- `batch_size=4` is very small -> update sees only 128 token positions -> makes the gradient noisy.
  - `lr=1e-3` may be fine for the smaller model but less ideal for deeper ones. The deeper models may not be “bad”; they may just need a different learning rate, schedule, warmup, dropout, or longer training.

*NOTE: dropout hasn't been implemented at this point yet*

### Future to-dos?

Implement / experiment with:
-  `batch_size` = how noisy / stable are updates, and how fast is hardware used?
- dropout: if too high, model may struggle to learn bc too much info is being randomly removed
- precision = controlling how much decimal detail the model keeps during calculations

## Analysis / plots

in `analysis.ipynb`.


## Some questions I had

- `best_val_loss == final_val_loss` does not prove it has not started to overfit

  The "loss gap has not widened (by the end of training)" - why is it not sufficient proof for "the model has not started to overfit"?

  If we use reasoning from first principles:
  - `train_loss`: how well the model fits the training sample
  - `val_loss`: an estimate of how well it generalizes to unseen data
  - `loss_gap = val_loss - train_loss`: a rough proxy for generalization gap

  a non-windening gap is not sufficient proof because:
  - val_loss is noisy; in our experiments i used `eval_iters=10` - each logged val_loss is based on a small random sample of batches, so a flat or small gap could just be noise.
  - model could still be worse on a separate test set
  - the gap is a symptom, not the definition; the definition of overfitting is worse expected generalization, not "gap got bigger"; the gap is just one way we try to detect it
  - train and val can both move; what matters most is the trend of `val_loss`, esp relative to its previous best

  so a better way to phrase the best (val) run is:

  > The model is not showing obvious overfitting yet; validation loss is still improving at the last logged point, so it may simply need more training.

- what does "noisy updates" mean?
  - to update a weights, we usually use a gradient estimate computed from a mini-batch instead of the whole dataset (`g_batch​ = g_true ​+ ϵ` where ϵ is gradient noise.)
- why optimization often becomes more fragile for deeper models
  - more layers / transformations, so update errors (from noisy gradients) can accumulate or amplify down the layers
  - often the case unless you adjust learning rate, schedule, warmup, dropout, or longer training
  - using the same learning rate = assuming the stepsize is equally safe for both models, which is not true for shallow vs deeper models
  - warmup = starting with a small LR and gradually increasing it
    - early gradients are often unreliable bc the network is randomly initialized
  - (learning-rate) schedule = how the LR changes over training, e.g. warmup, decay etc

## Experiment / learning notes

### Metrics we started off tracking

- `train_loss`, `val_loss`
- `loss_gap = val_loss - train_loss`
- `perplexity = exp(val_loss)` - less intuitive for char models, still useful
  - chars are not the units we usually think in when evaluating language quality
  - token-level perplexity is usually **more interpretable** than character-level perplexity because tokens are closer to the units where language starts carrying meaning
  - altho token-level is still an imperfect proxy for language quality
- `step_time_ms` - latency
  - can increase while `token_per_second` improves
  - bigger batches often make each step slower but processes more tokens per step
- `tokens_per_second` - throughput / how fast training is moving
  - useful for performance experiments like CPU vs MPS, float32 vs mixed precision, bigger batch size / context length
- `tokens_seen`
  - plotting `val_loss` vs `tokens_seen` is often more honest than plotting it vs `step`; use as fair x-axis when comparing batch sizes, context lengths, devices, or precision
    - bc `step` is not a unit of training data, it's a unit of optimizer updates; a model trained on small batches (vs big batches) would've seen less tokens by the same `step`, but are not equally trained models (bigger batches -> seen more tokens; so if its val loss is lower at step 500, may just be bc it's consumed much more data)
    - `tokens_seen` is fairer for tracking learning progress, `step` is useful for tracking optimizer behavior
- `mps_allocated_mb` - memory currently allocated by tensors on MPS
  - see how hyperparams affect tensor memory
- `parameters` - model capacity metric: number of trainable weights
- `device`, `dtype`, `batch_size`, `block_size`, `embd_dim`, `n_head`, `n_layer`

later - "why is this slow / unstable / memory hungry?" metrics:

- `mps_driver_mb` - memory allocated by the Metal driver / "pressure on the MPS"
  - includes overhead / caching - may stay high even after tensors are freed
- `grad_norm` - stability signal
  - if it explodes - learning rate may be too high or training is unstable
  - if tiny - learning may be weak or saturated
- `parameter_memory_mb` - raw memory used by the weights
  - NOTE: training uses much more than this bc we also have gradients, optimizer state, activations, temporary tensors etc
  - for AdamW, optimizer state alone is usually much larger than the parameter tensor memory

### Others

- Didn't spend much time analysing `grad_norm` because loss was improving


## Appendix

Summary stats for each run / config (in `analysis.ipynb`)

| index | cfg | best_val_loss | final_val_loss | final_train_loss | final_loss_gap | final_val_perplexity | max_tokens_seen | median_tokens_per_s | parameters | parameter_memory_mb | max_mps_allocated_mb |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | block=32, embd=256, layers=2, heads=4 | 1.934530 | 1.934530 | 1.875856 | 0.058674 | 6.920791 | 256000 | 3859.390793 | 1621056 | 6.183838 | 29.401367 |
| 5 | block=32, embd=256, layers=4, heads=2 | 2.020488 | 2.020488 | 1.864702 | 0.155786 | 7.542005 | 256000 | 3149.390386 | 3200576 | 12.209229 | 55.698975 |
| 7 | block=32, embd=256, layers=4, heads=8 | 2.023778 | 2.023778 | 1.868346 | 0.155432 | 7.566857 | 256000 | 1256.760093 | 3200576 | 12.209229 | 58.244141 |
| 6 | block=32, embd=256, layers=4, heads=4 | 2.027451 | 2.027451 | 1.860260 | 0.167191 | 7.594702 | 256000 | 2051.282660 | 3200576 | 12.209229 | 57.357910 |
| 1 | block=32, embd=128, layers=4, heads=4 | 2.036317 | 2.036317 | 1.833221 | 0.203096 | 7.662340 | 256000 | 2138.463463 | 813888 | 3.104736 | 16.224854 |
| 8 | block=32, embd=256, layers=6, heads=4 | 2.051404 | 2.051404 | 2.047226 | 0.004178 | 7.778817 | 256000 | 1410.861126 | 4780096 | 18.234619 | 85.302734 |
| 9 | block=32, embd=384, layers=6, heads=8 | 2.058080 | 2.144684 | 1.949562 | 0.195121 | 8.539340 | 384000 | 784.840413 | 10709056 | 40.851807 | 181.578369 |
| 2 | block=32, embd=128, layers=6, heads=8 | 2.084381 | 2.084381 | 1.972137 | 0.112244 | 8.039616 | 256000 | 871.850977 | 1210432 | 4.617432 | 26.528564 |
| 4 | block=32, embd=256, layers=4, heads=1 | 2.110833 | 2.110833 | 1.975203 | 0.135630 | 8.255112 | 256000 | 4255.329711 | 3200576 | 12.209229 | 56.616211 |
| 0 | block=16, embd=384, layers=6, heads=8 | 2.223586 | 2.223586 | 2.328996 | -0.105410 | 9.240409 | 192000 | 387.748601 | 10702912 | 40.828369 | 171.976074 |
| 10 | block=8, embd=384, layers=6, heads=8 | 2.346988 | 2.376045 | 2.314722 | 0.061323 | 10.762256 | 160000 | 200.579215 | 10699840 | 40.816650 | 167.691650 |
