The training loops we wrote in chapters 5–7 were deliberately minimal — just enough to compute a loss, backpropagate, and step the optimizer. Real LLM training adds a few **stabilizing tricks** on top, and this appendix covers the three most important: **learning-rate warmup**, **cosine decay**, and **gradient clipping**.

None of them changes the model architecture. They only change *how* the optimizer moves through the loss landscape — easing in gently at the start, coasting to a near-stop at the end, and capping any single update that tries to leap too far. Together they make training large models far less likely to blow up. We'll build each piece in isolation, then fold all three into a single upgraded `train_model` function.

::: objectives "What you'll learn"
- Why a **learning-rate warmup** stabilizes the first, most fragile steps of training
- How **cosine decay** smoothly anneals the learning rate toward (almost) zero
- What **gradient clipping** does, and how the L2 norm of the gradients is rescaled
- How to combine all three into one production-grade `train_model` training function
:::

To keep this appendix self-contained, we reinitialize the same GPT model from chapter 5 and rebuild its data loaders on "The Verdict" short story:

```python title="Reinitialize the chapter-5 model" collapsible
import torch
from chapter04 import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-key-value bias
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
model.eval()
```

```python title="Rebuild the train / validation data loaders" collapsible
from previous_chapters import create_dataloader_v1

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    text_data[:split_idx],
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    text_data[split_idx:],
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
```

## D.1 Learning rate warmup

A **learning-rate warmup** gradually increases the learning rate from a very low initial value (`initial_lr`) up to a user-specified maximum (`peak_lr`) over the first handful of steps. Starting with tiny weight updates lowers the risk of a large, destabilizing step early on — when the randomly initialized model is at its most fragile.

::: callout analogy "Easing the car forward before flooring it"
A warmup is like pulling onto the highway: you don't slam the accelerator while still in the driveway. You ease the car forward, build up speed gradually, and *then* press down once you're safely up to pace. Hitting a freshly initialized network with the full learning rate is like flooring a cold engine — you risk stalling (diverging) before you've gone anywhere.
:::

Suppose we plan to train for 15 epochs, ramping from an initial rate of 0.0001 up to a peak of 0.01:

```python title="Warmup hyperparameters"
n_epochs = 15
initial_lr = 0.0001
peak_lr = 0.01
warmup_steps = 20
```

The number of warmup steps is usually set between **0.1% and 20%** of the total number of training steps:

```python title="Sizing the warmup as a fraction of all steps"
total_steps = len(train_loader) * n_epochs
warmup_steps = int(0.2 * total_steps)   # 20% warmup
print(warmup_steps)
# 27
```

We can now sketch a training-loop template that increments the learning rate linearly while we're still inside the warmup window, then holds it at the peak:

```python title="Linear warmup inside the training loop"
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)

# This increment is how much we raise initial_lr at each warmup step
lr_increment = (peak_lr - initial_lr) / warmup_steps
global_step = -1
track_lrs = []

for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        global_step += 1

        if global_step < warmup_steps:               # still warming up
            lr = initial_lr + global_step * lr_increment
        else:
            lr = peak_lr

        for param_group in optimizer.param_groups:    # apply lr to optimizer
            param_group["lr"] = lr
        track_lrs.append(optimizer.param_groups[0]["lr"])
        # (loss + model update omitted here for simplicity)
```

::: callout note "Why update `param_group['lr']` directly?"
PyTorch optimizers store the learning rate per *parameter group*. Setting `param_group["lr"]` before each `optimizer.step()` is the lowest-level way to drive a custom schedule by hand — exactly what `torch.optim.lr_scheduler` classes do for you under the hood.
:::

Plotting `track_lrs` against the step number shows the rate climbing for the first 20 steps and then flattening at 0.01.

::: diagram appD-lr-schedule "A learning-rate schedule: a short linear warmup ramps the rate up to its peak, then cosine decay curves it smoothly down toward a small minimum by the end of training."
:::

## D.2 Cosine decay

After warmup, we don't simply hold the learning rate constant — we slowly **decay** it. **Cosine decay** modulates the rate so that, after the warmup stage, it follows the shape of a half-cosine curve, gliding from the peak down to nearly zero by the end of training.

A gradually shrinking learning rate decelerates the weight updates in the later phases. That matters because it minimizes the risk of **overshooting** the loss minimum once we're close to it — keeping training stable as it converges.

::: callout analogy "Gently coasting to a stop"
Cosine decay is like easing off the gas as you approach a parking spot. Early on you move quickly (high learning rate), but as you near the target you slow down smoothly so you don't overshoot and have to back up. A constant learning rate is like driving full-speed into the spot and slamming the brakes — jarring, and you'll likely overrun the mark.
:::

We extend the loop: during warmup we ramp linearly as before; afterward we compute a `progress` fraction (0 → 1 across the post-warmup steps) and feed it through a cosine:

```python title="Linear warmup followed by cosine annealing"
import math

min_lr = 0.1 * initial_lr
track_lrs = []
lr_increment = (peak_lr - initial_lr) / warmup_steps
global_step = -1

for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        global_step += 1

        if global_step < warmup_steps:                         # linear warmup
            lr = initial_lr + global_step * lr_increment
        else:                                                  # cosine annealing
            progress = ((global_step - warmup_steps) /
                        (total_training_steps - warmup_steps))
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        track_lrs.append(optimizer.param_groups[0]["lr"])
```

::: callout math "The cosine decay formula"
For a progress fraction $p \in [0, 1]$ (where $p=0$ is the end of warmup and $p=1$ is the last step), the learning rate is

$$\eta(p) = \eta_{\min} + (\eta_{\text{peak}} - \eta_{\min}) \cdot \tfrac{1}{2}\left(1 + \cos(\pi p)\right).$$

At $p=0$, $\cos(0)=1$, so the term in parentheses is $\tfrac{1}{2}(1+1)=1$ and $\eta = \eta_{\text{peak}}$. At $p=1$, $\cos(\pi)=-1$, so it becomes $\tfrac{1}{2}(1-1)=0$ and $\eta = \eta_{\min}$. The half-cosine sweeps smoothly between the two — fast at first, gentlest near the end.
:::

Plotting again confirms the shape: a linear ramp for the first 20 steps, then a smooth cosine glide down to the minimum.

## D.3 Gradient clipping

**Gradient clipping** caps how large a single update can be. It sets a threshold above which gradients are **downscaled** to a fixed maximum magnitude, ensuring backpropagation never pushes the parameters by an unmanageably large step. This is the primary defense against **exploding gradients**.

Concretely, PyTorch's `clip_grad_norm_` with `max_norm=1.0` ensures the **norm** of the gradients does not exceed 1.0. "Norm" here means the gradient vector's length — specifically the **L2 (Euclidean) norm**.

::: callout analogy "A speed limiter on the engine"
Gradient clipping is a speed limiter. No matter how hard the throttle is pressed, the governor caps the car at, say, 70 mph — preventing a wild, dangerous lurch. A single batch can occasionally produce a freakishly large gradient (a steep cliff in the loss surface); clipping caps that update so one bad step can't fling the weights into nonsense.
:::

::: callout math "Computing the L2 norm and the clip factor"
For a vector $v = [v_1, v_2, \ldots, v_n]$, the L2 norm is

$$\lVert v \rVert_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}.$$

The same applies to a gradient matrix. Take

$$G = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}, \qquad \lVert G \rVert_2 = \sqrt{1^2 + 2^2 + 2^2 + 4^2} = \sqrt{25} = 5.$$

Since $\lVert G \rVert_2 = 5$ exceeds our `max_norm` of 1, we scale every entry by the factor $\text{max\_norm} / \lVert G \rVert_2 = 1/5$, giving the clipped matrix

$$G' = \begin{bmatrix} 0.2 & 0.4 \\ 0.4 & 0.8 \end{bmatrix}, \qquad \lVert G' \rVert_2 = 1.$$
:::

::: diagram appD-grad-clip "Gradient clipping: a vector whose L2 norm spikes past the max-norm threshold is uniformly rescaled so its norm equals exactly the threshold — its direction is preserved, only its length is capped."
:::

To see it in action, we initialize a fresh model, compute a loss for one batch, and backpropagate so PyTorch fills each parameter's `.grad`:

```python title="Backprop one batch, then inspect gradients"
from chapter05 import calc_loss_batch

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

loss = calc_loss_batch(input_batch, target_batch, model, device)
loss.backward()
```

A small utility scans every parameter's `.grad` to find the single largest gradient value:

```python title="Find the highest gradient value"
def find_highest_gradient(model):
    max_grad = None
    for param in model.parameters():
        if param.grad is not None:
            grad_values = param.grad.data.flatten()
            max_grad_param = grad_values.max()
            if max_grad is None or max_grad_param > max_grad:
                max_grad = max_grad_param
    return max_grad

print(find_highest_gradient(model))
# tensor(0.0411)
```

Now apply clipping and re-check — the largest value drops substantially:

```python title="Clip to max_norm=1.0 and re-check"
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(find_highest_gradient(model))
# tensor(0.0185)
```

::: callout warning "Clip *after* backward, *before* step"
The order is mandatory: `loss.backward()` must populate the gradients first, then `clip_grad_norm_` rescales them in place, and only then does `optimizer.step()` apply the (now-capped) update. Clip in the wrong place and you either clip stale gradients or clip nothing at all.
:::

## D.4 The modified training function

Finally, we fold all three techniques into an upgraded `train_model` (compare with `train_model_simple` from chapter 5). The changes: it derives `peak_lr` from the optimizer, applies **linear warmup** then **cosine decay** to the learning rate each step, and applies **gradient clipping** once the warmup phase is over.

```python title="The full train_model with warmup + cosine + clipping" collapsible
from chapter05 import evaluate_model, generate_and_print_sample

def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6):

    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # Retrieve the initial learning rate from the optimizer (= peak lr)
    peak_lr = optimizer.param_groups[0]["lr"]
    # Total number of iterations in the training process
    total_training_steps = len(train_loader) * n_epochs
    # Learning-rate increment during the warmup phase
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # Adjust the lr based on the current phase (warmup or cosine)
            if global_step < warmup_steps:
                lr = initial_lr + global_step * lr_increment
            else:
                progress = ((global_step - warmup_steps) /
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress))

            # Apply the calculated learning rate to the optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # Apply gradient clipping after warmup to avoid exploding gradients
            if global_step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

            optimizer.step()
            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        generate_and_print_sample(
            model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen, track_lrs
```

::: diagram appD-three-tricks "The three training enhancements at a glance: warmup ramps the learning rate in, cosine decay curves it back down, and gradient clipping caps any oversized update — all feeding one optimizer step."
:::

With the function defined, we train just like before — only now passing `warmup_steps` and the initial/min learning rates. Note that `peak_lr` is set via the optimizer's `lr`:

```python title="Pretrain using the enhanced training function" collapsible
import tiktoken

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

peak_lr = 5e-4
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
tokenizer = tiktoken.get_encoding("gpt2")

n_epochs = 15
train_losses, val_losses, tokens_seen, lrs = train_model(
    model, train_loader, val_loader, optimizer, device, n_epochs=n_epochs,
    eval_freq=5, eval_iter=1, start_context="Every effort moves you",
    tokenizer=tokenizer, warmup_steps=warmup_steps,
    initial_lr=1e-5, min_lr=1e-5
)
```

On a MacBook Air this finishes in about 5 minutes. As with plain pretraining, the model overfits after a few epochs (the dataset is tiny and we loop over it many times), but the falling training loss confirms the enhanced loop works:

```text title="Training output (abridged)"
Ep 1 (Iter 000000): Train loss 10.934, Val loss 10.939
Ep 1 (Iter 000005): Train loss  9.151, Val loss  9.461
...
Ep 15 (Iter 000130): Train loss 0.041, Val loss 6.915
Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him
vindicated--and by me!"  He laughed again, and threw back his head to look up ...
```

For a fairer test, you're encouraged to rerun this on a larger corpus and compare the enhanced loop against `train_model_simple`.

## Key takeaways

::: takeaways
- **Learning-rate warmup** linearly ramps the rate from a small `initial_lr` to `peak_lr` over the first few steps (typically 0.1–20% of all steps), avoiding destabilizing updates while the model is fragile.
- **Cosine decay** anneals the rate after warmup along a half-cosine curve toward a small `min_lr`, using $\eta = \eta_{\min} + (\eta_{\text{peak}} - \eta_{\min})\cdot\tfrac{1}{2}(1+\cos(\pi p))$, to avoid overshooting the loss minimum late in training.
- **Gradient clipping** rescales gradients whose **L2 norm** exceeds `max_norm` (e.g., 1.0) by the factor `max_norm / ‖g‖₂`, preserving their direction while capping their magnitude — the main guard against exploding gradients.
- Order matters: clip **after** `loss.backward()` and **before** `optimizer.step()`.
- The upgraded `train_model` combines all three; none of them touches the model architecture — they only shape the optimizer's trajectory.
:::

## Additional references

::: refs
- [Appendix D code (appendix-D.ipynb)](https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-D) — GitHub · the complete, runnable notebook for this appendix.
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) — Paper · Loshchilov & Hutter, the origin of cosine-annealed learning-rate schedules.
- [torch.optim.lr_scheduler docs](https://pytorch.org/docs/stable/optim.html) — Docs · built-in PyTorch schedulers, including `CosineAnnealingLR` and `LinearLR` for warmup.
- [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) — Docs · the exact gradient-clipping function used here.
- [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063) — Paper · Pascanu et al., which introduced norm-based gradient clipping for exploding gradients.
- [How to Avoid Exploding Gradients With Gradient Clipping](https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/) — Blog · an accessible walkthrough of clip-by-norm vs. clip-by-value.
:::

## Test your knowledge

```flashcards
Q: What does a learning-rate warmup do, and why?
A: It linearly increases the learning rate from a small `initial_lr` to `peak_lr` over the first steps, so early updates on the fragile, freshly initialized model stay small and don't destabilize training.
---
Q: What fraction of total training steps is a warmup typically set to?
A: Between 0.1% and 20% of the total number of steps.
---
Q: What shape does the learning rate follow under cosine decay after warmup?
A: A half-cosine curve, gliding smoothly from `peak_lr` down to a small `min_lr` by the final step.
---
Q: In the cosine formula η = min + (peak − min)·½(1+cos(πp)), what are the values at p=0 and p=1?
A: At p=0, cos(0)=1 gives η = peak_lr. At p=1, cos(π)=−1 gives η = min_lr.
---
Q: What does gradient clipping cap, and which norm does PyTorch's clip_grad_norm_ use?
A: It caps the magnitude (length) of the gradient vector — its L2 / Euclidean norm — to a chosen max_norm.
---
Q: A gradient matrix has L2 norm 5 and max_norm is 1. By what factor is it scaled, and what is the new norm?
A: It is scaled by max_norm/‖G‖₂ = 1/5, giving a new norm of exactly 1. Direction is unchanged.
---
Q: In what order must clipping, backward, and the optimizer step occur?
A: loss.backward() first (to populate gradients), then clip_grad_norm_ (rescale in place), then optimizer.step().
---
Q: Do any of these three techniques change the model's architecture?
A: No. They only change how the optimizer moves through the loss landscape; the model layers and parameters are untouched.
```

```quiz
1. The main purpose of a learning-rate warmup is to:
   - ( ) increase the final accuracy of the model
   - (x) avoid large, destabilizing updates while the model is freshly initialized
   - ( ) reduce the number of parameters
   - ( ) speed up data loading
   > Small early updates keep the fragile, randomly initialized network from diverging before it has learned anything.

2. Under the cosine decay formula, the learning rate at the very last training step equals:
   - ( ) peak_lr
   - ( ) zero exactly
   - (x) min_lr
   - ( ) initial_lr
   > At progress p=1, cos(π)=−1, so the bracket ½(1+cos π)=0 and η collapses to min_lr.

3. Gradient clipping with max_norm=1.0 modifies a gradient whose L2 norm is 4 by:
   - ( ) setting every entry to 1.0
   - ( ) leaving it unchanged
   - (x) multiplying every entry by 1/4 so the norm becomes 1.0
   - ( ) zeroing it out
   > Clip-by-norm uniformly scales by max_norm/‖g‖₂ = 1/4, preserving direction but capping length at the threshold.

4. Why is gradient clipping applied after loss.backward() but before optimizer.step()?
   - (x) backward must first compute the gradients, and clipping must rescale them before the optimizer applies the update
   - ( ) so the loss is recomputed with clipped values
   - ( ) to avoid running backward twice
   - ( ) because the optimizer deletes gradients after stepping
   > Clipping operates in place on the .grad tensors that backward fills; the step then uses the capped gradients.

5. In the upgraded train_model, where does peak_lr come from?
   - ( ) a hardcoded constant inside the function
   - (x) the optimizer's initial learning rate (param_groups[0]["lr"])
   - ( ) the minimum learning rate times ten
   - ( ) the number of warmup steps
   > The function reads peak_lr = optimizer.param_groups[0]["lr"], so you set the peak by configuring the optimizer's lr.
```

```assignment "Plot and combine all three schedules" level=intermediate
Reproduce the learning-rate schedule from this appendix end to end. Run a training-loop template (no model updates needed) over 15 epochs that (a) linearly warms up from `initial_lr=1e-4` to `peak_lr=1e-2` over the first 20% of steps, then (b) applies cosine decay down to `min_lr = 0.1 * initial_lr`. Collect every per-step learning rate into `track_lrs` and plot it against the step index. Then, separately, backpropagate a single batch through a fresh `GPTModel`, print the highest gradient with `find_highest_gradient`, apply `clip_grad_norm_(..., max_norm=1.0)`, and print it again to confirm it shrank.

Hint: `warmup_steps = int(0.2 * len(train_loader) * n_epochs)`.
Hint: the post-warmup progress is `(global_step - warmup_steps) / (total_training_steps - warmup_steps)`, fed into `min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))`.
Hint: use `matplotlib.pyplot.plot(range(total_training_steps), track_lrs)` for the curve.
```
