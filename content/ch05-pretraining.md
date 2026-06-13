We have an LLM architecture, but its weights are random — feed it *"Every effort moves you"* and it babbles `rentingetic wasn? refres RexMe`. This is the chapter where we **train** it. We'll build a way to *measure* how wrong the model is (cross-entropy loss), wrap a standard PyTorch training loop around it, watch the loss fall as the model learns to write English, then add **decoding strategies** (temperature, top-k) that make its output more creative. Finally, we'll save our weights — and load OpenAI's far more capable GPT-2 weights into the very same class we built.

Pretraining a serious LLM costs hundreds of thousands of dollars in GPU time. We'll train on a single short story so it runs in minutes on a laptop, learning the exact same mechanics the big labs use.

::: objectives "What you'll learn"
- How to turn text into token IDs and back, to drive generation and evaluation
- **Cross-entropy loss** and **perplexity** — numerical measures of text quality
- Splitting data into **training** and **validation** sets and computing loss over both
- A complete PyTorch **training loop** with the **AdamW** optimizer
- **Decoding strategies**: temperature scaling and top-k sampling to control randomness
- Saving and loading model + optimizer **weights** with `torch.save` / `torch.load`
- Loading **OpenAI's pretrained GPT-2 weights** into our own `GPTModel`
:::

## Evaluating generative text models

Before we can train, we need a way to *score* the model's output. This first section recaps generation from chapter 4, then builds up a loss metric step by step, and finally computes that loss across a real dataset split into training and validation portions.

### Using GPT to generate text

We start by re-instantiating the GPT model from chapter 4. The only change: we shrink the context length from 1,024 to **256 tokens** to make laptop training feasible.

```python title="The model config (context length reduced to 256)"
import torch
from chapter04 import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,   # shortened from 1,024 to 256 tokens
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,        # dropout; it's common to set this to 0
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()
```

Generating text is a three-step cycle: the **tokenizer** encodes text into token IDs, the **model** turns those IDs into logits (one probability-distribution vector per position over the 50,257-token vocabulary), and the tokenizer decodes the chosen IDs back into text. To make this conversion painless throughout the chapter, we add two helper functions:

```python title="Listing 5.1 — Utility functions for text ↔ token ID conversion"
import tiktoken
from chapter04 import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # add batch dim
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)               # remove batch dim
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

The untrained model produces gibberish — `Every effort moves you rentingetic wasn? refres RexMeCHicular stren` — because it hasn't learned anything yet. To improve it, we first need a number that says *how bad* this output is.

### Calculating the text generation loss

Let's compute that number with a tiny worked example. We use two input sequences already mapped to token IDs, and **targets** that are the inputs shifted one position to the right (the next-token prediction setup from chapter 2):

```python title="Inputs and their shifted targets"
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])    #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107, 588, 11311]])   #  " really like chocolate"]
```

Feed the inputs through the model and softmax the logits into probabilities:

```python title="Logits → probabilities"
with torch.no_grad():        # not training yet, so skip gradient tracking
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
print(probas.shape)
# torch.Size([2, 3, 50257])  -> (batch, tokens, vocab_size)
```

The shape `[2, 3, 50257]` means: 2 examples, 3 tokens each, and a 50,257-long probability vector per token. Greedy generation would `argmax` each vector to pick the most likely next token — but for the *loss*, we care about something else: **how much probability the model assigned to the *correct* (target) token**.

::: diagram ch05-cross-entropy "Cross-entropy loss: softmax the logits, pluck out the probabilities at the target-token positions, take the negative log of each, and average. A confident-and-correct model gets a low loss; a surprised one gets a high loss."
:::

We extract the model's probability at each target position. For an untrained model these are minuscule (random guessing over 50k tokens hovers around 1/50,257 ≈ 0.00002):

```python title="Pick the probabilities at the target token positions"
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)
# tensor([7.4541e-05, 3.1061e-05, 1.1563e-05])

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)
# tensor([1.0337e-05, 5.6776e-05, 4.7559e-06])
```

Training's whole job is to push these target probabilities **up**. To turn them into a single training signal, we take the **log** of each, **average**, and **negate**:

```python title="Log → average → negate = cross-entropy loss"
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
# tensor([ -9.5042, -10.3796, -11.3677, -11.4798,  -9.7764, -12.2561])

avg_log_probas = torch.mean(log_probas)
# tensor(-10.7940)

neg_avg_log_probas = avg_log_probas * -1
# tensor(10.7940)  <- this is the cross-entropy loss
```

::: callout analogy "Loss = how shocked the model is"
Think of cross-entropy as a **surprise meter**. Each time the model sees the correct next word, we ask: *how shocked were you?* If it had assigned the right word a high probability, the surprise (−log) is small. If it gave the right word almost no probability, the surprise is huge. The loss is the **average surprise** across all tokens — and training works to keep the model from being surprised by the truth.
:::

::: callout math "Why logarithms and a minus sign?"
A probability lives in $(0, 1]$, so its log is negative — and **closer to 0 for higher probability** ($\log 1 = 0$). Logs also turn the *product* of many per-token probabilities into a *sum*, which is far more numerically stable to optimize. We negate so that a *better* model gives a *smaller, positive* number we can minimize. Formally, for target tokens $t_i$ the loss is
$$L = -\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(t_i)$$
We aim to drive $L$ toward 0 by updating the weights $\theta$.
:::

PyTorch bundles all of this — softmax, gather target probabilities, log, average, negate — into one function, `cross_entropy`. It expects the logits flattened over the batch dimension:

```python title="The same loss via torch.nn.functional.cross_entropy"
print("Logits shape:", logits.shape)    # torch.Size([2, 3, 50257])
print("Targets shape:", targets.shape)  # torch.Size([2, 3])

logits_flat = logits.flatten(0, 1)      # torch.Size([6, 50257])
targets_flat = targets.flatten()        # torch.Size([6])

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)
# tensor(10.7940)  -> identical to the manual computation
```

::: callout key "Cross-entropy = negative average log probability"
The terms are used interchangeably in practice. Cross-entropy measures the gap between two distributions — the *true* next-token distribution (a one-hot at the correct token) and the model's *predicted* distribution. Minimizing it is the same as maximizing the probability the model assigns to the real next word.
:::

A companion metric, **perplexity**, makes the loss more interpretable:

```python title="Perplexity = exp(loss)"
perplexity = torch.exp(loss)
print(perplexity)
# tensor(48725.8203)
```

::: diagram ch05-perplexity "Perplexity is just the exponential of the loss. A loss of 10.79 becomes a perplexity of ~48,725 — meaning the model is effectively choosing at random among ~48,725 words at each step."
:::

::: callout analogy "Perplexity = how many words it's guessing between"
Perplexity answers: *"At each step, how many words is the model effectively torn between?"* A perplexity of 48,725 means the model is as uncertain as if it were picking uniformly at random from ~48,725 tokens — basically lost. A well-trained model might reach a perplexity in the tens: confidently narrowing each choice to a handful of plausible words. Lower is better, and it tracks the loss exactly since it's just $e^{\text{loss}}$.
:::

### Calculating the training and validation set losses

Now we apply the loss to a real dataset: Edith Wharton's public-domain short story *"The Verdict"* (the same text from chapter 2). It's tiny — about 5,145 tokens — which is the point: training runs in minutes.

```python title="Load the dataset and count tokens"
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)   # 20479
print("Tokens:", total_tokens)            # 5145
```

We hold back 10% of the text as a **validation set** — data the model trains on *but never sees during weight updates* — so we can check whether it's genuinely learning the language or just memorizing the training text.

```python title="Split into 90% train / 10% validation"
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
```

::: callout analogy "Studying vs. a pop quiz"
The **training loss** is like grading a student on the exact homework they studied — they can ace it by memorizing. The **validation loss** is a *pop quiz on unseen questions*: it reveals whether they actually learned the material or just memorized answers. When training loss keeps dropping but validation loss stalls or rises, the model is **overfitting** — memorizing rather than generalizing.
:::

We wrap each split in the `create_dataloader_v1` from chapter 2, using the full 256-token context as both the chunk length and the stride (so chunks don't overlap):

```python title="Build the training and validation data loaders" collapsible
from chapter02 import create_dataloader_v1

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
```

This yields nine training batches and one validation batch, each of shape `[2, 256]`. Next we need a function that computes the loss for a single batch — note it moves the data to the right `device` (CPU or GPU):

```python title="Loss for one batch"
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss
```

And a function that averages that loss across all (or a capped number of) batches from a loader:

```python title="Listing 5.2 — Loss over a whole data loader"
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)        # iterate over all batches
    else:
        num_batches = min(num_batches, len(data_loader))  # cap to available
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()         # sum each batch's loss
        else:
            break
    return total_loss / num_batches           # average over batches
```

Applying it to both loaders on the untrained model:

```python title="Initial train/val losses (untrained)"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():     # no gradients needed for evaluation
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)    # 10.98758347829183
print("Validation loss:", val_loss)    # 10.98110580444336
```

Both losses sit near **11** — the model is effectively guessing randomly. Our entire goal now is to drive these down.

## Training an LLM

Time to train. We'll use a textbook PyTorch training loop. Figure 5.11 in the book lays it out as eight steps; the core five repeat for every batch, and the last two are optional progress checks.

::: diagram ch05-training-loop "The training loop. For each epoch, iterate over every batch: zero the gradients, forward-pass to get the loss, backpropagate, and let the optimizer update the weights. Every few steps, evaluate on train+val and print a text sample to watch progress."
:::

Here is the loop in code. The key five steps are: `zero_grad()` (clear stale gradients), `calc_loss_batch` (forward pass), `loss.backward()` (compute gradients), and `optimizer.step()` (update weights).

```python title="Listing 5.3 — The main pretraining function" collapsible
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):          # main training loop
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()            # reset gradients from last batch
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()                  # compute loss gradients
            optimizer.step()                 # update weights with gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:           # optional evaluation
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        generate_and_print_sample(           # print a sample after each epoch
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen
```

Two helpers complete the picture. `evaluate_model` computes train/val loss with the model in **eval mode** (dropout off) and gradients disabled — giving a clean, stable readout:

```python title="The evaluation helper"
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():            # no gradient tracking during eval
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss
```

And `generate_and_print_sample` prints a short generated snippet so we can *read* the model improving, not just watch a number:

```python title="The sample-generation helper"
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))   # compact print
    model.train()
```

::: callout note "Why toggle model.train() and model.eval()?"
Dropout (and other stochastic layers) must be **active during training** so the model can't over-rely on any one connection — but **disabled during evaluation and generation** so results are stable and reproducible. `model.train()` and `model.eval()` flip this switch. Forgetting `model.eval()` at inference time is a classic bug that makes outputs jitter run to run.
:::

Now we train for **10 epochs** with the **AdamW** optimizer:

```python title="Run training with AdamW"
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),        # all trainable weights
    lr=0.0004, weight_decay=0.1
)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
```

::: callout tip "What is AdamW?"
**Adam** is the go-to optimizer for deep nets: it adapts the learning rate per parameter using running estimates of the gradient's mean and variance. **AdamW** is a variant that handles **weight decay** correctly — gently shrinking large weights to curb overfitting (regularization), applied *separately* from the gradient step. This cleaner decoupling gives better generalization, which is why AdamW dominates LLM training.
:::

Training takes about 5 minutes on a MacBook Air. Watch the model go from punctuation soup to grammatical prose:

```text title="Training output (abridged)"
Ep 1 (Step 000000): Train loss 9.781, Val loss 9.933
Ep 1 (Step 000005): Train loss 8.111, Val loss 8.339
Every effort moves you,,,,,,,,,,,,.
Ep 2 (Step 000015): Train loss 5.961, Val loss 6.616
Every effort moves you, and, and, and, and, and, and, and, and, and, and, ...
[...]
Ep 9 (Step 000080): Train loss 0.541, Val loss 6.393
Every effort moves you?"  "Yes--quite insensible to the irony. She wanted
him vindicated--and by me!"  He laughed again ...
Ep 10 (Step 000085): Train loss 0.391, Val loss 6.452
Every effort moves you know," was one of the axioms he laid down across the
Sevres and silver of an exquisitely appointed luncheon-table ...
```

The training loss plummets from **9.781 to 0.391** — the model has learned to write. But notice the **validation loss** starts at 9.933 and only reaches **6.452**, and after epoch 2 the two curves *diverge*. That gap is the signature of **overfitting**: with only ~5,000 tokens trained over 10 epochs, the model is memorizing *"The Verdict"* verbatim rather than learning general English. (You can confirm it by searching the original text for phrases like *"quite insensible to the irony"*.)

::: callout warning "Overfitting is expected here — and normal to avoid in practice"
This memorization happens *because* the dataset is tiny and we run many epochs. Real LLM pretraining uses enormous corpora (think 60,000+ books, or trillions of tokens) for typically a **single pass**, where this kind of verbatim overfitting doesn't occur. We accept it here purely so the code runs on a laptop in minutes.
:::

The book plots the two loss curves with `matplotlib` (a `plot_losses` helper with a twin x-axis for "tokens seen"); the takeaway is the diverging-curves picture described above.

## Decoding strategies to control randomness

Our trained model uses **greedy decoding** — it always `argmax`es the single most likely next token. That makes it deterministic *and* prone to regurgitating memorized passages. Two knobs, **temperature** and **top-k**, inject controlled randomness for more original output. First, move the model back to CPU (inference on a small model doesn't need a GPU) and into eval mode:

```python title="Back to CPU, eval mode"
model.to("cpu")
model.eval()
```

To illustrate, we use a toy 9-word vocabulary and a hypothetical set of next-token logits:

```python title="A toy vocabulary and next-token logits"
vocab = {
    "closer": 0, "every": 1, "effort": 2, "forward": 3, "inches": 4,
    "moves": 5, "pizza": 6, "toward": 7, "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}

next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)
```

Greedy decoding always picks `"forward"` (highest logit). To add variety, we replace `argmax` with `torch.multinomial`, which **samples** a token *in proportion to* its probability:

```python title="Probabilistic sampling instead of argmax"
probas = torch.softmax(next_token_logits, dim=0)

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])   # "forward" (usually — but not always)
```

Sampling 1,000 times shows `"forward"` wins most of the time (≈582/1000), but `"toward"`, `"closer"`, and `"inches"` also get picked — exactly the diversity we want.

### Temperature scaling

**Temperature** is a creativity dial. We divide the logits by a temperature value before softmax:

```python title="Softmax with temperature"
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)
```

::: callout math "What the temperature does to the distribution"
For logits $z_i$ and temperature $T$, the probability is
$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
- $T = 1$: no change — the original softmax.
- $T < 1$ (e.g. 0.1): logits are *amplified*, so the gap between top tokens widens — the distribution gets **peaky**, approaching greedy `argmax`.
- $T > 1$ (e.g. 5): logits are *flattened*, so probabilities spread out — the distribution gets **uniform**, and unlikely tokens get a real shot.
:::

::: diagram ch05-temperature "Temperature reshapes the next-token distribution. At T=0.1 almost all probability piles onto the single best token (here 'forward'); at T=1 it's the model's natural distribution; at T=5 the bars flatten and even unlikely tokens like 'pizza' become possible."
:::

::: callout analogy "Temperature = a creativity dial"
Low temperature is a **cautious writer** who always reaches for the safest, most predictable word — coherent but repetitive. High temperature is a **caffeinated improviser** who'll throw in surprising words — more original, but liable to produce nonsense (at $T=5$, *"every effort moves you pizza"* shows up about 4% of the time). Most real systems sit between, around 0.7–1.0.
:::

### Top-k sampling

High temperature alone risks genuinely silly tokens. **Top-k sampling** fences that off: keep only the *k* highest-logit tokens as candidates, and zero out the rest *before* sampling.

::: diagram ch05-topk "Top-k sampling with k=3. Keep the three highest-logit tokens, set every other logit to −∞, then softmax so those three form a fresh distribution that sums to 1, and sample from them. Tokens outside the shortlist get exactly zero probability."
:::

```python title="Selecting the top-k tokens"
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits:", top_logits)     # tensor([6.7500, 6.2800, 4.5100])
print("Top positions:", top_pos)     # tensor([3, 7, 0])
```

We then use `torch.where` to set every logit *below* the smallest top-k logit to **−∞** — the same masking trick from causal attention in chapter 3:

```python title="Mask non-top-k logits with −inf, then softmax"
new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],   # below the top-3 cutoff
    input=torch.tensor(float('-inf')),              # → -inf
    other=next_token_logits                         # keep the rest
)
print(new_logits)
# tensor([4.5100, -inf, -inf, 6.7500, -inf, -inf, -inf, 6.2800, -inf])

topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)
# tensor([0.0615, 0., 0., 0.5775, 0., 0., 0., 0.3610, 0.])
```

Since $e^{-\infty} = 0$, the masked tokens get exactly zero probability and the surviving three renormalize to sum to 1 — guaranteeing we only ever sample a sensible word.

::: callout analogy "Top-k = a shortlist"
Top-k is like a hiring manager who first screens the applicant pool down to a **shortlist of k finalists**, then makes the final pick from that list. No matter how the dice fall, an unqualified candidate (a low-probability token like *"pizza"*) never even reaches the interview. Temperature still controls how the choice is made *among* the finalists.
:::

### Modifying the text generation function

We fold both techniques into a single `generate` function that supersedes `generate_text_simple`. It applies top-k filtering, then temperature sampling — and falls back to greedy `argmax` when `temperature` is 0:

```python title="Listing 5.4 — A text generation function with temperature + top-k" collapsible
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):

    for _ in range(max_new_tokens):           # same loop as before
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]             # only the last time step

        if top_k is not None:                 # 1) top-k filtering
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:                 # 2) temperature + sampling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:                                 # greedy decoding (as before)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:                # stop early on end-of-sequence
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

```python title="Generating with variety"
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# Every effort moves you stand to work on surprise, a one of us had gone with random-
```

The output is now novel rather than a memorized passage. Note: to recover deterministic, greedy behavior, set `temperature=0.0` and leave `top_k=None`.

## Loading and saving model weights in PyTorch

Pretraining is expensive, so we want to **persist** our trained weights. The recommended approach saves the model's `state_dict` — a dictionary mapping each layer to its parameter tensors:

```python title="Save the model weights"
torch.save(model.state_dict(), "model.pth")
```

To reload, instantiate a fresh model and load the saved `state_dict`. Always call `model.eval()` afterward to disable dropout for inference:

```python title="Load the model weights"
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
```

If you plan to **resume training** later, you must also save the **optimizer state**. Adaptive optimizers like AdamW keep per-parameter running statistics (the moment estimates); without them, the optimizer restarts cold and the model may converge poorly:

```python title="Save model + optimizer state together"
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
)
```

```python title="Restore both to continue training"
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();
```

::: callout tip "Save the optimizer, not just the model"
A common mistake is saving only `model.state_dict()` and then wondering why resumed training stutters. AdamW's momentum and variance estimates are part of the training *trajectory* — losing them is like a runner stopping mid-race and forgetting their pace. For inference-only use you don't need the optimizer; for resuming training you do.
:::

## Loading pretrained weights from OpenAI

Our laptop-trained model is a proof of concept. The exciting payoff: OpenAI **open-sourced GPT-2's weights**, so we can pour a genuinely capable model into the *exact same `GPTModel` class* we built from scratch — no $100k training run required.

The book downloads a small helper module, `gpt_download.py`, from the chapter's repository (OpenAI saved the weights with TensorFlow, so `tensorflow` and `tqdm` are prerequisites):

```python title="Download and load the GPT-2 weights"
# pip install tensorflow>=2.15.0 tqdm>=4.66
from gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)
print("Settings:", settings)
# {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
print("Parameter dictionary keys:", params.keys())
# dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])
```

`settings` mirrors our config dict; `params` holds the actual weight tensors. GPT-2 ships in four sizes — all the same architecture, differing only in embedding size and how many times the blocks/heads repeat:

```python title="The four GPT-2 sizes"
model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
```

We pick the small model and update our config. Two crucial tweaks: restore the **1,024-token context** OpenAI trained with, and enable **`qkv_bias`** (OpenAI used bias vectors in the attention projections, so we must match for the weights to fit):

```python title="Match our config to the pretrained model"
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})   # OpenAI used 1,024 tokens
NEW_CONFIG.update({"qkv_bias": True})          # OpenAI used QKV bias vectors

gpt = GPTModel(NEW_CONFIG)
gpt.eval()
```

::: diagram ch05-load-weights "Loading the weights is a careful layer-by-layer match: OpenAI's parameter dictionary (with its TensorFlow naming) is copied into our GPTModel's embeddings, attention, feed-forward, and layer-norm tensors via an assign() helper that shape-checks every copy."
:::

The `assign` helper copies a weight while verifying the shapes match — a built-in safety net against mis-wiring:

```python title="The shape-checking assign helper"
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                          "Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
```

Finally, `load_weights_into_gpt` walks every layer and assigns OpenAI's tensors into ours. OpenAI stored the combined Q/K/V projection in one tensor, so we `np.split` it into three; their naming differs from ours, so each line is a deliberate mapping:

```python title="Listing 5.5 — Loading OpenAI weights into our GPTModel" collapsible
import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):     # one transformer block at a time
        # split the fused QKV weight into three parts
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])  # weight tying
```

::: callout note "Weight tying"
The very last line reuses the **token embedding weights** (`wte`) as the output layer's weights. OpenAI's GPT-2 ties these two matrices together — the same matrix that maps tokens *into* embeddings also maps the final hidden states *back to* vocabulary logits. This **weight tying** shaves off millions of parameters and often improves quality.
:::

Now load the weights and generate. With real pretrained weights, the same `generate` function produces genuinely coherent text:

```python title="Load the weights and generate coherent text"
load_weights_into_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# Every effort moves you toward finding an ideal new way to practice something!
# What makes us want to be on top of that?
```

The fact that it writes fluently is itself the verification: a single mis-wired tensor would have caused the `assign` shape check to fail, or produced garbage output. We now have a capable GPT-2 living inside the architecture we built by hand — and in the next chapters we'll **fine-tune** it to classify text and follow instructions.

## Key takeaways

::: takeaways
- LLMs generate one token at a time; by default the next token is the highest-probability one (**greedy decoding**).
- **Cross-entropy loss** = the negative average log-probability the model assigns to the correct next tokens. Lower means less "surprised." `torch.nn.functional.cross_entropy` computes it in one call.
- **Perplexity** = $e^{\text{loss}}$ — an interpretable measure of how many tokens the model is effectively guessing between.
- Splitting data into **training** and **validation** sets lets you detect **overfitting**: when training loss keeps dropping but validation loss stalls, the model is memorizing.
- The **training loop** is standard PyTorch: for each batch, `zero_grad → forward (loss) → backward → optimizer.step`. We use the **AdamW** optimizer for its decoupled weight-decay regularization.
- **Temperature** scaling sharpens (T<1) or flattens (T>1) the next-token distribution; **top-k** sampling restricts choices to the k most likely tokens (masking the rest with −∞). Together they control output diversity.
- Save/restore weights with `torch.save`/`torch.load` on the `state_dict`; save the **optimizer state** too if you'll resume training.
- OpenAI's **pretrained GPT-2 weights** load into our own `GPTModel` via a layer-by-layer `assign`, giving a capable model without costly pretraining (note the **weight tying** of input embeddings and output head).
:::

## Additional references

::: refs
- [Chapter 5 code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05) — GitHub · the full pretraining, decoding, and weight-loading code from this chapter.
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Video · Andrej Karpathy trains a GPT live, covering the loss and training loop end to end.
- [How to generate text with Transformers](https://huggingface.co/blog/how-to-generate) — Blog · Hugging Face's hands-on guide to greedy, temperature, top-k, and nucleus decoding.
- [A Gentle Introduction to Cross-Entropy for Machine Learning](https://machinelearningmastery.com/cross-entropy-for-machine-learning/) — Blog · cross-entropy and its link to negative log-likelihood, with worked examples.
- [Two minutes NLP — Perplexity explained](https://medium.com/nlplanet/two-minutes-nlp-perplexity-explained-with-simple-probabilities-6cdc46884584) — Blog · perplexity as the effective branching factor of a language model.
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — Paper · Loshchilov & Hutter's paper introducing the AdamW optimizer used here.
- [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) — Docs · the official PyTorch guide to `state_dict`, `torch.save`/`load`, and checkpoints.
:::

## Test your knowledge

Quiz yourself on loss, evaluation, decoding, and weight loading before moving on.

```flashcards
Q: What does cross-entropy loss measure for a language model?
A: The negative average log-probability the model assigns to the correct (target) tokens — intuitively, the model's average "surprise" at the true next words. Lower is better.
---
Q: How is perplexity computed from the loss, and what does it mean?
A: perplexity = exp(loss). It's the effective number of tokens the model is choosing between at each step — a perplexity of 50 means roughly "torn between 50 words."
---
Q: Why split data into training and validation sets?
A: The validation set is unseen during weight updates, so its loss reveals genuine learning vs. memorization. A growing gap between low train loss and high val loss signals overfitting.
---
Q: What are the five core steps inside the training loop for each batch?
A: optimizer.zero_grad() → compute loss (forward) → loss.backward() (gradients) → optimizer.step() (update weights). (Plus periodic evaluation/sampling.)
---
Q: What does the temperature parameter do in sampling?
A: It divides the logits before softmax. T<1 sharpens the distribution (more confident/greedy); T>1 flattens it (more diverse/random); T=1 is the unscaled distribution.
---
Q: How does top-k sampling work?
A: Keep only the k highest-logit tokens, set all other logits to −∞ so softmax gives them zero probability, then sample from the renormalized top-k distribution.
---
Q: When saving a model to resume training, what must you save besides the model's state_dict?
A: The optimizer's state_dict. AdamW keeps per-parameter moment estimates; losing them makes resumed training converge poorly.
---
Q: What is "weight tying" in GPT-2?
A: Reusing the token embedding matrix as the output projection layer's weights, so one matrix both embeds tokens and maps final hidden states back to vocabulary logits — saving parameters.
```

```quiz
1. Cross-entropy loss for an LLM is equivalent to:
   - ( ) the average probability of the correct tokens
   - (x) the negative average log-probability of the correct tokens
   - ( ) the sum of all token probabilities
   - ( ) the variance of the logits
   > Cross-entropy = negative average log-probability of the target tokens; PyTorch's cross_entropy computes exactly this.

2. The training loss falls steadily but the validation loss flattens and the two diverge. This indicates:
   - ( ) the learning rate is too low
   - ( ) the model has converged perfectly
   - (x) the model is overfitting (memorizing the training data)
   - ( ) the optimizer is broken
   > A widening train/val gap is the classic signature of overfitting — common here because the dataset is tiny and trained for many epochs.

3. Setting the temperature to a value greater than 1 will:
   - ( ) make the distribution peakier and output more repetitive
   - (x) flatten the distribution, making unlikely tokens more probable and output more diverse
   - ( ) have no effect on sampling
   - ( ) disable sampling entirely
   > Dividing logits by T>1 shrinks their differences, flattening the softmax toward uniform — more variety, more risk of nonsense.

4. In top-k sampling, the non-top-k logits are set to −∞ so that:
   - (x) after softmax their probability is exactly 0 and the top-k probabilities renormalize to sum to 1
   - ( ) they become the most likely tokens
   - ( ) the loss is minimized
   - ( ) the model trains faster
   > e^(−∞)=0, so masked tokens get zero probability and only the top-k survive, summing to 1 — the same trick as causal masking.

5. Why does the book enable qkv_bias=True before loading OpenAI's GPT-2 weights?
   - ( ) bias vectors speed up inference
   - ( ) it reduces the parameter count
   - (x) OpenAI's GPT-2 used bias vectors in the attention projections, so our architecture must match for the weights to fit
   - ( ) it is required for the AdamW optimizer
   > Loading pretrained weights requires the architectures to match exactly; OpenAI used QKV biases, so we must enable them (the assign shape-check would otherwise fail).
```

```assignment "Exercise 5.3 — Force deterministic generation" level=intermediate
Find the combination(s) of settings for the `generate` function that make it deterministic — always producing the same output for a given start context, just like the original `generate_text_simple`. Then verify by running it twice on `"Every effort moves you"` and confirming the outputs are identical.

Hint: Look at the two branches inside `generate` — the `if temperature > 0.0` branch (which samples) versus the `else` branch (which uses `argmax`). Which branch is deterministic?
Hint: Setting `temperature=0.0` triggers greedy `argmax` decoding. With `top_k` left as `None` (or any value, since argmax is unaffected by it), the result is fully deterministic.
Hint: Call `generate(model, idx=..., max_new_tokens=25, context_size=..., temperature=0.0)` twice and compare the decoded strings.
```
