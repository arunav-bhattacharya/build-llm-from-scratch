You've coded the hardest piece — [multi-head attention](@/chapters/ch03-attention.html). Now we assemble the **whole GPT**. The good news: despite its size, a GPT is surprisingly simple, because almost everything is the *same transformer block repeated a dozen times*. We'll build each remaining part — layer normalization, a GELU feed-forward network, shortcut connections — snap them into a transformer block, stack twelve of those, and finish with a loop that actually **generates text**.

By the end you'll have a real, 124-million-parameter GPT-2 that runs on your laptop. It won't say anything sensible yet (that's training, in Chapter 5) — but every wire will be in place.

::: objectives "What you'll learn"
- The full **GPT architecture** and the `GPT_CONFIG_124M` settings that define it
- **Layer normalization** — re-centering activations to mean 0 and variance 1 for stable training
- The **GELU** activation and the **feed-forward** network that uses it
- **Shortcut (residual) connections** and how they defeat vanishing gradients
- The **transformer block** that fuses attention, normalization, and feed-forward
- Assembling the complete **`GPTModel`** and counting its parameters
- **Generating text** token by token with a simple autoregressive loop
:::

## Coding an LLM architecture

A GPT (short for **generative pretrained transformer**) is a big deep neural network that produces text one token at a time. "Big" mostly means *repetition*: the same building block stacked over and over. We're scaling up from the tiny embeddings of Chapter 3 to the smallest real GPT-2, with **124 million parameters**. (The original paper said 117M; that was later corrected to 124M.)

A **parameter** is just a trainable weight — one adjustable number inside the model. A single `2048 × 2048` weight matrix already holds over 4 million of them. GPT-2 has 124 million; GPT-3 has 175 *billion* of the same kind, trained on more data.

We capture the model's shape in one configuration dictionary:

```python title="The GPT-2 124M configuration"
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}
```

::: diagram ch04-gpt-config "The seven knobs of GPT-2 small: a 50,257-token vocabulary, a 1,024-token context window, 768-dimensional embeddings, 12 attention heads, 12 stacked layers, 10% dropout, and no bias on the Q/K/V projections."
:::

Each entry has a job: `vocab_size` is the 50,257 tokens the [BPE tokenizer](@/chapters/ch02-text-data.html) knows; `context_length` is the most tokens the model can read at once; `emb_dim` turns every token into a 768-number vector; `n_heads` and `n_layers` set the width and depth; `drop_rate` is the dropout strength; and `qkv_bias` decides whether the attention projections carry a bias (off, following modern practice — we revisit it in Chapter 6 when loading OpenAI's weights).

### A placeholder model

Before coding the real internals, we sketch a **skeleton** — a `DummyGPTModel` — so we can see how the pieces connect and what shapes flow through. It uses empty placeholders where the transformer block and layer norm will go.

```python title="Listing 4.1 — A placeholder GPT model" collapsible
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):   # does nothing — yet
    def __init__(self, cfg):
        super().__init__()
    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):          # does nothing — yet
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    def forward(self, x):
        return x
```

The `forward` method is the whole pipeline in miniature: look up token embeddings, add positional embeddings, apply dropout, run through the transformer blocks, normalize, then project to **logits** with a linear output head. The two `Dummy*` classes just return their input unchanged so the code runs end to end while we fill in the real parts.

Let's feed it a small batch of two sentences (tokenized with the `tiktoken` GPT-2 tokenizer from Chapter 2):

```python title="Tokenize a batch and run the dummy model"
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
# Output shape: torch.Size([2, 4, 50257])
```

The output is shape `[2, 4, 50257]`: **2** sentences, **4** tokens each, and a **50,257-dimensional** vector per token — one score for every word in the vocabulary. Those raw scores are the **logits**; later we'll turn them back into actual words. The skeleton works; now we replace the dummies with the real machinery, starting with layer normalization.

## Normalizing activations with layer normalization

Deep networks are finicky to train. As signals pass through many layers, they can blow up or shrink toward zero — the dreaded **exploding/vanishing gradients** — and training stalls. **Layer normalization** is a simple fix: after a layer computes its outputs (its **activations**), we rescale them so they have **mean 0 and variance 1** before passing them on. This keeps every layer working in a sane numeric range and makes training converge faster and more reliably.

::: callout analogy "Re-centering a recipe's measurements"
Imagine a recipe where one cook measures in grams, another in cups, another in pinches — the numbers are all over the place and impossible to combine. Layer normalization is the chef who **rewrites every measurement onto one common scale** before anything goes in the pot. The *relative* proportions are preserved, but now every value lives in the same comfortable range, so the next step can work with it cleanly.
:::

In GPT-2 and modern transformers, layer norm is applied **before** the attention and feed-forward sub-layers (and once more before the final output). Let's see it in action on a tiny example — a linear layer with a ReLU, applied to two random inputs:

```python title="A small layer to normalize"
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
```

**ReLU** (rectified linear unit) simply clips negatives to zero, so `out` contains only non-negative values. Now check its statistics:

```python title="Mean and variance of the raw activations"
mean = out.mean(dim=-1, keepdim=True)
var  = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)       # e.g. [[0.1324], [0.2170]]
print("Variance:\n", var)    # e.g. [[0.0231], [0.0398]]
```

::: callout note "What dim=-1 and keepdim=True do"
`dim=-1` means "compute along the **last** dimension" — here, across the six features of each row, giving one statistic per row. `keepdim=True` keeps the result as a column shape `[2, 1]` instead of collapsing to `[2]`, so it **broadcasts cleanly** when we subtract it. Using `dim=-1` keeps working unchanged when the data later becomes 3-D `[batch, tokens, embedding]`.
:::

Normalizing is just *subtract the mean, divide by the standard deviation* (the square root of the variance):

```python title="Normalize: mean 0, variance 1"
out_norm = (out - mean) / torch.sqrt(var)
print("Mean:\n", out_norm.mean(dim=-1, keepdim=True))     # ≈ 0
print("Variance:\n", out_norm.var(dim=-1, keepdim=True))  # = 1
```

The normalized rows now have mean ≈ 0 (a value like `-5.96e-08` is just floating-point dust) and variance = 1.

::: diagram ch04-layernorm "Layer normalization re-centers raw activations to mean 0 and unit variance, then applies two learned parameters — a per-feature scale and shift — so the model can re-stretch the distribution if that helps."
:::

### The LayerNorm class

The real module adds two refinements: a tiny **epsilon** in the denominator to avoid dividing by zero, and two **trainable parameters** — `scale` and `shift` — that let the model re-stretch and re-offset the normalized values if doing so improves learning.

```python title="Listing 4.2 — A layer normalization class"
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

The formula, in math:

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu$ and $\sigma^2$ are the per-token mean and variance, and $\gamma$ (`scale`) and $\beta$ (`shift`) are learned. They start at 1 and 0 (a no-op) and drift only if training finds it useful.

::: callout math "Why unbiased=False?"
`var(..., unbiased=False)` divides by $n$ rather than $n-1$ (it skips **Bessel's correction**). For a 768-dimensional embedding, the difference is negligible — and it matches how the original GPT-2 was trained in TensorFlow, so the pretrained weights we load in Chapter 6 stay compatible.
:::

::: callout note "Layer norm vs. batch norm"
You may know **batch normalization**, which normalizes across the *batch* dimension. Layer norm instead normalizes across the *feature* dimension, so it treats each input **independently of batch size**. That's a big win for LLMs, where batch sizes vary with hardware and you may even run one sequence at a time.
:::

Two building blocks down. Next: the activation function and feed-forward network.

## Implementing a feed-forward network with GELU activations

Inside every transformer block sits a small neural network — the **feed-forward** module — and it relies on a smoother activation than ReLU, called **GELU** (Gaussian error linear unit).

ReLU has a hard corner at zero and outputs *exactly* zero for every negative input. **GELU** is its smooth cousin: it curves gently through zero and lets slightly-negative inputs pass a small non-zero signal. The exact definition is $\text{GELU}(x) = x \cdot \Phi(x)$, where $\Phi$ is the standard Gaussian's cumulative distribution function. In practice we use the cheaper curve-fitted approximation (the one the original GPT-2 used):

$$\text{GELU}(x) \approx 0.5\,x\left(1 + \tanh\!\left[\sqrt{\tfrac{2}{\pi}}\left(x + 0.044715\,x^3\right)\right]\right)$$

```python title="Listing 4.3 — The GELU activation function"
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

::: diagram ch04-gelu-vs-relu "GELU is smooth and dips slightly below zero for small negative inputs; ReLU is piecewise-linear with a sharp kink at zero and a flat zero on the left. GELU's gentle slope gives deep networks better gradients."
:::

::: callout analogy "A dimmer switch, not an on/off switch"
ReLU is a light switch: below zero, off; above zero, fully proportional. GELU is a **dimmer** — as inputs drop below zero the output fades gradually instead of snapping to dark. Because even slightly-negative neurons still emit a faint signal (and a non-zero gradient), they keep **contributing to learning** rather than going completely silent. Smooth curves are also easier to optimize than sharp corners.
:::

### The feed-forward module

The feed-forward network is two linear layers with a GELU in between. The trick is that the first layer **expands** the embedding by a factor of 4 (768 → 3072), GELU adds non-linearity in that wider space, and the second layer **contracts** back to 768.

```python title="Listing 4.4 — A feed-forward neural network module"
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)
```

Feed it a batch and the shape comes out unchanged:

```python title="Shape in = shape out"
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)
# torch.Size([2, 3, 768])
```

::: callout analogy "Unfolding a map to find a route, then refolding it"
The expand-then-contract design is like **unfolding a crumpled map** to a much larger surface, drawing the best route across all that room, then folding it back to pocket size. The wider 3,072-dimensional space gives the network room to explore richer combinations of features; the contraction packs the useful result back into 768 dimensions so it slots neatly into the next layer. Keeping input and output dimensions equal is what lets us **stack** these blocks without ever resizing in between.
:::

## Adding shortcut connections

Here's a problem that bites deep networks hard. During training, gradients flow *backward* from the output to update each layer's weights. In a very deep stack, those gradients get multiplied down layer after layer and can **shrink to almost nothing** by the time they reach the early layers — the **vanishing gradient problem**. Early layers barely learn.

**Shortcut connections** (a.k.a. **skip** or **residual** connections) fix this with a beautifully simple move: **add a layer's input to its output**. This creates a parallel "express lane" that the gradient can travel down without being squeezed through every transformation.

::: callout analogy "An express elevator that skips floors"
Picture a 50-story building where a message must be relayed person-to-person down every floor — by the ground floor it's a faint whisper, hopelessly garbled. A shortcut connection is an **express elevator** that carries a copy of the message straight down, bypassing the floors. The signal (and during training, the gradient) arrives strong and intact, so even the lowest floors get clear instructions.
:::

Let's prove it. Here's a 5-layer network that can optionally add shortcuts:

```python title="Listing 4.5 — A network to demonstrate shortcuts" collapsible
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            # add the input back if shapes match
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
```

A helper to print the average gradient magnitude at each layer after one backward pass:

```python title="Inspecting gradients through the network"
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()(output, target)
    loss.backward()                       # PyTorch computes all gradients
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
```

**Without** shortcuts, gradients shrink dramatically toward the early layers:

```python title="Without shortcuts — gradients vanish"
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
print_gradients(model_without_shortcut, sample_input)
# layers.0.0.weight ... 0.00020   <- early layer, tiny
# layers.1.0.weight ... 0.00012
# layers.2.0.weight ... 0.00072
# layers.3.0.weight ... 0.00139
# layers.4.0.weight ... 0.00505   <- last layer
```

**With** shortcuts, the early-layer gradients stay healthy:

```python title="With shortcuts — gradients stay strong"
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)
# layers.0.0.weight ... 0.2217    <- early layer, healthy!
# layers.1.0.weight ... 0.2069
# layers.2.0.weight ... 0.3290
# layers.3.0.weight ... 0.2666
# layers.4.0.weight ... 1.3259
```

The first layer's gradient jumped from `0.0002` to `0.22` — a thousand-fold rescue. Shortcut connections are a core ingredient of every large model, and they'll keep gradients flowing when we train GPT in Chapter 5.

## Connecting attention and linear layers in a transformer block

Now we fuse everything into the **transformer block** — the unit repeated 12 times in GPT-2 small. It combines **masked multi-head attention**, **layer normalization**, **dropout**, the **feed-forward** network, and **two shortcut connections**.

The division of labor is elegant: multi-head attention looks **across** tokens to find relationships ("which earlier words matter for this one?"), while the feed-forward network transforms **each token independently** to enrich its representation. Together they give the model both context and depth.

::: callout analogy "An assembly-line station"
A transformer block is one **station on an assembly line**. A part (the sequence) arrives, gets two treatments — first the attention machine, which lets parts compare notes with their neighbors, then the feed-forward machine, which refines each part on its own — and leaves the station the **exact same size** it entered, ready for the next identical station. Because every station has the same input/output shape, you can bolt as many in a row as you like.
:::

::: diagram ch04-transformer-block "One transformer block: the input is normalized, passed through masked multi-head attention, dropped out, and added back via shortcut 1; then normalized again, passed through the GELU feed-forward network, dropped out, and added back via shortcut 2. The output keeps the input's shape."
:::

```python title="Listing 4.6 — The transformer block" collapsible
from chapter03 import MultiHeadAttention   # from Chapter 3

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x                  # save input
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut              # shortcut 1: add input back

        shortcut = x                  # save again
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut              # shortcut 2: add input back
        return x
```

Notice the order: layer norm comes **before** each sub-layer (attention, feed-forward), and dropout comes **after**. This is **Pre-LayerNorm**, the modern arrangement. The original 2017 transformer put the norm *after* each sub-layer (Post-LayerNorm), which tends to train worse. Each sub-layer is wrapped in its own shortcut so the input is always added back to the output.

Run it and the shape is preserved:

```python title="The block preserves input shape"
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)     # torch.Size([2, 4, 768])
print("Output shape:", output.shape)  # torch.Size([2, 4, 768])
```

Same length, same feature size — but the *content* of each vector is now re-encoded to carry context from the entire sequence. That shape-preservation is exactly what lets us stack these blocks freely.

## Coding the GPT model

We're ready to replace the placeholders in `DummyGPTModel` with the real `TransformerBlock` and `LayerNorm`. The result is the genuine 124M GPT-2.

```python title="Listing 4.7 — The GPT model" collapsible
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

::: diagram ch04-gpt-architecture "The full GPT data flow: tokenized text becomes token + positional embeddings, runs through 12 stacked transformer blocks, a final LayerNorm, and a linear output head that produces logits over all 50,257 vocabulary tokens."
:::

Thanks to `TransformerBlock`, the `GPTModel` is tiny and readable. The constructor builds the token and positional embedding layers, a `Sequential` stack of `n_layers` transformer blocks, the final layer norm, and a bias-free linear head that projects to the vocabulary. The `forward` pass is the same pipeline we sketched with the dummy model — only now every box is real.

Run the real model on our batch:

```python title="Run the real GPT model"
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Output shape:", out.shape)
# Output shape: torch.Size([2, 4, 50257])
```

Same `[2, 4, 50257]` shape as the dummy — two sentences, four tokens, a full vocabulary's worth of logits per token.

### How many parameters?

```python title="Counting parameters"
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
# Total number of parameters: 163,009,536
```

163 million? We promised 124M! The gap is a concept called **weight tying**, used in the original GPT-2: it **reuses the token-embedding matrix as the output-head matrix**. Both are `[50257, 768]` — huge and identical in shape:

```python title="The two big matrices have the same shape"
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)
# both: torch.Size([50257, 768])
```

If we subtract the output head (which GPT-2 ties to the embeddings), we land exactly on 124M:

```python title="Subtracting the tied output head"
total_params_gpt2 = (
    total_params - sum(p.numel() for p in model.out_head.parameters())
)
print(f"Considering weight tying: {total_params_gpt2:,}")
# Considering weight tying: 124,412,160
```

::: callout note "We keep the layers separate"
Weight tying shrinks the memory footprint, but the author finds that **separate** embedding and output layers train better — so our `GPTModel` keeps them separate (which is why it reports 163M). We'll revisit tying in Chapter 6 when loading OpenAI's pretrained weights.
:::

And the storage cost, at 4 bytes per 32-bit float:

```python title="Model size in memory"
total_size_mb = (total_params * 4) / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")
# Total size of the model: 621.83 MB
```

About **622 MB** — a reminder that even a "small" LLM is hefty. To build a larger GPT-2, you change *only the config*: GPT-2 medium uses `emb_dim=1024, n_layers=24, n_heads=16`; XL uses `1600 / 48 / 25`. Same `GPTModel` class, bigger numbers.

## Generating text

The model spits out logits — now we make it **write**. A GPT generates **autoregressively**: predict the next token, append it to the input, and repeat. Each new token becomes part of the context for the next prediction.

For a single step: take the model's logits, focus on the **last token's** row, softmax it into probabilities, pick the index of the highest probability with **argmax** (that index *is* the token ID), and append it.

::: diagram ch04-generation-loop "The autoregressive loop: feed the context to GPT, keep the last token's logits, softmax to probabilities, argmax to pick the most likely token, append its ID to the context, and repeat — building the sentence one token at a time."
:::

```python title="Listing 4.8 — Generating text"
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]        # crop to context window
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                # last token only
        probas = torch.softmax(logits, dim=-1)   # → probabilities
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # pick the top
        idx = torch.cat((idx, idx_next), dim=1)  # append, then loop
    return idx
```

Three details worth noting. `idx[:, -context_size:]` **crops** the running sequence so it never exceeds the model's context window. `logits[:, -1, :]` keeps **only the last position's** prediction — that's the next token. And picking the highest-probability token every time is called **greedy decoding**.

::: callout analogy "Predictive keyboard autocomplete"
This is exactly your phone's **predictive text**, taken to the extreme. You type "Hello, I am" and it suggests the single most likely next word; tap it and that word joins the sentence, shifting the next suggestion. Run it on autopilot, always tapping the top suggestion, and the phone writes a whole sentence by itself — one greedy word at a time.
:::

::: callout note "The softmax here is technically optional"
Softmax is **monotonic** — it preserves order — so the argmax of the probabilities equals the argmax of the raw logits. We could skip softmax and call argmax on the logits directly for identical results. It's kept here to illustrate the full logits → probabilities → token pipeline. (In Chapter 5 we'll add temperature and top-k sampling, which *do* need the probabilities, to make generation more creative.)
:::

Let's generate. Encode a prompt, switch the model to `eval()` mode (which **turns off dropout**), and run six steps:

```python title="Generate from 'Hello, I am'"
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # add batch dim

model.eval()                                          # disable dropout
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
# Hello, I am Featureiman Byeswickattribute argue
```

Gibberish! And that's exactly right. We built and **wired** the architecture, but initialized it with **random weights** — it hasn't learned anything yet. Teaching it to produce coherent text is the whole job of **training**, which is Chapter 5. The engine is fully assembled; next we fuel it.

## Key takeaways

::: takeaways
- A **GPT** is mostly the *same transformer block repeated* — `n_layers` times (12 for GPT-2 small). The whole model is defined by a small config dict.
- **Layer normalization** rescales each token's activations to mean 0 and variance 1, plus learned `scale` and `shift`, stabilizing and speeding up training. It normalizes across **features**, independent of batch size.
- **GELU** is a smooth activation that, unlike ReLU, passes a small non-zero signal (and gradient) for negative inputs — better for deep optimization.
- The **feed-forward** network expands the embedding 4× (768 → 3072), applies GELU, and contracts back, exploring a richer representation space while preserving shape.
- **Shortcut (residual) connections** add a layer's input to its output, creating an express path that keeps gradients from vanishing in deep stacks.
- A **transformer block** = LayerNorm → masked multi-head attention → dropout → +shortcut → LayerNorm → feed-forward → dropout → +shortcut, with **Pre-LayerNorm** ordering. It preserves input shape so blocks stack freely.
- The full **`GPTModel`** assembles embeddings + stacked blocks + final LayerNorm + linear head. GPT-2 small is ~124M parameters (163M without weight tying), about 622 MB.
- **Text generation** is autoregressive: logits → (softmax) → argmax → append → repeat. With random weights it produces gibberish — training comes next.
:::

## Additional references

::: refs
- [Chapter 4 code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04) — GitHub · the complete GPT implementation from this chapter.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Blog · Jay Alammar's visual tour of the GPT-2 architecture and generation.
- [Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Video · Andrej Karpathy assembles a GPT, including blocks and residuals, live.
- [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) — Paper · the original GELU activation that GPT uses.
- [Layer Normalization](https://arxiv.org/abs/1607.06450) — Paper · Ba, Kiros & Hinton introduce layer normalization.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — Paper · the ResNet paper that introduced shortcut connections.
- [Building a GPT-Style LLM Classifier From Scratch](https://sebastianraschka.com/blog/2024/building-a-gpt-style-llm-classifier.html) — Blog · the author's deep dive reusing this exact architecture.
:::

## Test your knowledge

Lock in the architecture with flashcards, a quiz, and a hands-on build.

```flashcards
Q: What does layer normalization do to a layer's activations?
A: Re-centers them to **mean 0 and variance 1** (across the feature dimension), then applies a learned **scale** and **shift**, stabilizing training.
---
Q: Why does GPT use GELU instead of ReLU?
A: GELU is smooth and gives a small non-zero output/gradient for negative inputs (no hard corner), which improves optimization in deep networks.
---
Q: What is the shape transformation inside the FeedForward module for emb_dim=768?
A: 768 → 3072 (expand 4×) → GELU → 3072 → 768 (contract). Input and output dimensions match.
---
Q: What problem do shortcut (residual) connections solve, and how?
A: The **vanishing gradient** problem. By adding a layer's input to its output, they create an express path so gradients stay strong in early layers.
---
Q: What are the components of a transformer block, in order?
A: LayerNorm → masked multi-head attention → dropout → +shortcut → LayerNorm → FeedForward(GELU) → dropout → +shortcut. (Pre-LayerNorm.)
---
Q: Why does GPTModel report 163M parameters when GPT-2 small is "124M"?
A: GPT-2 uses **weight tying** (shared token-embedding and output-head matrix). Counting that shared matrix once gives 124M; our model keeps them separate.
---
Q: In generate_text_simple, why slice logits with [:, -1, :]?
A: Only the **last token's** logits predict the next token; earlier positions are ignored during generation.
---
Q: Why does the untrained GPT produce gibberish?
A: It has the correct architecture but **random, untrained weights** — it hasn't learned language yet. Training (Chapter 5) fixes this.
```

```quiz
1. Layer normalization normalizes across which dimension?
   - ( ) the batch dimension
   - (x) the feature (embedding) dimension
   - ( ) the sequence-length dimension
   - ( ) all dimensions at once
   > Layer norm normalizes each token's features independently of batch size — unlike batch norm, which normalizes across the batch.

2. What is the main benefit of a shortcut (residual) connection?
   - ( ) it reduces the number of parameters
   - ( ) it speeds up the forward pass
   - (x) it preserves gradient flow, mitigating vanishing gradients
   - ( ) it removes the need for layer normalization
   > Adding the input back to the output gives gradients an unobstructed path, keeping early-layer gradients from shrinking to zero.

3. The FeedForward network's first linear layer changes the dimension how?
   - ( ) keeps it at 768
   - (x) expands it by 4× (768 → 3072)
   - ( ) shrinks it to 192
   - ( ) expands it to the vocabulary size
   > The first layer expands 4×, GELU adds non-linearity, and the second layer contracts back to 768.

4. In a Pre-LayerNorm transformer block, layer normalization is applied…
   - (x) before the attention and feed-forward sub-layers
   - ( ) after the attention and feed-forward sub-layers
   - ( ) only once, at the very end
   - ( ) never — only dropout is used
   > GPT-2 and modern transformers normalize *before* each sub-layer (Pre-LayerNorm); the original transformer used Post-LayerNorm, which trains worse.

5. What does generate_text_simple do at each step after computing probabilities?
   - ( ) samples randomly from the full distribution
   - ( ) averages the top 5 tokens
   - (x) takes the argmax (most likely token) and appends it
   - ( ) stops if the probability is below 0.5
   > It uses greedy decoding: argmax picks the single most likely token, which is appended to the running context for the next iteration.
```

```assignment "Build and run the full GPT-2 small" level=intermediate
Assemble the complete `GPTModel` using `GPT_CONFIG_124M` and confirm it runs end to end. Then verify two facts numerically: (1) the total parameter count is **163,009,536**, and (2) after subtracting the output head's parameters (weight tying), the count drops to **124,412,160**. Finally, call `generate_text_simple` on the prompt `"Hello, I am"` for `max_new_tokens=6` and decode the result — confirm it's gibberish because the model is untrained.

Hint: count parameters with `sum(p.numel() for p in model.parameters())`.
Hint: remember `model.eval()` before generating to disable dropout, and add a batch dimension to your encoded prompt with `.unsqueeze(0)`.
```
