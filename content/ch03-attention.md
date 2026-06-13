This is the chapter where LLMs stop feeling like magic. **Self-attention** is the engine at the heart of every transformer, and we're going to build it from absolutely nothing — starting with a version that has *no* trainable weights, then adding the real machinery used in GPT: query/key/value projections, scaled dot-product attention, a causal mask, dropout, and finally **multi-head attention**.

It's the toughest material in the book. Take it slowly — once it clicks, the rest of the LLM is comparatively easy.

::: objectives "What you'll learn"
- Why older RNNs struggled with long sequences, and how attention fixes it
- A simplified self-attention mechanism (no trainable weights) to build intuition
- Real self-attention with trainable **query, key, value** matrices — *scaled dot-product attention*
- **Causal (masked) attention** so the model can't peek at future tokens
- **Dropout** on attention weights to reduce overfitting
- **Multi-head attention** — running attention in parallel across many subspaces
:::

## The problem with modeling long sequences

Before transformers, translation used an **encoder–decoder RNN**: the encoder reads the whole input sentence and crams its meaning into a single final **hidden state**, which the decoder then unpacks word by word. The flaw: the decoder only sees that *one* compressed summary — it can't reach back to specific earlier words. For long sentences with long-range dependencies, context gets lost.

::: callout analogy "Summarizing a novel in one sentence"
An encoder–decoder RNN is like reading an entire novel, writing **one sentence** of notes, then handing only that sentence to a friend who must reproduce the whole story. Inevitably, details vanish. **Attention** lets the friend re-open the book and look at any page they need, as often as they need.
:::

The 2014 **Bahdanau attention** mechanism let the decoder *selectively* look back at all input words. Three years later, researchers realized you don't even need the RNN — and the **transformer** with **self-attention** was born.

## Capturing data dependencies with self-attention

**Self-attention** lets each position in a sequence look at — "attend to" — every other position, and decide how relevant each one is, when building its own representation. The "self" means it relates positions *within the same sequence* (unlike older attention that linked two separate sequences).

::: callout analogy "Resolving 'it' in a sentence"
Read *"The bee landed on the flower because **it** had pollen."* To know what "it" means, you weigh the other words and conclude "it" ≈ "flower." Self-attention does exactly this for **every** word at once: each token produces a blend of all tokens, weighted by relevance.
:::

The output for each token is a **context vector** — an enriched embedding that mixes in information from all the other tokens.

## A simple self-attention mechanism (no trainable weights)

Let's build intuition with a weight-free version. Our input is the sentence *"Your journey starts with one step"*, already embedded into 3-dimensional vectors:

```python title="The input sequence (6 tokens × 3 dims)"
import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89],  # Your    (x^1)
   [0.55, 0.87, 0.66],  # journey (x^2)
   [0.57, 0.85, 0.64],  # starts  (x^3)
   [0.22, 0.58, 0.33],  # with    (x^4)
   [0.77, 0.25, 0.10],  # one     (x^5)
   [0.05, 0.80, 0.55]]  # step    (x^6)
)
```

::: diagram ch03-attention-intro "Self-attention turns each input embedding into a context vector — a weighted sum of all input embeddings. Here we compute the context vector for the 2nd token, 'journey'."
:::

**Step 1 — attention scores.** Pick a *query* token (say the 2nd, "journey"). Its attention score against every token is the **dot product** of the query with that token:

```python title="Step 1: attention scores for the query"
query = inputs[1]                              # "journey"
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)
# tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
```

::: callout analogy "Dot product = how aligned two arrows are"
A **dot product** multiplies two vectors element-wise and sums the result — and it measures **similarity**: the more two vectors point the same way, the larger it is. So a high attention score means "these two tokens are alike / relevant to each other."
:::

**Step 2 — attention weights.** Normalize the scores so they're positive and sum to 1, using **softmax** (better gradients than plain division):

```python title="Step 2: normalize scores into weights"
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print(attn_weights_2)        # sums to 1.0
# tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
```

**Step 3 — context vector.** Multiply each input by its weight and sum — a **weighted average** of all tokens:

```python title="Step 3: the context vector for token 2"
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)
# tensor([0.4419, 0.6515, 0.5683])
```

### Computing all context vectors at once

For-loops are slow. The whole thing is just **matrix multiplication**:

```python title="All attention weights and context vectors"
attn_scores = inputs @ inputs.T          # [6, 6] all pairwise dot products
attn_weights = torch.softmax(attn_scores, dim=-1)   # normalize each row
all_context_vecs = attn_weights @ inputs # [6, 3] weighted sums
print(all_context_vecs[1])               # matches context_vec_2 above
```

`inputs @ inputs.T` gives a 6×6 matrix of every token's score against every other; softmax along `dim=-1` normalizes each **row** to sum to 1; multiplying by `inputs` produces all six context vectors. That's the simplified mechanism — no learning yet.

## Implementing self-attention with trainable weights

The real mechanism — **scaled dot-product attention** — adds three trainable weight matrices, $W_q$, $W_k$, $W_v$, that project each input into a **query**, a **key**, and a **value**. Training tunes these matrices so the model learns to build *useful* context vectors.

::: callout analogy "Query, key, value — like a search engine"
Borrow the terms from databases. The **query** is what you're searching for (the current token's "question"). Each token also exposes a **key** — a label advertising what it offers. You match your query against all keys to decide relevance, then pull the matching **values** — the actual content. A token attends most to the tokens whose *keys* best answer its *query*.
:::

::: diagram ch03-qkv "Each input token is projected by three learned weight matrices into a query, a key, and a value vector."
:::

Step by step for one token (input `d_in=3`, output `d_out=2`):

```python title="Project inputs into queries, keys, values"
x_2 = inputs[1]
d_in, d_out = inputs.shape[1], 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
keys    = inputs @ W_key       # keys/values for ALL tokens
values  = inputs @ W_value
```

The attention score is now the dot product of the **query** with each **key** (not the raw inputs). Then we **scale** by $\sqrt{d_k}$ before softmax:

```python title="Scores → scaled softmax → context vector"
attn_scores_2 = query_2 @ keys.T                 # query vs all keys
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)   # scale!
context_vec_2 = attn_weights_2 @ values          # weighted sum of VALUES
```

::: diagram ch03-scaled-steps "Trainable self-attention: project to Q/K/V, score query·keys, scale by √dₖ, softmax to weights, then take the weighted sum of the value vectors."
:::

The attention weight is $\text{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right)$, and the context vector is that times $V$.

::: callout math "Why divide by √dₖ?"
With large embedding dimensions, dot products grow large, which pushes softmax toward a near-one-hot spike — and a spiky softmax has **tiny gradients**, stalling training. Dividing by $\sqrt{d_k}$ keeps the scores in a sane range. This scaling is exactly why it's called **scaled** dot-product attention.
:::

### A compact self-attention class

Wrapping it in an `nn.Module`. The improved `v2` uses `nn.Linear` (better weight initialization, and it does the matrix multiply for us):

```python title="Listing 3.1 / 3.2 — Self-attention as a class"
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)
        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values
```

## Hiding future words with causal attention

A GPT generates text **left to right**, so when predicting the next token it must only look at tokens **at or before** the current position — never the future. **Causal (masked) attention** enforces this by zeroing out all attention weights *above the diagonal*.

::: diagram ch03-causal-mask "Causal attention masks the upper triangle: each token may attend only to itself and earlier tokens, never future ones."
:::

::: callout analogy "An exam with the next questions covered"
Causal masking is like taking a test where each question is revealed only after you answer the previous one — you can't peek ahead. Token 3 may use tokens 1–3, but tokens 4–6 are hidden. This is what makes next-word prediction honest.
:::

The clean trick: set the upper triangle to **−∞ before softmax**. Since $e^{-\infty}=0$, softmax assigns those positions exactly zero weight — and the remaining weights still sum to 1, so no renormalization is needed and **no future information leaks**:

```python title="The -inf masking trick"
context_length = attn_scores.shape[0]
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
```

### Masking additional weights with dropout

**Dropout** randomly zeroes some values during training so the model can't over-rely on any single connection. In attention, we drop a fraction of the attention weights (and scale the rest up to compensate). It's **only active during training**.

::: callout analogy "Benching random players at practice"
Dropout is like a coach randomly benching players during practice. The team learns not to depend on any one star and builds redundant strengths — so on game day (inference, dropout off) it's more robust. Typical LLM dropout is a gentle 0.1.
:::

### A compact causal attention class

This handles **batches** and registers the mask as a buffer (so it moves to GPU with the model):

```python title="Listing 3.3 — CausalAttention"
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)              # batched
        attn_scores.masked_fill_(                                  # causal mask
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ values
```

## Extending single-head to multi-head attention

A single attention computation is **one head**. **Multi-head attention** runs several in parallel, each with its own Q/K/V matrices, then concatenates their outputs. Each head can specialize — one might track grammar, another long-range references.

::: callout analogy "A panel of specialists"
Multi-head attention is like asking a **panel of experts** to each read the sentence with a different lens — a grammarian, a topic expert, a pronoun-tracker — then merging their notes. More heads = more relationships captured at once.
:::

::: diagram ch03-multihead "Multi-head attention runs several attention heads in parallel, each producing its own context vectors, which are concatenated into the final output."
:::

The intuitive (but slower) version just stacks `CausalAttention` modules:

```python title="Listing 3.4 — MultiHeadAttentionWrapper (stacking)"
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

The **efficient** version uses a single set of larger matrices and *splits* them into heads via tensor reshaping — one matrix multiply instead of one-per-head:

```python title="Listing 3.5 — MultiHeadAttention (weight splits)"
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads          # split d_out across heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)     # combine heads
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)
        # split d_out into (num_heads, head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # -> (b, num_heads, num_tokens, head_dim)
        keys, queries, values = keys.transpose(1, 2), queries.transpose(1, 2), values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)             # per-head scores
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)    # back to (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # combine heads
        return self.out_proj(context_vec)                        # final projection
```

::: callout note "Why the efficient version is faster"
Both produce the same result. But the wrapper repeats the (expensive) Q/K/V matrix multiplications *once per head*; the `MultiHeadAttention` class does **one** big multiply, then reshapes with `.view`/`.transpose` to split the result into heads. PyTorch's batched matrix multiplication then handles all heads at once.
:::

For scale: the smallest GPT-2 (117M params) uses **12 heads** and an embedding size of **768**; in GPT models the input and output embedding sizes are equal (`d_in = d_out`). This `MultiHeadAttention` class is the exact component we'll drop into the GPT model in Chapter 4.

## Key takeaways

::: takeaways
- **Attention** builds a **context vector** for each token: a weighted sum of all tokens, where weights reflect relevance.
- In **simplified** self-attention, scores are raw **dot products** of input embeddings; softmax turns them into weights that sum to 1.
- Real **scaled dot-product attention** projects inputs into **queries, keys, values** with trainable matrices, scores query·key, scales by $\sqrt{d_k}$, softmaxes, and weights the **values**.
- The $\sqrt{d_k}$ scaling prevents large dot products from making softmax too peaky (which would kill gradients).
- **Causal attention** masks future tokens (upper triangle → −∞ before softmax) so the model can't cheat when predicting the next token.
- **Dropout** on attention weights reduces overfitting and is active only during training.
- **Multi-head attention** runs several attention heads in parallel and concatenates them; the efficient implementation splits one big projection into heads with `.view`/`.transpose`.
:::

## Additional references

::: refs
- [Chapter 3 code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03) — GitHub · all attention implementations from this chapter.
- [Attention in transformers, step by step](https://www.3blue1brown.com/lessons/attention/) — Video · 3Blue1Brown's stunning visual breakdown of Q/K/V attention.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Blog · the canonical visual explanation of multi-head attention.
- [Understanding and Coding Self-Attention from Scratch](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) — Blog · the author's own deep dive, mirroring this chapter.
- [Let's build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Video · Andrej Karpathy codes attention live.
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — Blog · Lilian Weng's thorough survey of attention variants.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Paper · the original scaled dot-product + multi-head attention.
:::

## Test your knowledge

```flashcards
Q: What is a "context vector" in self-attention?
A: An enriched embedding for a token — a weighted sum of all tokens' value vectors, where the weights are the attention weights.
---
Q: What three vectors does scaled dot-product attention compute for each token?
A: A **query** (what it's looking for), a **key** (what it offers), and a **value** (its content). All via trainable matrices Wq, Wk, Wv.
---
Q: How is an attention score computed in trainable self-attention?
A: As the dot product of one token's **query** with another token's **key**.
---
Q: Why divide attention scores by √dₖ before softmax?
A: Large dot products make softmax too peaky, producing near-zero gradients. Scaling keeps gradients healthy — hence "scaled" dot-product attention.
---
Q: How does causal attention prevent peeking at future tokens?
A: It sets attention scores above the diagonal to −∞ before softmax, so those (future) positions get exactly zero weight.
---
Q: When is dropout active, and what does it do in attention?
A: Only during **training**. It randomly zeroes a fraction of attention weights (rescaling the rest) to reduce overfitting.
---
Q: What does multi-head attention give you over a single head?
A: Multiple heads attend to different representation subspaces in parallel (e.g., grammar vs. long-range references), capturing richer relationships.
---
Q: Why is the weight-split MultiHeadAttention more efficient than stacking CausalAttention modules?
A: It computes Q/K/V with a single large matrix multiply and reshapes into heads, instead of repeating the costly multiply once per head.
```

```quiz
1. The attention weight matrix is computed as:
   - ( ) softmax(Q + K) · V
   - (x) softmax(QKᵀ / √dₖ) · V
   - ( ) Q · K · V
   - ( ) softmax(V) · Q
   > Scores are QKᵀ, scaled by √dₖ, softmaxed into weights, then applied to V.

2. In causal attention, which positions are masked for the token at position i?
   - ( ) all positions except i
   - ( ) positions before i
   - (x) all positions after i (the future)
   - ( ) no positions
   > Causal masking hides future tokens (those after the current position), keeping only positions ≤ i.

3. Why set masked positions to −∞ rather than 0 before softmax?
   - (x) Because e^(−∞)=0, so softmax gives them zero weight and the row still sums to 1 with no renormalization
   - ( ) To make the scores larger
   - ( ) Because softmax cannot handle zeros
   - ( ) To speed up matrix multiplication
   > −∞ becomes 0 after softmax automatically, avoiding a separate renormalization step and any information leakage.

4. If d_out=768 and num_heads=12, what is head_dim?
   - ( ) 768
   - ( ) 12
   - (x) 64
   - ( ) 780
   > head_dim = d_out / num_heads = 768 / 12 = 64.

5. A dot product between two vectors is large when…
   - (x) the vectors are well aligned (similar direction)
   - ( ) the vectors are orthogonal
   - ( ) the vectors point in opposite directions
   - ( ) one vector is all zeros
   > The dot product measures alignment/similarity; aligned vectors yield a high value.
```

```assignment "Exercise 3.3 — GPT-2-sized attention" level=intermediate
Instantiate a `MultiHeadAttention` module matching the **smallest GPT-2**: 12 heads, input and output embedding dimension 768, and a context length of 1,024. Confirm it runs on a random input batch of shape `[2, 1024, 768]` and that the output shape is also `[2, 1024, 768]`.

Hint: `MultiHeadAttention(d_in=768, d_out=768, context_length=1024, dropout=0.1, num_heads=12)`.
Hint: feed it `torch.rand(2, 1024, 768)` and print `out.shape`.
```
