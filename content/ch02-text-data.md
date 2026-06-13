An LLM is a machine that does math on numbers — but text is made of words and punctuation, not numbers. So before any "learning" can happen, we have to translate raw text into a numerical form the network can chew on. This chapter builds that translation pipeline end to end: split text into **tokens**, map tokens to **token IDs**, batch them with a **sliding window**, and finally turn them into **embedding vectors** carrying both *meaning* and *position*.

By the end you'll have the complete **input pipeline** that feeds the GPT model we build in the next chapters.

::: objectives "What you'll learn"
- Why neural networks need **embeddings**, and what an embedding actually is
- How to split text into tokens and build a **vocabulary** of token IDs
- Special context tokens like `<|unk|>` and `<|endoftext|>`, and why they matter
- **Byte pair encoding (BPE)** — the subword tokenizer behind GPT
- Sampling input–target pairs with a **sliding window** and a PyTorch `DataLoader`
- Turning token IDs into **token embeddings** and adding **positional embeddings**
:::

::: callout note "Where we are in the build"
This is **Stage 1, step 1**: the data pipeline. We're preparing the fuel; the engine (attention + the GPT architecture) comes in Chapters 3–4.
:::

## Understanding word embeddings

Deep neural networks can't process raw text — text is *categorical*, and networks do math on *continuous numbers*. The fix is an **embedding**: a mapping from discrete objects (words) to points in a continuous vector space.

::: callout analogy "A map of meaning"
Think of an embedding as a **GPS coordinate for meaning**. Just as a map places Paris near Lyon and far from Tokyo, an embedding space places *"cat"* near *"dog"* and far from *"thermodynamics."* Words used in similar contexts end up as neighbors. The famous early method **Word2Vec** learned exactly this — and its space even captured analogies like *king − man + woman ≈ queen*.
:::

::: diagram ch02-embedding-space "In a (2-D) embedding space, similar words cluster together — birds near birds, countries near their capitals."
:::

Embeddings can have anywhere from one to thousands of dimensions. More dimensions can capture subtler relationships, but cost more compute. For scale: the smallest **GPT-2** uses **768** dimensions; the largest **GPT-3** uses **12,288**.

While you *can* use pretrained embeddings like Word2Vec, **LLMs learn their own embeddings** as part of the input layer, optimized during training so they're tuned to the exact task. We'll create such an embedding layer at the end of this chapter.

## Tokenizing text

**Tokenizing** means splitting text into individual units — words or punctuation. We'll use a public-domain short story, *"The Verdict"* by Edith Wharton, as our practice text.

```python title="Listing 2.1 — Load the sample text"
import urllib.request

url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
urllib.request.urlretrieve(url, "the-verdict.txt")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of characters:", len(raw_text))
print(raw_text[:99])
```

This prints `20479` characters. Now, how do we split it? A quick tour of Python's `re` (regular expression) module shows the idea — we split on whitespace **and** punctuation, then drop empty/whitespace entries:

```python title="Building up a regex tokenizer"
import re

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)
# ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

::: callout analogy "Snapping text into LEGO bricks"
Tokenizing is like breaking a sentence into individual **LEGO bricks** — each word and each punctuation mark becomes its own reusable piece. Later, the model studies how the bricks fit together. (Note we *keep* capitalization: "Apple" the company vs "apple" the fruit carry different meaning.)
:::

Applying this to the whole story yields **4,690 tokens**:

```python title="Tokenize the full story"
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))   # 4690
print(preprocessed[:30])
```

::: diagram ch02-tokenization "Raw text is split into a flat list of word and punctuation tokens."
:::

## Converting tokens into token IDs

Tokens are still strings. To feed them to a network we map each unique token to an integer — a **token ID**. The set of all unique tokens, sorted and numbered, is the **vocabulary**.

```python title="Listing 2.2 — Build a vocabulary"
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)          # 1130 unique tokens
vocab = {token: integer for integer, token in enumerate(all_words)}
```

::: diagram ch02-vocab "Tokenize the training text, keep the unique tokens sorted alphabetically, and assign each one an integer ID. That mapping is the vocabulary."
:::

Now we wrap encoding/decoding in a class. `encode` turns text → IDs; `decode` turns IDs → text (and cleans up spaces before punctuation):

```python title="Listing 2.3 — A simple tokenizer (V1)"
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)   # tidy punctuation
        return text
```

It round-trips nicely on text from the story. But feed it a word the story never used —

```python title="The out-of-vocabulary problem"
tokenizer = SimpleTokenizerV1(vocab)
tokenizer.encode("Hello, do you like tea?")
# KeyError: 'Hello'
```

— and it **crashes**, because `"Hello"` isn't in the vocabulary. Real text always contains unseen words, so we need a fix.

## Adding special context tokens

We extend the vocabulary with two **special tokens**:

- `<|unk|>` — a stand-in for any **unknown** word (not in the vocabulary).
- `<|endoftext|>` — a **separator** placed between unrelated documents so the model knows "this is a fresh start."

::: diagram ch02-special-tokens "Special tokens extend the vocabulary: <|unk|> stands in for unknown words; <|endoftext|> separates unrelated documents."
:::

::: callout analogy "Editor's marks"
These are like an editor's marks. `<|endoftext|>` is the **chapter break** that says "a new, unrelated passage starts here." `<|unk|>` is the **[illegible]** stamp for a word the reader has never seen. They give the model structural cues beyond the words themselves.
:::

```python title="Listing 2.4 — Tokenizer V2 (handles unknown words)"
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>"
                        for item in preprocessed]                       # NEW
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
```

Other tokenizers use extra special tokens — `[BOS]` (beginning of sequence), `[EOS]` (end of sequence), `[PAD]` (padding, to make batched texts equal length). The **GPT tokenizer keeps it minimal**: it uses only `<|endoftext|>` (which doubles as a separator *and* padding), and — crucially — it needs no `<|unk|>` token at all, thanks to **byte pair encoding**.

## Byte pair encoding

**Byte pair encoding (BPE)** is the tokenizer behind GPT-2, GPT-3, and the original ChatGPT. Instead of whole words, it works with **subword units**. Implementing BPE from scratch is involved, so we use OpenAI's fast `tiktoken` library:

```python title="Using the GPT-2 BPE tokenizer"
# pip install tiktoken
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace.")
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(ids)
print(tokenizer.decode(ids))
```

Two things stand out:

1. `<|endoftext|>` gets the ID **50256** — the BPE vocabulary has **50,257** tokens, and `<|endoftext|>` is the very last one.
2. The made-up word `someunknownPlace` encodes and decodes **correctly**, with no `<|unk|>` needed.

How? BPE breaks unknown words into smaller subwords or even single characters. It builds its vocabulary by **iteratively merging the most frequent adjacent pairs** — starting from single characters ("a", "b", …), merging frequent pairs ("d"+"e" → "de"), then frequent subwords into words.

::: diagram ch02-bpe "BPE breaks an unknown word into known subword pieces, so any word — even gibberish — can be encoded without an <|unk|> token."
:::

::: callout analogy "Sounding out a new word"
BPE is like **sounding out an unfamiliar word** by syllables. You've never seen "Akwirw," but you can still pronounce it as *Ak-w-ir-w* from familiar pieces. BPE does the same: it always has the single characters as a fallback, so it can spell *anything*.
:::

```assignment "Exercise 2.1 — BPE on unknown words" level=beginner
Run the `tiktoken` GPT-2 tokenizer on the string `"Akwirw ier"`. Print the token IDs, then call `decode` on each individual integer to see the subword pieces, and finally `decode` the whole list to confirm it reconstructs the original string.

Hint: `tokenizer.encode("Akwirw ier")` gives the IDs; loop over them and `tokenizer.decode([id])` each one.
```

## Data sampling with a sliding window

LLMs train by predicting the **next** token. So we need **input–target pairs**, where the target is the input shifted right by one position. We generate these by sliding a fixed-size window across the tokenized text.

First, tokenize the whole story with BPE (it yields 5,145 tokens), and look at one window of size 4:

```python title="Input–target pairs (shift by one)"
enc_text = tokenizer.encode(raw_text)
enc_sample = enc_text[50:]           # skip ahead for a more interesting passage

context_size = 4
x = enc_sample[:context_size]        # inputs
y = enc_sample[1:context_size + 1]   # targets = inputs shifted by 1

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
# " and"                    ---->  " established"
# " and established"        ---->  " himself"
# " and established himself ---->  " in"
# ...
```

Everything left of the arrow is what the model sees; the token on the right is what it must predict.

::: diagram ch02-sliding-window "A sliding window slices the token stream into input chunks; each target is the input shifted one token to the right. The stride controls how far the window jumps between samples."
:::

Now we wrap this in PyTorch's `Dataset`/`DataLoader` machinery for efficient batching:

```python title="Listing 2.5 — A Dataset of input/target chunks"
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)              # tokenize the whole text
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]   # shifted by 1
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

```python title="Listing 2.6 — A DataLoader that batches the pairs"
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last,        # drop a final undersized batch (avoids loss spikes)
        num_workers=num_workers,    # CPU worker processes
    )
    return dataloader
```

The **`stride`** is the key knob: it's how far the window jumps between samples. A `stride` equal to `max_length` means **no overlap** between consecutive samples (we use the whole dataset without repeating tokens — less overfitting); a smaller stride overlaps more.

```python title="A batch of 8, context 4, no overlap"
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
inputs, targets = next(iter(dataloader))
print("Inputs shape:", inputs.shape)   # torch.Size([8, 4])
```

::: callout analogy "Reading with a finger over the next word"
The sliding window is like learning to read while a friend covers the rest of the line with their finger: you see *"The cat sat on the"*, guess *"mat,"* then the finger slides one word right and you guess again. **Stride** is how many words the finger jumps each time.
:::

::: callout warning "Why drop_last and stride matter"
`drop_last=True` discards a final partial batch so every batch is the same size (a too-small batch can cause unstable, spiky loss). And setting `stride = max_length` avoids overlapping samples that would otherwise make the model see the same tokens repeatedly and overfit.
:::

## Creating token embeddings

Token IDs are still just integers — and integer *size* is meaningless (ID 50256 isn't "bigger" than ID 5 in any useful sense). We convert each ID into a learnable vector with an **embedding layer**, which starts as random numbers and gets optimized during training.

A tiny example — vocabulary of 6, embedding size 3:

```python title="An embedding layer is a lookup table"
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size, output_dim = 6, 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)        # a 6 x 3 matrix of random values

print(embedding_layer(torch.tensor([3])))   # -> the 4th row (index 3)
print(embedding_layer(input_ids))            # -> a 4 x 3 matrix
```

The `Embedding` layer is **just a lookup table**: token ID `i` returns row `i` of the weight matrix. (It's mathematically equivalent to one-hot encoding followed by a matrix multiply, but far more efficient — and fully trainable via backpropagation.)

::: diagram ch02-embedding-lookup "An embedding layer is a lookup table: each token ID indexes a row of the weight matrix to retrieve that token's embedding vector."
:::

::: callout analogy "The coat-check counter"
An embedding layer is a **coat-check counter**. You hand over a numbered ticket (the token ID) and get back the specific coat stored in that slot (the embedding vector). Training is the staff gradually swapping the coats in each slot until every ticket returns the *most useful* vector for the task.
:::

## Encoding word positions

There's a subtle gap. The embedding layer maps a token ID to the **same vector regardless of where it appears** in the sequence. But word *order* carries meaning — *"the dog bit the man"* ≠ *"the man bit the dog"* — and the attention mechanism (Chapter 3) is itself **position-agnostic**. So we must inject position information.

::: callout analogy "Seat numbers at a dinner table"
Token embeddings tell you *who* is at the table; positional embeddings tell you *which seat* they're in. Without seat numbers, *"dog bites man"* and *"man bites dog"* look identical to the model — same guests, scrambled order. Adding a position vector restores the seating chart.
:::

There are two families: **absolute** positional embeddings (a distinct vector per position) and **relative** ones (encoding the *distance* between tokens). **GPT uses absolute positional embeddings that are learned during training.** We create a second embedding layer indexed by position and **add** it to the token embeddings.

::: diagram ch02-input-pipeline "The full input pipeline: text → tokens → token IDs → token embeddings, plus positional embeddings of the same size, summed into the final input embeddings."
:::

Putting it together at a realistic scale — vocabulary 50,257, embedding size 256:

```python title="Token + positional embeddings → input embeddings"
vocab_size, output_dim = 50257, 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False)
inputs, targets = next(iter(dataloader))      # inputs: [8, 4]

token_embeddings = token_embedding_layer(inputs)        # [8, 4, 256]

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))   # [4, 256]

input_embeddings = token_embeddings + pos_embeddings    # broadcast -> [8, 4, 256]
print(input_embeddings.shape)   # torch.Size([8, 4, 256])
```

PyTorch **broadcasts** the `[4, 256]` positional tensor across all 8 sequences in the batch, adding the same position vectors to each. The resulting `[8, 4, 256]` tensor — 8 sequences × 4 tokens × 256 dims — is exactly what the LLM's main layers consume. That's the finished input pipeline.

## Key takeaways

::: takeaways
- Neural networks need numbers, so text is converted to **embeddings** — vectors in a continuous space where similar meanings sit close together.
- Text is first split into **tokens** (words/punctuation), which are mapped to integer **token IDs** via a **vocabulary**.
- **Special tokens** like `<|unk|>` (unknown) and `<|endoftext|>` (document separator) add structure; GPT uses only `<|endoftext|>`.
- **Byte pair encoding (BPE)** tokenizes into **subwords**, so it can encode *any* word (even unseen ones) without an `<|unk|>` token. GPT's BPE vocab is **50,257**.
- Training data is created as **input–target pairs** with a **sliding window**; the target is the input shifted by one token. **`stride`** controls overlap; PyTorch `Dataset`/`DataLoader` batch it efficiently.
- An **embedding layer** is a trainable **lookup table** mapping each token ID to a vector.
- Because attention is order-agnostic, GPT **adds learned absolute positional embeddings** to the token embeddings. The sum is the final **input embedding**.
:::

## Additional references

::: refs
- [Chapter 2 code & the-verdict.txt](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02) — GitHub · the official notebooks for this chapter.
- [tiktoken — OpenAI's BPE tokenizer](https://github.com/openai/tiktoken) — Library · the fast BPE implementation used here.
- [Tiktokenizer (interactive)](https://tiktokenizer.vercel.app/) — Tool · paste text and watch GPT tokenizers split it in real time. Great for visual learners.
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) — Blog · Jay Alammar's visual guide to embeddings and analogies.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — Docs · BPE, WordPiece, and Unigram compared.
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — Paper · Sennrich et al., the paper that introduced BPE for NLP.
- [Efficient Estimation of Word Representations (Word2Vec)](https://arxiv.org/abs/1301.3781) — Paper · Mikolov et al.
:::

## Test your knowledge

```flashcards
Q: Why can't an LLM process raw text directly?
A: Networks do math on continuous numbers; text is categorical. We convert tokens to numeric **embedding vectors** first.
---
Q: What is the difference between a token and a token ID?
A: A **token** is a string piece (word/subword/punctuation); a **token ID** is the unique integer that token maps to in the vocabulary.
---
Q: What does the `<|endoftext|>` token do?
A: It separates unrelated documents (and serves as padding for GPT), signaling boundaries between concatenated texts.
---
Q: How does BPE handle a word it has never seen?
A: It breaks the word into smaller **subword units or individual characters** from its vocabulary, so it never needs an `<|unk|>` token.
---
Q: In next-word-prediction data, how is the target sequence related to the input?
A: The target is the input sequence **shifted right by one token** — each position's label is the next token.
---
Q: What does the `stride` parameter control in the sliding-window data loader?
A: How many tokens the window moves between samples. `stride = max_length` gives non-overlapping samples.
---
Q: An `nn.Embedding` layer is conceptually what kind of operation?
A: A **lookup table** — token ID *i* returns row *i* of a trainable weight matrix (equivalent to one-hot × matrix multiply).
---
Q: Why are positional embeddings added to token embeddings?
A: Attention is order-agnostic, so position vectors inject **where** each token sits; without them, word order would be invisible to the model.
```

```quiz
1. What is the GPT-2/GPT-3 BPE vocabulary size?
   - ( ) 768
   - ( ) 12,288
   - (x) 50,257
   - ( ) 1,130
   > The BPE vocabulary has 50,257 tokens, with `<|endoftext|>` assigned the last ID (50256).

2. Which statement about an `nn.Embedding` layer is TRUE?
   - ( ) It is a fixed, non-trainable hash function
   - (x) It is a trainable lookup table optimized during training
   - ( ) It converts embeddings back into text
   - ( ) It requires one-hot vectors as input
   > The embedding layer's weight matrix starts random and is optimized by backpropagation; you index it with integer token IDs.

3. If `max_length=4` and `stride=4`, consecutive training samples will…
   - (x) not overlap at all
   - ( ) overlap by 3 tokens
   - ( ) be identical
   - ( ) overlap by 1 token
   > A stride equal to the window length advances the window by a full window, so samples don't share tokens.

4. Why does GPT add positional embeddings?
   - ( ) To increase the vocabulary size
   - ( ) To compress the sequence
   - (x) Because the attention mechanism has no built-in sense of token order
   - ( ) To replace token embeddings
   > Self-attention is position-agnostic, so order information must be injected via positional embeddings added to the token embeddings.

5. What shape results from embedding a batch of 8 sequences of 4 tokens into 256-dim vectors?
   - ( ) [4, 256]
   - ( ) [8, 256]
   - (x) [8, 4, 256]
   - ( ) [8, 4]
   > Batch (8) × tokens per sequence (4) × embedding dimension (256).
```

```assignment "Build a mini input pipeline" level=intermediate
Using `tiktoken`'s GPT-2 encoder and PyTorch, write a short script that: (1) encodes a paragraph of your choice, (2) creates input–target pairs with `max_length=8`, `stride=8` using a `Dataset`/`DataLoader`, (3) embeds the inputs with a `vocab_size=50257`, `output_dim=128` embedding layer, and (4) adds positional embeddings. Print the final `input_embeddings.shape` and confirm it equals `[batch_size, 8, 128]`.

Hint: reuse `GPTDatasetV1` and `create_dataloader_v1` from Listings 2.5–2.6.
Hint: the positional layer is `nn.Embedding(context_length, output_dim)` indexed by `torch.arange(context_length)`.
```
