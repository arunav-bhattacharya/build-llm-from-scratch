Each chapter of *Build a Large Language Model (From Scratch)* sprinkles in short exercises that ask you to extend the code you just wrote — pad inputs differently, resize the model, swap a prompt format, fine-tune more layers. This appendix collects **worked solutions** to those exercises, with the reasoning spelled out so a solution never feels like a magic trick.

Solutions are grouped by chapter, with one heading per exercise. Code listings are transcribed from the book; the surrounding prose explains *why* each change has the effect it does. The complete runnable code for every answer also lives in the book's [GitHub repository](https://github.com/rasbt/LLMs-from-scratch).

::: callout tip "Try first, peek second"
You'll learn far more by attempting each exercise *before* reading the solution. The exercises are deliberately small — most are a one- or two-line change to code already in the chapter. Struggle for a few minutes first; the solution will stick much better once you've formed your own hypothesis.
:::

::: objectives "What you'll find here"
- Verbatim **solution code** for every exercise in chapters 2 through 7
- An explanation of the **reasoning** behind each answer, not just the code
- The **measured results** (accuracies, parameter counts, timings) the author observed, so you can check your own
:::

## Chapter 2

### Exercise 2.1 — Byte-pair encoding of unknown words

The goal is to tokenize the unfamiliar string `"Akwirw ier"` with the GPT-2 BPE tokenizer, then reconstruct it. Because BPE breaks unknown words into known subword (or single-character) tokens, you can probe it one piece at a time and confirm there are no out-of-vocabulary failures.

Prompt the encoder with one substring at a time to read off the individual token IDs:

```python title="Solution 2.1 — encode piece by piece"
print(tokenizer.encode("Ak"))
print(tokenizer.encode("w"))
# ...
# [33901]
# [86]
# ...
```

Then decode the full list of IDs to reassemble the original string exactly:

```python title="Solution 2.1 — decode back to text"
print(tokenizer.decode([33901, 86, 343, 86, 220, 959]))
# 'Akwirw ier'
```

The lesson: BPE never fails on an unseen word — in the worst case it falls back to single bytes/characters — and decoding is a perfect inverse of encoding.

### Exercise 2.2 — Data loaders with different strides

This exercise asks you to observe how `max_length` and `stride` shape the batches produced by `create_dataloader`. Recall that `max_length` is the number of tokens per sample and `stride` is how far the sliding window advances between samples.

With `max_length=2` and `stride=2`, consecutive samples don't overlap:

```python title="Solution 2.2 — max_length=2, stride=2"
dataloader = create_dataloader(
    raw_text, batch_size=4, max_length=2, stride=2
)
# tensor([[  40,  367],
#         [2885, 1464],
#         [1807, 3619],
#         [ 402,  271]])
```

With `max_length=8` and `stride=2`, each sample is longer and successive samples **overlap heavily** (the window slides only 2 tokens while spanning 8):

```python title="Solution 2.2 — max_length=8, stride=2"
dataloader = create_dataloader(
    raw_text, batch_size=4, max_length=8, stride=2
)
# tensor([[   40,   367,  2885,  1464,  1807,  3619,   402,   271],
#         [ 2885,  1464,  1807,  3619,   402,   271, 10899,  2138],
#         [ 1807,  3619,   402,   271, 10899,  2138,   257,  7026],
#         [  402,   271, 10899,  2138,   257,  7026, 15632,   438]])
```

The takeaway: a stride smaller than `max_length` reuses tokens across samples (more training signal, more redundancy); a stride equal to `max_length` partitions the text with no overlap.

## Chapter 3

### Exercise 3.1 — Comparing `SelfAttention_v1` and `SelfAttention_v2`

`SelfAttention_v1` stores its weights as raw `nn.Parameter` matrices, while `v2` uses `nn.Linear` layers. The two initialize their weights differently, so they normally produce different outputs. To make `v1` reproduce `v2`'s output, copy `v2`'s weights into `v1` — but transpose them, because `nn.Linear` stores its weight matrix in transposed form relative to the manual `@` convention used in `v1`:

```python title="Solution 3.1 — transfer weights (note the transpose)"
sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_key   = torch.nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)
```

The `.T` is the whole point: it reminds you that `nn.Linear(d_in, d_out)` holds a `[d_out, d_in]` weight and computes `x @ W.T`, whereas `v1` computes `x @ W` with a `[d_in, d_out]` parameter.

### Exercise 3.2 — Returning two-dimensional embedding vectors

`MultiHeadAttentionWrapper` concatenates the outputs of its heads, so its output dimension is `num_heads × d_out`. To match the single-head output dimension of 2 while using `num_heads=2`, set each head's projection dimension `d_out` to **1** (so 2 × 1 = 2):

```python title="Solution 3.2 — set d_out=1 so 2 heads concatenate to dim 2"
d_out = 1
mha = MultiHeadAttentionWrapper(d_in, d_out, block_size, 0.0, num_heads=2)
```

### Exercise 3.3 — Initializing GPT-2-size attention modules

Configure the `MultiHeadAttention` class to match the smallest GPT-2 model: a context length of 1,024, input and output embedding dimensions of 768, and 12 heads.

```python title="Solution 3.3 — smallest GPT-2 attention"
block_size = 1024
d_in, d_out = 768, 768
num_heads = 12
mha = MultiHeadAttention(d_in, d_out, block_size, 0.0, num_heads)
```

With `d_out=768` split across 12 heads, each head has dimension `768 / 12 = 64` — the head size used throughout GPT-2.

## Chapter 4

### Exercise 4.1 — Parameters in the feed-forward vs. attention modules

Count the parameters in a single transformer block's feed-forward (`ff`) and attention (`att`) submodules:

```python title="Solution 4.1 — count parameters per submodule"
block = TransformerBlock(GPT_CONFIG_124M)

total_params = sum(p.numel() for p in block.ff.parameters())
print(f"Total number of parameters in feed forward module: {total_params:,}")

total_params = sum(p.numel() for p in block.att.parameters())
print(f"Total number of parameters in attention module: {total_params:,}")

# Total number of parameters in feed forward module: 4,722,432
# Total number of parameters in attention module: 2,360,064
```

The feed-forward module holds roughly **twice** as many parameters as attention — a reminder that most of a transformer block's weight (and compute, at modest context lengths) lives in the MLP, not the attention.

### Exercise 4.2 — Initializing larger GPT models

The four GPT-2 sizes differ only in three config values: embedding dimension, number of layers, and number of heads. To build any of them, copy the 124M config and override those three fields. Shown here for GPT-2 XL:

```python title="Solution 4.2 — GPT-2 XL configuration"
GPT_CONFIG = GPT_CONFIG_124M.copy()
GPT_CONFIG["emb_dim"]  = 1600
GPT_CONFIG["n_layers"] = 48
GPT_CONFIG["n_heads"]  = 25
model = GPTModel(GPT_CONFIG)
```

Reusing the parameter-counting code from section 4.6 then gives, for GPT-2 XL:

```text
Total number of parameters: 1,637,792,000
Number of trainable parameters considering weight tying: 1,557,380,800
Total size of the model: 6247.68 MB
```

(GPT-2 Large uses `emb_dim=1280, n_layers=36, n_heads=20`; GPT-2 Medium uses `emb_dim=1024, n_layers=24, n_heads=16`.)

### Exercise 4.3 — Separate dropout rates

Chapter 4 uses dropout in three places — the embedding layer, the shortcut connections, and multi-head attention — all sharing one rate. To control them independently, split `drop_rate` into three config keys and wire each into the right layer.

The modified configuration:

```python title="Solution 4.3 — three independent dropout rates"
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate_attn": 0.1,        # dropout for multi-head attention
    "drop_rate_shortcut": 0.1,    # dropout for shortcut connections
    "drop_rate_emb": 0.1,         # dropout for the embedding layer
    "qkv_bias": False
}
```

The `TransformerBlock` and `GPTModel` then read the appropriate key for each dropout layer:

```python title="Solution 4.3 — TransformerBlock and GPTModel" collapsible
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate_attn"],        # attention dropout
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(
            cfg["drop_rate_shortcut"]             # shortcut dropout
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate_emb"])   # embedding dropout
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

## Chapter 5

### Exercise 5.1 — Counting sampled tokens at different temperatures

Using the `print_sampled_tokens` function from section 5.3.1, count how often the word **"pizza"** is drawn at various temperatures. At temperature 0 or 0.1 it is sampled **0×**; scaled up to temperature 5 it is sampled **32×** out of 1,000 draws — an estimated probability of 32/1000 = **3.2%**. The actual probability is **4.3%**, found at `scaled_probas[2][6]` in the rescaled softmax tensor. Higher temperature flattens the distribution, so rare tokens like "pizza" start to appear.

### Exercise 5.2 — Choosing temperature and top-k

These settings trade determinism against creativity, and the right choice depends on the application:

- **Low top-k (< 10) and temperature below 1** → less random, more deterministic output. Best when you need predictable, coherent, factual text: formal documents and reports, technical analysis, code generation, and question answering or educational content.
- **Higher top-k (20–40) and temperature above 1** → more diverse, surprising output. Best for brainstorming and creative writing such as fiction.

There is no single "correct" pair — tune them to the desired degree of randomness for your task.

### Exercise 5.3 — Deterministic behavior in the `generate` function

There are two ways to make `generate` fully deterministic:

1. Set `top_k=None` **and** apply no temperature scaling (so the model always takes the argmax).
2. Set `top_k=1` (which leaves only the single highest-probability token to choose from).

Either route removes all randomness from decoding.

### Exercise 5.4 — Continuing pretraining from a checkpoint

To resume training, reload both the model **and** the optimizer state saved in the main chapter, then call the training function again:

```python title="Solution 5.4 — restore model + optimizer, train one more epoch"
checkpoint = torch.load("model_and_optimizer.pth")

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

Restoring the optimizer (not just the model) matters because AdamW carries running moment estimates; dropping them would disrupt training. Then call `train_simple_function` with `num_epochs=1` to train one more epoch.

### Exercise 5.5 — Training and validation loss of the pretrained model

Compute the loss of OpenAI's pretrained GPT-2 weights on "The Verdict":

```python title="Solution 5.5 — losses with pretrained GPT-2 weights"
train_loss = calc_loss_loader(train_loader, gpt, device)
val_loss   = calc_loss_loader(val_loader, gpt, device)

# Training loss:   3.754748503367106
# Validation loss: 3.559617757797241
```

The two losses are in the same ballpark, which is the key observation. Either "The Verdict" was **not** in GPT-2's pretraining data (so the model isn't overfitting and performs similarly on both splits — the slightly *lower* validation loss is just noise on a small dataset), or it **was**, in which case the validation split can't reveal overfitting because it was effectively trained on too. Distinguishing the two would require a dataset created after GPT-2 finished training.

### Exercise 5.6 — Loading the largest GPT-2 model

The chapter uses the 124M model to keep resource use low, but switching to GPT-2 XL (1,558M) is a **two-line** change — the model name and the size argument to the loader:

```python title="Solution 5.6 — load GPT-2 XL instead of small"
hparams, params = download_and_load_gpt2(
    model_size="1558M", models_dir="gpt2")
model_name = "gpt2-xl (1558M)"
```

Everything downstream works unchanged because all GPT-2 sizes share the same architecture.

## Chapter 6

### Exercise 6.1 — Padding to the maximum context length

Instead of padding each batch to its longest sequence, pad every sample to the model's full 1,024-token context by passing `max_length=1024` when constructing the datasets:

```python title="Solution 6.1 — pad to the full context length"
train_dataset = SpamDataset(..., max_length=1024, ...)
val_dataset   = SpamDataset(..., max_length=1024, ...)
test_dataset  = SpamDataset(..., max_length=1024, ...)
```

The result is **worse**: test accuracy drops to **78.33%** (vs. 95.67% in the chapter). The extra padding dilutes the signal, showing that padding only as much as needed is the better default.

### Exercise 6.2 — Fine-tuning the whole model

Rather than fine-tuning only the final transformer block, unfreeze the entire model by **removing** the lines that disable gradients:

```python title="Solution 6.2 — remove this freezing code to train all layers"
for param in model.parameters():
    param.requires_grad = False
```

With every layer trainable, test accuracy improves by about 1 point to **96.67%** (vs. 95.67%). Fine-tuning more layers helps here, at the cost of more compute.

### Exercise 6.3 — Fine-tuning the first vs. last output token

The chapter classifies using the **last** token's output. To use the **first** token instead, change `model(input_batch)[:, -1, :]` to `model(input_batch)[:, 0, :]` everywhere in the code. Because the first token (with causal attention) has seen *less* of the sequence than the last, it carries less information — and test accuracy falls substantially to **75.00%** (vs. 95.67%). This confirms why the last token is the natural choice for a causal LLM.

## Chapter 7

### Exercise 7.1 — Changing prompt styles (Phi-3 format)

Swap the Alpaca-style prompt for the more compact Phi-3 format, which looks like:

```text
<|user|>
Identify the correct spelling of the following word: 'Occasion'
<|assistant|>
The correct spelling is 'Occasion'.
```

Modify `format_input` to emit the Phi-3 markup:

```python title="Solution 7.1 — Phi-3 format_input"
def format_input(entry):
    instruction_text = (
        f"<|user|>\n{entry['instruction']}"
    )
    input_text = f"\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text
```

Then update how the generated response is extracted, stripping the `<|assistant|>` marker:

```python title="Solution 7.1 — extract the Phi-3 response" collapsible
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = (
        generated_text[len(input_text):]
        .replace("<|assistant|>:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text
```

Because the Phi-3 template produces shorter inputs, fine-tuning runs about **17% faster**, and the evaluation score stays near 50 — the same ballpark as the Alpaca-style prompts.

### Exercise 7.2 — Masking instruction tokens

By default the loss is computed over the whole sequence. To mask the **instruction** tokens (so loss is computed only on the response), record each example's instruction length in the dataset, then set those target positions to the ignore index in the collate function.

First, have `InstructionDataset` return the instruction length alongside the encoded text:

```python title="Solution 7.2 — InstructionDataset tracks instruction length" collapsible
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.instruction_lengths = []     # separate list for instruction lengths
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            self.encoded_texts.append(tokenizer.encode(full_text))

            instruction_length = len(
                tokenizer.encode(instruction_plus_input))
            self.instruction_lengths.append(instruction_length)

    def __getitem__(self, index):    # returns both length and text
        return self.instruction_lengths[index], self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
```

Then update `custom_collate_fn` — each batch item is now a `(instruction_length, item)` tuple — and set the instruction positions in the targets to `-100`:

```python title="Solution 7.2 — collate fn masks instruction tokens" collapsible
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for instruction_length, item in batch)
    inputs_lst, targets_lst = [], []

    for instruction_length, item in batch:     # batch is now a tuple
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        inputs  = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        targets[:instruction_length-1] = -100  # mask all instruction tokens

        if allowed_max_length is not None:
            inputs  = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor  = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
```

Interestingly, masking the instructions makes the model perform **slightly worse** here (about 4 points on the Ollama Llama 3 evaluation) — consistent with the *"Instruction Tuning with Loss Over Instructions"* paper ([arXiv 2405.14394](https://arxiv.org/abs/2405.14394)), which finds that *not* masking can help.

### Exercise 7.3 — Using the original Stanford Alpaca dataset

To fine-tune on the full Stanford Alpaca dataset, change only the data URL:

```python title="Solution 7.3 — point at the Alpaca dataset"
url = ("https://raw.githubusercontent.com/tatsu-lab/"
       "stanford_alpaca/main/alpaca_data.json")
```

This dataset has **52,000** entries (about 50× more than the chapter's, and longer), so training on a GPU is strongly recommended. If you hit out-of-memory errors, lower the batch size from 8 to 4, 2, or 1, and consider reducing `allowed_max_length` from 1024 to 512 or 256.

### Exercise 7.4 — Instruction fine-tuning with LoRA

Combine instruction fine-tuning with LoRA by importing the LoRA utilities from appendix E, freezing the base weights, and replacing the linear layers with LoRA-adapted ones:

```python title="Solution 7.4 — apply LoRA to the model"
from appendix_E import LoRALayer, LinearWithLoRA, replace_linear_with_lora

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters before: {total_params:,}")

for param in model.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters after: {total_params:,}")

replace_linear_with_lora(model, rank=16, alpha=16)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable LoRA parameters: {total_params:,}")

model.to(device)
```

On an Nvidia L4 GPU, LoRA fine-tuning runs in **1.30 min** vs. **1.80 min** for full fine-tuning — about **28% faster** — while the evaluation score stays around 50, in the same ballpark as the original.

::: callout note "Appendix A exercises"
Appendix C also includes solutions to the PyTorch exercises (A.1–A.4) — counting parameters in a small network and timing CPU vs. GPU matrix multiplication. Those are covered alongside the PyTorch primer in [Appendix A](appA-pytorch.html); the headline results are 752 trainable parameters for the example network and roughly a 4× speedup on a V100 GPU.
:::

## Key takeaways

::: takeaways
- **Most exercises are tiny, high-leverage edits.** Resizing GPT (4.2), swapping decoding to deterministic (5.3), or loading a bigger model (5.6) are one- or two-line changes — proof that the architecture generalizes cleanly across scales.
- **More fine-tuning isn't always better.** Padding to the full context (6.1), using the first token (6.3), and masking instructions (7.2) all *hurt* accuracy, while unfreezing more layers (6.2) helps slightly — measure, don't assume.
- **Watch the conventions.** Exercise 3.1's transpose is a reminder that `nn.Linear` stores weights as `[d_out, d_in]` and computes `x @ W.T`.
- **Efficiency wins compound.** The Phi-3 format (7.1, ~17% faster) and LoRA (7.4, ~28% faster) reach comparable quality with less compute — exactly the kind of trade-off that matters at scale.
- **The book's repo has it all.** Every solution here is runnable end to end at [github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).
:::
