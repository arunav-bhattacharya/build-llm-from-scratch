Fine-tuning a large model the naive way means updating *every* weight — all 124 million of them for the smallest GPT-2, and billions for larger models. That's expensive in compute, memory, and storage. **Low-Rank Adaptation (LoRA)** is one of the most widely used **parameter-efficient fine-tuning** techniques because it sidesteps that cost: instead of editing the giant weight matrices, it freezes them and learns a tiny, low-rank *correction* on the side.

This appendix builds LoRA from scratch on the spam-classification example from chapter 6 (the same idea applies to the instruction fine-tuning of chapter 7). We'll see the math behind the low-rank trick, then implement a `LoRALayer`, wrap PyTorch `Linear` layers with it, freeze the original model, and fine-tune only the small LoRA matrices — cutting trainable parameters by nearly 50× while matching the accuracy of full fine-tuning.

::: objectives "What you'll learn"
- What "low-rank" means and how LoRA approximates a weight update $\Delta W$ as the product $AB$
- Why keeping the LoRA matrices *separate* from the frozen weights is so practical
- How to implement a `LoRALayer` and a `LinearWithLoRA` wrapper in PyTorch
- How to **freeze** the pretrained weights and swap in LoRA across every `Linear` layer
- How rank and alpha trade off adaptability against the number of trainable parameters
:::

## E.1 Introduction to LoRA

LoRA adapts a pretrained model to a specific (often smaller) dataset by adjusting only a small subset of its weights. The **"low-rank"** part refers to restricting the model's adjustments to a smaller-dimensional subspace of the full weight space — which turns out to capture the most influential *directions* of change during training. The payoff: efficient fine-tuning of large models on task-specific data, at a fraction of the usual compute and storage.

Consider one layer's weight matrix $W$. (LoRA can be applied to *all* linear layers in an LLM; we focus on one for clarity.) During normal backpropagation we learn an update matrix $\Delta W$ — how much to nudge $W$ to reduce the loss. In **regular** fine-tuning the new weights are simply:

$$W_{\text{updated}} = W + \Delta W.$$

LoRA, proposed by [Hu et al. (2021)](https://arxiv.org/abs/2106.09685), replaces the full $\Delta W$ with a cheap low-rank approximation:

$$\Delta W \approx A B,$$

where $A$ and $B$ are two matrices **much smaller** than $W$. If $W$ is $d \times k$, then $A$ is $d \times r$ and $B$ is $r \times k$, with the **inner dimension $r$** (the "rank") chosen to be tiny. The LoRA weight update becomes:

$$W_{\text{updated}} = W + A B.$$

::: callout analogy "Editing a huge document with sticky notes"
Imagine a 500-page contract you need to tailor for a new client. You *could* retype the entire thing (full fine-tuning) — slow, and now you're storing a whole second copy. Or you could leave the original untouched and clip on a few **sticky notes** with the changes (LoRA). The notes are small, easy to swap, and you keep one master document with different note-sets for different clients. LoRA's $A$ and $B$ are those sticky notes.
:::

::: callout math "Why two skinny matrices? The low-rank intuition"
A full update $\Delta W$ of shape $d \times k$ has $d \cdot k$ free numbers. Factoring it as $A B$ with inner rank $r$ uses only $d \cdot r + r \cdot k$ numbers. When $r \ll d, k$ this is a massive reduction. For a $768 \times 768$ matrix ($\approx 590{,}000$ params) at $r=16$, LoRA needs just $768\cdot16 + 16\cdot768 \approx 24{,}600$ — about **4%**. The bet is that the *useful* fine-tuning update lives in a low-rank subspace, so a skinny $A$ and $B$ can approximate it well.
:::

::: diagram appE-lora-decomposition "Regular fine-tuning updates the full d×k weight matrix W with ΔW. LoRA freezes W and approximates ΔW as the product of two skinny matrices A (d×r) and B (r×k) — only A and B are trainable, and r is small."
:::

Thanks to the **distributive law** of matrix multiplication, we don't have to merge the update into $W$. For an input $x$, regular fine-tuning computes $x(W + \Delta W) = xW + x\,\Delta W$, and LoRA likewise computes:

$$x\,(W + AB) = xW + x\,(AB).$$

That separation is the practical superpower: the pretrained weights $W$ stay **unchanged**, and the LoRA matrices are applied dynamically on top. You can keep one frozen base model and store a small $A,B$ pair *per task or per customer*, instead of a full model copy for each — slashing storage and improving scalability.

## E.2 Preparing the dataset

The data prep simply repeats chapter 6's spam pipeline. We download the SMS Spam Collection, balance and split it, and save CSVs:

```python title="Listing E.1 — Downloading and preparing the dataset" collapsible
from pathlib import Path
import pandas as pd
from ch06 import (
    download_and_unzip_spam_data,
    create_balanced_dataset,
    random_split
)

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)
balanced_df = create_balanced_dataset(df)
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)
```

Then we wrap them in `SpamDataset` objects and build data loaders (batch size 8):

```python title="Listings E.2 & E.3 — Datasets and data loaders" collapsible
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from chapter06 import SpamDataset

tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = SpamDataset("train.csv", max_length=None, tokenizer=tokenizer)
val_dataset = SpamDataset(
    "validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
test_dataset = SpamDataset(
    "test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size,
    shuffle=True, num_workers=num_workers, drop_last=True)
val_loader = DataLoader(
    dataset=val_dataset, batch_size=batch_size,
    num_workers=num_workers, drop_last=False)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size,
    num_workers=num_workers, drop_last=False)
```

A quick check confirms each batch holds 8 examples of 120 tokens, with 130 / 19 / 38 batches across the train / validation / test splits.

## E.3 Initializing the model

Again repeating chapter 6, we download GPT-2 small (124M) and load its weights into `GPTModel`:

```python title="Listing E.4 — Loading a pretrained GPT model" collapsible
from gpt_download import download_and_load_gpt2
from chapter04 import GPTModel
from chapter05 import load_weights_into_gpt

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}
model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
```

We then replace the output head with a 2-class classification layer (spam vs. ham), exactly as in chapter 6:

```python title="Swap in a classification head"
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

Before fine-tuning, the not-yet-trained classifier scores about 50% accuracy — coin-flip territory — confirming it can't yet tell spam from ham (Training 46.25%, Validation 45.00%, Test 48.75%).

## E.4 Parameter-efficient fine-tuning with LoRA

Now the heart of it. We start with a `LoRALayer` that holds the two matrices $A$ and $B$, the rank $r$, and the scaling factor `alpha`. It takes an input and returns the low-rank update applied to it:

```python title="Listing E.5 — Implementing a LoRA layer"
import math

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        # Same initialization PyTorch uses for Linear layers
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
```

Two details drive everything:

- **`rank`** sets the inner dimension of $A$ and $B$, and therefore the number of extra parameters LoRA introduces. It balances the model's *adaptability* against its *efficiency*.
- **`alpha`** is a scaling factor for the low-rank output — it dictates how strongly the adaptation influences the original layer's output, regulating LoRA's effect.

::: callout math "The LoRA scaling factor"
The adapted output is scaled by alpha: $\Delta y = \alpha \cdot x A B$. In Hu et al.'s original formulation the scale is written $\frac{\alpha}{r}$ (dividing by the rank), so that changing $r$ doesn't require re-tuning the learning rate. The book uses a plain $\alpha$ multiplier; either way, alpha is the knob controlling how much the low-rank correction nudges the frozen output. A common heuristic is to set alpha to half, equal to, or double the rank.
:::

::: callout key "Why initialize B to zeros?"
$A$ gets the usual Kaiming initialization, but **$B$ is all zeros**. So at the very start $AB$ is a zero matrix, and `linear(x) + lora(x)` equals `linear(x)` exactly — adding zero changes nothing. The model therefore begins fine-tuning as an *identical copy* of the pretrained model, and the LoRA path only starts to matter once $B$ moves away from zero during training. (This is why the pre-training accuracies below are unchanged.)
:::

In practice we want LoRA to **substitute** existing `Linear` layers, so the update applies directly alongside the pretrained weights:

```python title="Listing E.6 — A Linear layer wrapped with LoRA"
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear   # the original (frozen) Linear layer
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

The `forward` adds the original linear output to the LoRA output — the additive $xW + x(AB)$ structure from section E.1.

::: diagram appE-lora-insert "A LinearWithLoRA wraps a frozen Linear layer: the input flows through both the original weights W (frozen) and the small LoRA branch A·B, and their outputs are summed. Only A and B carry gradients."
:::

To apply this across the whole model, a small recursive helper swaps every `Linear` for a `LinearWithLoRA`:

```python title="Replace every Linear with LinearWithLoRA"
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):       # replace Linear
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:                                          # recurse into children
            replace_linear_with_lora(module, rank, alpha)
```

### Freezing the original weights

Before swapping in LoRA, we **freeze** every existing parameter so gradients won't flow into the pretrained weights:

```python title="Freeze all pretrained parameters"
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters before: {total_params:,}")

for param in model.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters after: {total_params:,}")
# Total trainable parameters before: 124,441,346
# Total trainable parameters after: 0
```

All 124M parameters are now frozen — zero are trainable. Then we inject LoRA, which adds back *only* the small $A$/$B$ matrices as trainable:

```python title="Listing E.7 (setup) — Inject LoRA and count trainable params"
replace_linear_with_lora(model, rank=16, alpha=16)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable LoRA parameters: {total_params:,}")
# Total trainable LoRA parameters: 2,666,528
```

That's a **~50× reduction** in trainable parameters (2.7M instead of 124M). A rank and alpha of 16 are solid defaults; raising the rank adds trainable capacity, and alpha is typically set to half, equal to, or double the rank.

::: diagram appE-param-savings "Trainable parameters: full fine-tuning updates all ~124M weights, while LoRA (rank 16) trains only ~2.7M — roughly a 50× reduction, shown as a tall bar beside a tiny one."
:::

Printing the model confirms each `Linear` inside the attention blocks, feed-forward modules, and output head is now a `LinearWithLoRA` wrapping the original frozen `Linear` plus a `LoRALayer`:

```text title="Model architecture after LoRA injection (abridged)" collapsible
GPTModel(
  ...
  (trf_blocks): Sequential(
    (11): TransformerBlock(
      (att): MultiHeadAttention(
        (W_query): LinearWithLoRA(
          (linear): Linear(in_features=768, out_features=768, bias=True)
          (lora): LoRALayer()
        )
        (W_key):   LinearWithLoRA( (linear): Linear(...=768) (lora): LoRALayer() )
        (W_value): LinearWithLoRA( (linear): Linear(...=768) (lora): LoRALayer() )
        (out_proj):LinearWithLoRA( (linear): Linear(...=768) (lora): LoRALayer() )
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (ff): FeedForward(
        (layers): Sequential(
          (0): LinearWithLoRA( (linear): Linear(768, 3072) (lora): LoRALayer() )
          (1): GELU()
          (2): LinearWithLoRA( (linear): Linear(3072, 768) (lora): LoRALayer() )
        )
      )
      ...
    )
  )
  (final_norm): LayerNorm()
  (out_head): LinearWithLoRA(
    (linear): Linear(in_features=768, out_features=2, bias=True)
    (lora): LoRALayer()
  )
)
```

Because $B$ starts at zero, the initial accuracy is *identical* to chapter 6's untrained classifier (46.25% / 45.00% / 48.75%) — adding $AB = 0$ changes nothing yet.

### Fine-tuning with LoRA

Now the exciting part: we reuse chapter 6's `train_classifier_simple` unchanged. The optimizer sees only the trainable LoRA parameters (the frozen ones have `requires_grad=False`):

```python title="Listing E.7 — Fine-tuning the model with LoRA layers"
import time
from chapter06 import train_classifier_simple

start_time = time.time()
torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
        tokenizer=tokenizer
    )

end_time = time.time()
print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")
```

```text title="Training output (abridged)"
Ep 1 (Step 000000): Train loss 3.820, Val loss 3.462
Ep 1 (Step 000100): Train loss 0.111, Val loss 0.229
Training accuracy: 97.50% | Validation accuracy: 95.00%
...
Ep 5 (Step 000600): Train loss 0.000, Val loss 0.056
Training accuracy: 100.00% | Validation accuracy: 97.50%
Training completed in 12.10 minutes.
```

::: callout note "Why LoRA can be slower on a small model"
Here LoRA training took *longer* than full fine-tuning, because the extra LoRA branch adds computation to every forward pass while the model is small enough that backprop was already cheap. For **large** models, where backpropagating through all weights dominates the cost, LoRA is typically *faster* — most parameters are frozen, so there are far fewer gradients to compute.
:::

Evaluating on the full splits gives excellent results — Training 100%, Validation 96.64%, Test ~98% — essentially matching full fine-tuning. The slight gap between train and test points to mild overfitting, but it's remarkable considering we trained only **2.7 million LoRA weights** instead of the full **124 million**.

## Key takeaways

::: takeaways
- **LoRA** freezes the pretrained weights $W$ and learns a low-rank approximation of the update: $\Delta W \approx AB$, so $W_{\text{updated}} = W + AB$ with inner rank $r \ll d,k$.
- By the distributive law, the layer computes $xW + x(AB)$ — the frozen path plus a small additive LoRA branch — so $W$ never changes and the $A,B$ pair can be stored/swapped per task.
- **`LinearWithLoRA`** wraps a frozen `Linear` and adds a `LoRALayer`; `replace_linear_with_lora` recursively swaps every `Linear` in the model.
- Initializing **$B$ to zeros** makes $AB=0$ at the start, so the model begins as an exact copy of the pretrained one and only diverges as training updates $B$.
- After freezing all 124M params and injecting LoRA (rank 16), only **~2.7M** params are trainable — a ~50× reduction — yet accuracy matches full fine-tuning.
- **Rank** sets capacity/parameter count; **alpha** scales the adaptation's influence (commonly ½×, 1×, or 2× the rank).
:::

## Additional references

::: refs
- [Appendix E code (appendix-E.ipynb)](https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-E) — GitHub · the complete, runnable LoRA notebook for this appendix.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Paper · Hu et al. (2021), the original LoRA method this appendix implements.
- [Parameter-Efficient LLM Finetuning With LoRA](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html) — Blog · the author's own deep dive into how and why LoRA works.
- [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — Blog · Raschka's lessons from hundreds of LoRA experiments (rank, alpha, where to apply it).
- [Hugging Face PEFT](https://github.com/huggingface/peft) — Library · production implementations of LoRA, QLoRA, and other parameter-efficient methods.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Paper · Dettmers et al., combining 4-bit quantization with LoRA to fine-tune huge models on a single GPU.
:::

## Test your knowledge

```flashcards
Q: What does the "low-rank" in LoRA refer to?
A: Restricting the weight update to a small-dimensional subspace — approximating the full update ΔW (d×k) as the product of two skinny matrices A (d×r) and B (r×k) with a small inner rank r.
---
Q: How does LoRA reformulate the fine-tuning weight update?
A: Instead of W_updated = W + ΔW, it uses W_updated = W + AB, where AB approximates ΔW and only A and B are trained.
---
Q: Why does keeping the LoRA matrices separate from W matter in practice?
A: W stays unchanged, so you keep one frozen base model and store a small A,B pair per task/customer instead of a full model copy each — saving storage and improving scalability.
---
Q: In LoRALayer, why is matrix B initialized to zeros?
A: So AB = 0 at the start; linear(x) + lora(x) then equals linear(x), meaning the model begins identical to the pretrained one and only changes as B trains.
---
Q: What do the rank and alpha hyperparameters control?
A: Rank sets the inner dimension of A and B (hence the number of trainable parameters / adaptability); alpha is a scaling factor for how strongly the LoRA output affects the layer's output.
---
Q: What does replace_linear_with_lora do?
A: It recursively walks the model and swaps every torch.nn.Linear layer for a LinearWithLoRA (frozen Linear + a LoRALayer).
---
Q: Roughly how many trainable parameters remain after applying LoRA (rank 16) to GPT-2 small, and how does that compare to full fine-tuning?
A: About 2.7 million, versus 124 million for full fine-tuning — roughly a 50× reduction.
---
Q: Why might LoRA train slower on a small model but faster on a large one?
A: The LoRA branch adds forward-pass computation; on a small model backprop was already cheap, so this dominates. On large models, freezing most weights means far fewer gradients, so LoRA is faster overall.
```

```quiz
1. LoRA approximates the weight update ΔW as:
   - ( ) W + ΔW
   - (x) the product AB of two much smaller matrices
   - ( ) a diagonal matrix
   - ( ) the inverse of W
   > LoRA learns ΔW ≈ AB, where A is d×r, B is r×k, and r is a small inner rank.

2. With W of shape d×k and inner rank r, the LoRA matrices A and B have shapes:
   - ( ) d×k and k×d
   - ( ) r×r and r×r
   - (x) d×r and r×k
   - ( ) d×d and k×k
   > A maps the d-dim input down to r, B maps r back up to k, so their product AB matches W's d×k shape.

3. Initializing B to zeros at the start of training ensures that:
   - (x) AB = 0, so the LoRA-wrapped model initially behaves exactly like the pretrained model
   - ( ) the gradients explode
   - ( ) A is never updated
   - ( ) the learning rate must be zero
   > Adding a zero matrix changes nothing, so fine-tuning starts from the unmodified pretrained behavior.

4. After freezing GPT-2 small and applying LoRA with rank 16, the number of trainable parameters is approximately:
   - ( ) 124 million
   - ( ) zero
   - (x) 2.7 million
   - ( ) 50 million
   > Freezing leaves 0 trainable; injecting LoRA adds ~2.7M trainable params, about a 50× reduction from 124M.

5. Which statement about LoRA in production is true?
   - ( ) It overwrites the pretrained weights, so you must keep one full copy per task.
   - (x) Because A and B stay separate from W, you can store a small adapter per task and share one frozen base model.
   - ( ) It requires retraining the entire model from scratch.
   - ( ) It only works on the output layer.
   > Keeping the LoRA matrices separate is precisely what lets you swap small task-specific adapters over a single shared frozen model.
```

```assignment "Sweep the LoRA rank" level=intermediate
Using the spam-classification setup from this appendix, investigate how the rank hyperparameter trades parameters for performance. After freezing the base model, call `replace_linear_with_lora(model, rank=r, alpha=r)` for at least three ranks (e.g., r = 4, 16, 64), each time printing the total trainable LoRA parameter count. Fine-tune each variant with `train_classifier_simple` and record the final test accuracy. Summarize the trend: how do trainable parameters and test accuracy change as rank grows?

Hint: re-load a fresh pretrained model and re-freeze it before each run, otherwise LoRA layers stack on top of each other.
Hint: count params with `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
Hint: keep alpha tied to rank (a common heuristic) so only one variable changes per run.
```
