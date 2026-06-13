We've coded a GPT architecture, pretrained it, and loaded OpenAI's GPT-2 weights into it. Now we finally **reap the reward**: we take that general-purpose model and **fine-tune** it into something useful for one specific job — telling **spam** text messages apart from legitimate ones.

This is the first of two fine-tuning chapters. Here we do **classification fine-tuning**: we bolt a tiny new output layer onto the pretrained model, freeze most of the network, and train only the top of the stack on labeled spam/not-spam examples. In about five epochs (a few minutes on a laptop) the model climbs from coin-flip accuracy to ~96% on held-out messages.

::: objectives "What you'll learn"
- The difference between **classification** and **instruction** fine-tuning, and when to use each
- Preparing the **SMS spam** dataset: download, balance the classes, and split train/validation/test
- Building a PyTorch `Dataset` and `DataLoader` that **pad** variable-length messages to a uniform length
- Loading pretrained GPT-2 weights as the starting point
- **Replacing the output head** with a 2-class linear layer and **freezing** the lower layers
- Why we read only the **last token's** output, and how to compute classification loss and accuracy
- The supervised fine-tuning loop, and using the finished model to classify brand-new messages
:::

## Different categories of fine-tuning

There are two common ways to fine-tune a language model. **Instruction fine-tuning** trains the model on many tasks phrased as natural-language prompts ("Translate into German: …", "Answer with yes or no: …") so it gets better at following instructions in general — that's Chapter 7. **Classification fine-tuning** trains the model to sort inputs into a fixed set of **class labels**, like "spam" and "not spam."

If you've done any classical machine learning, classification fine-tuning will feel familiar. The crucial difference from instruction tuning: a classification-tuned model is **restricted to the classes it saw during training**. Our spam model will only ever answer "spam" or "not spam" — it can't summarize the message, translate it, or do anything else.

::: diagram ch06-finetune-types "A single pretrained foundation model can be fine-tuned two ways: into a narrow classifier that emits class labels, or into a general assistant that follows free-form instructions."
:::

::: callout analogy "Retraining a versatile graduate for one job"
A pretrained LLM is like a **broadly educated graduate** — it knows a bit about everything but isn't an expert at any single task. **Classification fine-tuning** is sending that graduate to a short, intensive course for **one specific role** (a spam analyst). They keep all their general language ability, but now they're sharply tuned for this one decision. **Instruction fine-tuning** is the opposite philosophy: coaching them to handle *any* request a customer might phrase. The specialist is cheaper to train and very accurate at its niche; the generalist is more flexible but needs far more data and compute.
:::

::: callout note "Choosing the right approach"
**Instruction fine-tuning** is best when the model must handle a *variety* of tasks from complex user prompts — it maximizes flexibility but demands larger datasets and more compute. **Classification fine-tuning** is ideal when you need precise sorting into predefined buckets (sentiment analysis, spam detection, topic tagging). It needs less data and compute, but the model is confined to the classes it was trained on. Classification tasks go far beyond email filtering: identifying plant species from photos, routing news into "sports / politics / tech," or flagging tumors as benign vs. malignant.
:::

## Preparing the dataset

We'll fine-tune the same GPT model we built and pretrained, using a dataset of real **SMS text messages** labeled as spam or "ham" (the standard term for non-spam). The first step is to download and unzip it.

```python title="Listing 6.1 — Downloading and unzipping the dataset" collapsible
import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(
        url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download "
              "and extraction.")
        return
    with urllib.request.urlopen(url) as response:        # downloads the file
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:      # unzips the file
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)        # adds a .tsv extension
    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
```

The dataset lands as a tab-separated file. We load it into a pandas `DataFrame` and inspect the class distribution:

```python title="Loading the dataset and checking the class balance"
import pandas as pd
df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)
print(df["Label"].value_counts())
# ham     4825
# spam     747
```

There's a big **class imbalance** — "ham" outnumbers "spam" almost 7-to-1. For a small, fast example, we **undersample** the majority class: keep all 747 spam messages and randomly draw 747 ham messages, giving a balanced 50/50 set.

```python title="Listing 6.2 — Creating a balanced dataset"
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]        # count "spam"
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123                       # match spam count
    )
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]            # combine the two
    ])
    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
# ham     747
# spam    747
```

::: callout note "Why balance the classes?"
With a 7-to-1 imbalance, a lazy model could score ~87% accuracy by always guessing "ham" — without learning anything about spam. Balancing the classes (here by undersampling) removes that shortcut so accuracy actually reflects skill. Undersampling is the simplest fix; the book's appendix B points to other techniques for handling imbalance.
:::

Next we map the string labels to integers — `ham → 0`, `spam → 1` — exactly the kind of text-to-ID conversion we did for tokens, except now there are only two IDs instead of 50,000+:

```python title="Encoding labels as integers"
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
```

Finally we shuffle and split into **70% train, 10% validation, 20% test** — the classic trio for fitting, tuning, and honestly evaluating a model.

```python title="Listing 6.3 — Splitting the dataset"
def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # shuffle
    train_end = int(len(df) * train_frac)                            # split indices
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]                  # test is the remainder (0.2)
    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)
```

## Creating data loaders

When we worked with raw text for pretraining, every training chunk was the **same length** (a fixed sliding window). Spam messages aren't — they range from a few words to over a hundred. To batch tensors together they must share a shape, so we have two options:

- **Truncate** every message to the length of the *shortest* one — cheap, but throws away information.
- **Pad** every message up to the length of the *longest* one — preserves all content.

We choose **padding**, using the `<|endoftext|>` token (ID **50256**) as the pad token — the same special token GPT-2 already knows.

::: diagram ch06-spam-pipeline "At inference, a raw message is tokenized and padded to a fixed length, run through the fine-tuned model, and the last token's logits decide spam vs. not spam. Training uses the same tokenize-and-pad front end."
:::

::: callout analogy "Padding = same-size envelopes for the mail sorter"
A batching machine can only grab a stack of **identically sized envelopes**. Real messages come in all lengths, so we slip each short one into a standard-size envelope and fill the empty space with blank filler (the padding token). The content is untouched; the envelopes are just uniform so the machine can process a whole batch at once.
:::

The `SpamDataset` class pretokenizes every message, finds the longest one, and pads (or truncates) all the rest to that length:

```python title="Listing 6.4 — A PyTorch Dataset for the spam data" collapsible
import torch
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None,
                 pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [                           # pretokenize texts
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [                       # truncate if too long
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
        self.encoded_texts = [                           # pad to longest
            encoded_text + [pad_token_id] *
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
```

We build the training dataset first — its longest message is **120 tokens**, comfortably under GPT-2's 1,024-token limit. Then we build the validation and test sets, **reusing the training set's `max_length`** so every split shares one consistent length. (Any longer message in val/test gets truncated by the `encoded_text[:self.max_length]` line above.)

```python title="Building the three datasets"
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = SpamDataset(
    csv_file="train.csv", max_length=None, tokenizer=tokenizer
)
print(train_dataset.max_length)    # 120

val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length, tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length, tokenizer=tokenizer
)
```

Now we wrap each dataset in a `DataLoader`. The key difference from pretraining: each batch's **targets are class labels** (0 or 1), not next tokens. With `batch_size=8`, every batch is 8 messages × 120 tokens plus 8 labels.

```python title="Listing 6.5 — Creating PyTorch data loaders" collapsible
from torch.utils.data import DataLoader

num_workers = 0       # ensures compatibility with most computers
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size,
    shuffle=True, num_workers=num_workers, drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset, batch_size=batch_size,
    num_workers=num_workers, drop_last=False,
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size,
    num_workers=num_workers, drop_last=False,
)
```

A quick sanity check confirms the shapes — 8 examples of 120 tokens, plus 8 labels — and the dataset sizes:

```python title="Verifying batch shapes and counts"
for input_batch, target_batch in train_loader:
    pass
print("Input batch dimensions:", input_batch.shape)   # torch.Size([8, 120])
print("Label batch dimensions", target_batch.shape)   # torch.Size([8])

print(f"{len(train_loader)} training batches")         # 130
print(f"{len(val_loader)} validation batches")         # 19
print(f"{len(test_loader)} test batches")              # 38
```

## Initializing a model with pretrained weights

The model we fine-tune starts as a **pretrained GPT-2 small (124M)**, configured exactly as in pretraining. Note `qkv_bias=True` and `drop_rate=0.0` — these match OpenAI's released weights.

```python title="Listing 6.6 — Loading a pretrained GPT-2 model" collapsible
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,      # vocabulary size
    "context_length": 1024,   # context length
    "drop_rate": 0.0,         # dropout rate
    "qkv_bias": True          # query-key-value bias
}
model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

from gpt_download import download_and_load_gpt2
from chapter05 import GPTModel, load_weights_into_gpt

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
)
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
```

Before changing anything, it's worth confirming the model still works — and checking whether the *raw* pretrained model can already classify spam if we just ask it. It can't:

```python title="The pretrained model can't follow a spam-classification instruction"
from chapter05 import text_to_token_ids, token_ids_to_text
from chapter04 import generate_text_simple

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
# ...just echoes the prompt back — it doesn't answer "yes" or "no"
```

The model rambles instead of answering. That's expected: it has only been **pretrained** (next-word prediction) and never taught to follow instructions. Time to fine-tune it for classification.

## Adding a classification head

Here's the central trick of this chapter. The pretrained GPT ends in an output layer mapping 768 hidden dimensions to **50,257** logits — one per vocabulary token. For binary classification we don't need 50,257 outputs; we need **two** (one per class). So we **replace** that output head with a small `nn.Linear(768, 2)`.

::: diagram ch06-classification-head "We swap GPT-2's 50,257-way vocabulary head for a 2-class linear head, freeze the embeddings and the lower 11 transformer blocks, and make only the last block, the final LayerNorm, and the new head trainable."
:::

We don't retrain the whole network. We **freeze** every existing parameter, then unfreeze only the top: the **last transformer block**, the **final LayerNorm**, and the **new head**.

```python title="Listing 6.7 — Freezing the model and adding a classification layer"
# 1) Freeze every existing parameter
for param in model.parameters():
    param.requires_grad = False

# 2) Replace the 50,257-way head with a 2-class head
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],   # 768 for gpt2-small
    out_features=num_classes              # 2: "not spam" / "spam"
)

# 3) Unfreeze the last transformer block and the final LayerNorm
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
```

A freshly created `nn.Linear` has `requires_grad=True` by default, so the new head trains automatically. Training *only* that head is technically enough — but the author found that also unfreezing the last block and final norm noticeably improves accuracy.

::: callout analogy "Keep the foundation, only re-tile the roof"
The pretrained model is a finished house. Its **lower floors** — the embeddings and early transformer blocks — capture general language structure that's useful for almost any task, so we **keep them as-is** (frozen). We only **re-tile the roof**: the top block, final norm, and a brand-new output layer shaped for our two classes. Far less work than rebuilding the whole house, and the solid foundation does most of the heavy lifting.
:::

::: callout key "Why fine-tune only the top layers?"
In language models, **lower layers** learn broad, reusable features (basic syntax and semantics), while **upper layers** specialize toward the task at hand. Because our pretrained model already has excellent low-level language understanding, we can freeze those layers and adapt only the top — this is **transfer learning**. The payoff is twofold: far **fewer parameters to update** (faster, less memory) and **less overfitting** on our small dataset.
:::

Feeding in a 4-token example now yields a `[1, 4, 2]` output instead of `[1, 4, 50257]` — one 2-value row per input token:

```python title="The output now has 2 columns, not 50,257"
inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)             # shape [1, 4]
with torch.no_grad():
    outputs = model(inputs)
print("Outputs dimensions:", outputs.shape)            # torch.Size([1, 4, 2])
print("Last output token:", outputs[:, -1, :])         # tensor([[-3.5983, 3.9902]])
```

Notice we grabbed `outputs[:, -1, :]` — only the **last** token's row. Why ignore the other three?

### Why we read only the last token

::: diagram ch06-last-token "Every input token produces an output row, but only the last token's two logits feed the classifier. Thanks to causal attention, the last token has attended to every earlier token, so its representation already summarizes the whole message."
:::

Recall **causal attention** from Chapter 3: each token can attend only to itself and the tokens *before* it. That means the **last token is the only position that has 'seen' the entire sequence** — its representation has accumulated information from every preceding token. So it's the natural place to read off a whole-message decision.

::: callout analogy "The reader who has finished the whole message"
Because of causal masking, token 1 has read only word 1, token 2 has read words 1–2, and so on — but the **final** token has read the *entire* message. Asking the last token "spam or not?" is like asking the one person in the room who actually finished reading the text, rather than someone who only saw the opening words.
:::

## Calculating the classification loss and accuracy

To turn the last token's two logits into a label, we do exactly what we did for next-token prediction: take the **argmax**. (Softmax is optional here — the largest logit is already the most probable class, and argmax is unaffected by the monotonic softmax.)

```python title="From logits to a class label"
logits = outputs[:, -1, :]              # [[-3.5983, 3.9902]]
label = torch.argmax(logits, dim=-1)
print("Class label:", label.item())     # 1  → "spam"
```

Applying this across a whole loader gives **classification accuracy** — the fraction of correct predictions:

```python title="Listing 6.8 — Calculating classification accuracy" collapsible
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]     # last-token logits
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break
    return correct_predictions / num_examples
```

Measuring the **untrained** classifier (random new head) confirms it's no better than guessing — around 50%, as expected for a balanced two-class problem:

```python title="Baseline accuracy before fine-tuning"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy   = calc_accuracy_loader(val_loader,   model, device, num_batches=10)
test_accuracy  = calc_accuracy_loader(test_loader,  model, device, num_batches=10)
print(f"Training accuracy:   {train_accuracy*100:.2f}%")   # ~46%
print(f"Validation accuracy: {val_accuracy*100:.2f}%")     # ~45%
print(f"Test accuracy:       {test_accuracy*100:.2f}%")    # ~49%
```

Accuracy itself isn't differentiable, so we can't optimize it directly. As in pretraining, we minimize **cross-entropy loss** as a stand-in. The loss functions are nearly identical to the pretraining versions — the *only* change is slicing `[:, -1, :]` to use the last token's logits.

::: callout math "Cross-entropy on the last token"
For one example with true class $y \in \{0, 1\}$, the model outputs logits $z = (z_0, z_1)$ from the last token. Softmax turns them into probabilities $p_k = \dfrac{e^{z_k}}{e^{z_0}+e^{z_1}}$, and the loss is $\mathcal{L} = -\log p_y$. Minimizing it pushes the probability of the **correct** class toward 1. Averaged over a batch, this is exactly `torch.nn.functional.cross_entropy(logits, target_batch)`.
:::

```python title="Listing 6.9 — Classification loss (last-token cross-entropy)" collapsible
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]      # last-token logits
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:                                       # don't exceed available batches
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
```

The initial losses are high (≈2.5) — consistent with random predictions. Fine-tuning will drive them down.

## Fine-tuning the model on supervised data

The training loop is **the same one we used for pretraining**, with just two changes: we track the number of **training examples seen** (instead of tokens), and we compute **classification accuracy** after each epoch (instead of generating sample text).

```python title="Listing 6.10 — Fine-tuning loop to classify spam" collapsible
def train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter):
    # initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):              # main training loop
        model.train()                            # set model to training mode
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                # reset gradients
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()                      # compute gradients
            optimizer.step()                     # update weights
            examples_seen += input_batch.shape[0]   # NEW: track examples
            global_step += 1

            if global_step % eval_freq == 0:     # optional evaluation step
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        # NEW: compute accuracy after each epoch
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen
```

The `evaluate_model` helper is **identical** to the pretraining one — it just averages the loss over a few batches of the training and validation sets with gradients turned off:

```python title="The evaluate_model helper (unchanged from pretraining)"
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
```

Now we kick off training with **AdamW**, a learning rate of `5e-5`, and **5 epochs**. This takes about 6 minutes on an M3 MacBook Air, or under 30 seconds on a V100/A100 GPU.

```python title="Running the fine-tuning"
import time
start_time = time.time()
torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5
    )

end_time = time.time()
print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")
```

The training log shows loss dropping sharply and accuracy climbing toward 100%:

```text title="Training output (abridged)"
Ep 1 (Step 000000): Train loss 2.153, Val loss 2.392
Ep 1 (Step 000100): Train loss 0.523, Val loss 0.557
Training accuracy: 70.00% | Validation accuracy: 72.50%
...
Ep 4 (Step 000500): Train loss 0.222, Val loss 0.137
Training accuracy: 100.00% | Validation accuracy: 97.50%
Ep 5 (Step 000600): Train loss 0.083, Val loss 0.074
Training accuracy: 100.00% | Validation accuracy: 97.50%
Training completed in 5.65 minutes.
```

::: callout tip "Reading the loss curves"
Both training and validation loss fall steeply in the first epoch, then flatten by epoch 5 — and they stay **close together**. A small, stable gap between train and validation loss is the signature of **healthy learning with little overfitting**. If validation loss had started *rising* while training loss kept falling, that would signal overfitting and a cue to stop earlier. Five epochs is a sensible default; adjust up or down based on the curves.
:::

Evaluating across the **full** datasets (no `eval_iter` cap this time) gives the final verdict:

```python title="Final accuracy over the entire datasets"
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy   = calc_accuracy_loader(val_loader,   model, device)
test_accuracy  = calc_accuracy_loader(test_loader,  model, device)
print(f"Training accuracy:   {train_accuracy*100:.2f}%")   # 97.21%
print(f"Validation accuracy: {val_accuracy*100:.2f}%")     # 97.32%
print(f"Test accuracy:       {test_accuracy*100:.2f}%")    # 95.67%
```

From ~50% (random) to **~96% on unseen test data** — and train/test scores are close, confirming the model generalizes rather than memorizes.

## Using the LLM as a spam classifier

The final payoff: a `classify_review` function that takes raw text and returns a verdict. It mirrors the `SpamDataset` preprocessing (tokenize, truncate to the model's context, pad), runs inference, reads the **last token's** logits, and maps the argmax to a label.

```python title="Listing 6.12 — Classifying new text with the fine-tuned model" collapsible
def classify_review(
        text, model, tokenizer, device, max_length=None,
        pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)                   # prepare inputs
    supported_context_length = model.pos_emb.weight.shape[1]
    input_ids = input_ids[:min(                          # truncate if too long
        max_length, supported_context_length
    )]
    input_ids += [pad_token_id] * (max_length - len(input_ids))  # pad
    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)                                       # add batch dimension

    with torch.no_grad():                                # inference, no grads
        logits = model(input_tensor)[:, -1, :]           # last-token logits
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"
```

Trying it on a textbook scam message and a friendly note — both correct:

```python title="Trying the classifier on two messages"
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)
print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))   # spam

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)
print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))   # not spam
```

Finally, save the fine-tuned weights so you can reload the classifier later without retraining:

```python title="Saving and reloading the fine-tuned model"
torch.save(model.state_dict(), "review_classifier.pth")

# later...
model_state_dict = torch.load("review_classifier.pth", map_location=device)
model.load_state_dict(model_state_dict)
```

## Key takeaways

::: takeaways
- Two main fine-tuning strategies: **classification** (sort into fixed labels) and **instruction** (follow free-form prompts). Classification needs less data/compute but is confined to its trained classes.
- **Classification fine-tuning** replaces the LLM's vocabulary output head with a small layer whose output count equals the **number of classes** (here, 2 nodes for spam / not spam).
- Variable-length messages are made uniform by **padding** to the longest sequence with the `<|endoftext|>` token (ID 50256), so they batch cleanly.
- We **freeze** the embeddings and lower transformer blocks and train only the **last block, final LayerNorm, and new head** — a form of transfer learning that's fast and resists overfitting.
- We read only the **last token's** output: because of causal attention, it's the one position that has attended to the entire message.
- Accuracy isn't differentiable, so we optimize **cross-entropy loss** on the last token as a proxy — the same loss used in pretraining.
- The fine-tuning loop reuses the pretraining loop almost verbatim; in ~5 epochs the spam classifier reaches **~96% test accuracy**.
:::

## Additional references

::: refs
- [Chapter 6 code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch06) — GitHub · the complete notebooks and scripts for this chapter.
- [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — Dataset · the UCI source of the 5,574 labeled SMS messages used here.
- [Fine-tune a pretrained model](https://huggingface.co/docs/transformers/training) — Docs · Hugging Face's guide to fine-tuning, including swapping in a classification head.
- [Text classification task guide](https://huggingface.co/docs/transformers/tasks/sequence_classification) — Docs · end-to-end sequence classification with `AutoModelForSequenceClassification`.
- [A Survey on Transfer Learning](https://ieeexplore.ieee.org/document/5288526) — Paper · Pan & Yang's foundational survey on why frozen pretrained features transfer.
- [Losses Learned — NLL and Cross-Entropy in PyTorch](https://sebastianraschka.com/blog/2022/losses-learned-part1.html) — Blog · the author on cross-entropy and why we use 2 output nodes rather than 1.
- [Finetuning Large Language Models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) — Blog · Raschka's overview of fine-tuning strategies, from feature-based to full updates.
:::

## Test your knowledge

```flashcards
Q: What is the difference between classification and instruction fine-tuning?
A: Classification fine-tuning trains a model to output one of a **fixed set of class labels** (e.g., spam / not spam). Instruction fine-tuning trains it to **follow free-form natural-language instructions** across many tasks.
---
Q: How is a pretrained GPT model's output layer modified for binary spam classification?
A: The original `Linear(768 → 50257)` vocabulary head is replaced with a new `Linear(768 → 2)` head — one output node per class.
---
Q: Why are messages padded, and which token is used as the pad token?
A: Variable-length messages must share a shape to batch as tensors, so shorter ones are padded to the longest length using `<|endoftext|>` (token ID 50256).
---
Q: Which layers are made trainable during fine-tuning, and which are frozen?
A: Trainable: the new output head, the final LayerNorm, and the last transformer block. Frozen: the token/positional embeddings and the other 11 transformer blocks.
---
Q: Why do we use only the last token's output for classification?
A: Causal attention lets each token attend only to itself and earlier tokens, so the **last** token is the only one that has "seen" the entire sequence — its representation summarizes the whole message.
---
Q: Why minimize cross-entropy loss instead of directly maximizing accuracy?
A: Accuracy is not a differentiable function, so it can't be optimized by gradient descent. Cross-entropy is a differentiable proxy whose minimization increases accuracy.
---
Q: How does the classification training loop differ from the pretraining loop?
A: Only two changes: it tracks the number of training **examples** seen (not tokens), and it computes **classification accuracy** each epoch instead of generating sample text.
---
Q: After fine-tuning for 5 epochs, roughly what test accuracy does the spam classifier reach, starting from what baseline?
A: It rises from a ~50% random baseline (balanced two-class problem) to about **96%** test accuracy.
```

```quiz
1. For a binary spam-classification task, how many output nodes does the new classification head have?
   - ( ) 1
   - (x) 2
   - ( ) 50257
   - ( ) 768
   > We use one output node per class (general approach): 2 nodes for "not spam" and "spam". Using 1 node is possible but would require a different loss function.

2. Which token ID is used to pad shorter messages to a uniform length?
   - ( ) 0
   - ( ) 1
   - (x) 50256
   - ( ) 768
   > GPT-2's `<|endoftext|>` token (ID 50256) is reused as the padding token.

3. During fine-tuning, which parts of the model are frozen (requires_grad = False)?
   - ( ) The new output head
   - ( ) The final LayerNorm
   - (x) The token/positional embeddings and the lower 11 transformer blocks
   - ( ) The last transformer block
   > Lower layers capture general language features and are frozen; only the last block, final norm, and new head are trained.

4. Why does the classifier use the output of the last token rather than the first?
   - (x) Causal attention means only the last token has attended to the entire sequence
   - ( ) The first token is always a padding token
   - ( ) The last token is processed first
   - ( ) It is chosen at random
   > With a causal mask, the final position is the only one whose representation incorporates information from every earlier token.

5. Why is cross-entropy loss used as the training objective instead of accuracy?
   - ( ) Accuracy is slower to compute
   - (x) Accuracy is not differentiable, so it can't be optimized by gradient descent
   - ( ) Cross-entropy always equals accuracy
   - ( ) Accuracy only works for regression
   > Gradient-based training needs a differentiable loss; cross-entropy is a differentiable proxy that, when minimized, raises accuracy.
```

```assignment "Exercise 6.2 — Fine-tune the whole model" level=intermediate
In this chapter we froze most of the network and trained only the last transformer block, the final LayerNorm, and the new classification head. Re-run the fine-tuning but make the **entire** model trainable, and compare the final test accuracy and training time against the partially-frozen version.

Hint: skip step 3 of Listing 6.7 (don't selectively unfreeze) and instead set `param.requires_grad = True` for every parameter after replacing the output head.
Hint: keep all other settings (AdamW, lr=5e-5, 5 epochs) the same so the comparison is fair.
Hint: expect higher compute cost per epoch — note whether the extra trainable parameters meaningfully improve test accuracy or mainly increase training time.
```
