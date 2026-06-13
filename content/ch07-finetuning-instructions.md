This is the final build step — and the one that turns a raw text predictor into something that feels like an assistant. So far our model can *complete* text, and in the last chapter we fine-tuned it to *classify* spam. But ask a pretrained LLM to "Convert this sentence to passive voice" and it tends to ramble, repeat the prompt, or ignore the request entirely. **Instruction fine-tuning** fixes that: we train the model on thousands of instruction–response pairs until following requests becomes second nature.

This is the technique behind ChatGPT-style assistants. We'll do the whole pipeline end to end — prepare a dataset, batch it with a clever masking trick, load GPT-2 medium, fine-tune it, extract its answers, and then have *another* LLM grade those answers automatically.

::: objectives "What you'll learn"
- Why pretrained LLMs ignore instructions, and how supervised instruction fine-tuning fixes it
- Formatting raw `{instruction, input, output}` entries with the **Alpaca prompt template**
- Building training batches: padding with `<|endoftext|>`, shifting targets, and masking padding with **-100**
- Wiring an `InstructionDataset` and a custom collate function into PyTorch data loaders
- Loading the larger **GPT-2 medium (355M)** and fine-tuning it on the instruction data
- Extracting and saving model responses for the held-out test set
- **Evaluating** with an LLM-as-judge: scoring responses 0–100 using Llama 3 via Ollama
:::

## Introduction to instruction fine-tuning

Pretraining teaches an LLM to generate one word at a time, so the result is a strong **text completer** — give it a fragment and it finishes the paragraph. But that's not the same as *following directions*. Asked to "Fix the grammar in this text" or "Provide a synonym for bright," a purely pretrained model often produces plausible-sounding text that doesn't actually do what was asked.

**Instruction fine-tuning** (also called *supervised instruction fine-tuning*) takes that pretrained model and further trains it on a dataset of explicit instruction–response pairs. After enough examples, the model learns the *pattern* of "read the request, produce the appropriate answer."

::: callout analogy "Teaching a brilliant but blunt expert some manners"
A pretrained LLM is like a wildly well-read person who, when you ask them a question, just keeps talking about whatever the topic reminds them of. They *know* an enormous amount — they're just not in the habit of answering *your* question. Instruction fine-tuning is the etiquette course: show them thousands of examples of "when asked X, respond with Y," and they learn to actually address the request instead of free-associating.
:::

The pipeline has three stages, and the whole chapter walks through them in order:

1. **Prepare the dataset** — download, format with a prompt template, and batch it.
2. **Fine-tune the LLM** — load a pretrained model and train it on the instruction data.
3. **Evaluate the LLM** — extract responses and score their quality.

::: diagram ch07-finetune-flow "The instruction fine-tuning pipeline: a pretrained GPT-2 medium is trained on a formatted instruction dataset until it reliably follows requests."
:::

## Preparing a dataset for supervised instruction fine-tuning

The dataset for this chapter has **1,100 instruction–response pairs**, created specifically for the book, in a small (204 KB) JSON file. JSON mirrors Python dictionaries, so it's both human-readable and trivial to load.

```python title="Listing 7.1 — Downloading the dataset"
import json
import os
import urllib

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:                                               # skip download if present
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)
data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))   # 1100
```

Each entry is a dictionary with three fields. The `'input'` field is sometimes empty:

```python title="Inspecting two entries"
print("Example entry:\n", data[50])
# {'instruction': 'Identify the correct spelling of the following word.',
#  'input': 'Ocassion', 'output': "The correct spelling is 'Occasion.'"}

print("Another example entry:\n", data[999])
# {'instruction': "What is an antonym of 'complicated'?",
#  'input': '', 'output': "An antonym of 'complicated' is 'simple'."}
```

### Formatting with the Alpaca prompt template

We can't feed raw dictionaries to the model — we need to flatten each entry into a single string. The book uses the **Alpaca prompt style**: a fixed preamble followed by labeled `### Instruction:`, `### Input:`, and `### Response:` sections. (Alpaca was one of the first models to publicly detail its instruction fine-tuning recipe, which is why its format became a de-facto standard; Microsoft's Phi-3 uses a simpler `<|user|>` / `<|assistant|>` style instead.)

::: callout analogy "A standard form everyone fills in"
The prompt template is like a government form with labeled boxes. Whether the request is about spelling, synonyms, or unit conversion, it always gets written onto the *same* form — preamble at top, instruction here, optional input there, response at the bottom. Because every example has identical structure, the model learns exactly where the "answer goes" and what shape the whole thing should take.
:::

::: diagram ch07-instruction-format "A raw entry (instruction, input, output) is reformatted into one Alpaca-style string. The ### Input: section is omitted when the input field is empty."
:::

The `format_input` function builds the preamble plus instruction, and conditionally appends the input section:

```python title="Listing 7.2 — Implementing the prompt formatting function"
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text
```

Note that the `### Response:` section is *not* part of `format_input` — that's deliberate. When formatting training data we append the response ourselves; at inference time we stop after `### Response:` so the model fills in the answer. Testing on `data[50]`:

```text title="The formatted input + response"
Below is an instruction that describes a task. Write a response that
appropriately completes the request.

### Instruction:
Identify the correct spelling of the following word.

### Input:
Ocassion

### Response:
The correct spelling is 'Occasion.'
```

For an entry with an empty `'input'` (like `data[999]`), the `### Input:` section simply doesn't appear.

### Splitting into train, validation, and test sets

Just as with the spam classifier, we partition the data — 85% train, 10% test, 5% validation:

```python title="Listing 7.3 — Partitioning the dataset"
train_portion = int(len(data) * 0.85)              # 85% for training
test_portion = int(len(data) * 0.1)                # 10% for testing
val_portion = len(data) - train_portion - test_portion   # remaining 5%

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("Training set length:", len(train_data))      # 935
print("Validation set length:", len(val_data))      # 55
print("Test set length:", len(test_data))           # 110
```

## Organizing data into training batches

To train efficiently we group multiple examples into a **batch**. But the examples have different lengths, and PyTorch tensors must be rectangular — so every sequence in a batch has to be padded to the same length. The default `DataLoader` collate function can't handle the special masking we need, so we build a **custom collate function**. We'll get there in five substeps: (2.1) apply the template, (2.2) tokenize, (2.3) pad, (2.4) create shifted targets, and (2.5) replace padding with the `-100` placeholder.

### The InstructionDataset class

First, a `Dataset` that formats *and* pre-tokenizes every entry up front (just like `SpamDataset` in chapter 6). Pre-tokenizing in `__init__` means the work happens once, not on every training step:

```python title="Listing 7.4 — Implementing an instruction dataset class"
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:                              # pretokenize every entry
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
```

### Padding to equal length

As in chapter 6, we use the `<|endoftext|>` token as the padding token. Rather than padding the *text*, we append its token ID directly to the pre-tokenized lists. We can confirm the ID with the tokenizer:

```python title="The <|endoftext|> token ID"
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
# [50256]
```

The clever part: instead of padding every example to the longest sequence in the *whole dataset*, we pad only to the longest sequence in *each batch*. That minimizes wasted compute, and it's exactly why batches can have different lengths. Here's a first draft that handles just the inputs:

```python title="custom_collate_draft_1 — padding the inputs"
def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)   # longest in batch
    inputs_lst = []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])     # drop the extra padding token
        inputs_lst.append(inputs)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor
```

Trying it on three short example lists:

```python title="Testing the padding"
inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (inputs_1, inputs_2, inputs_3)
print(custom_collate_draft_1(batch))
# tensor([[    0,     1,     2,     3,     4],
#         [    5,     6, 50256, 50256, 50256],
#         [    7,     8,     9, 50256, 50256]])
```

All three are now padded to length 5.

### Creating shifted target token IDs

For next-token prediction, the **targets** are the inputs shifted one position to the right — exactly as in pretraining. At each position the model sees the input token and must predict the *next* one. So we slice the padded sequence two ways: `padded[:-1]` for inputs, `padded[1:]` for targets.

```python title="custom_collate_draft_2 — adding shifted targets"
def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])    # truncate last token for inputs
        targets = torch.tensor(padded[1:])    # shift +1 to the right for targets
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
```

For `input [0, 1, 2, 3, 4]` the target becomes `[1, 2, 3, 4, 50256]` — the same IDs shifted left, minus the first input ID, plus a trailing end-of-text token.

### Masking padding tokens with -100

Now the key trick. If we leave the padding tokens (`50256`) in the targets, the model would be trained to *predict padding* — wasting capacity on meaningless filler. We replace those padding targets with the special value **-100**, which PyTorch's cross-entropy loss ignores by default (`ignore_index=-100`). But we keep the **first** end-of-text token, so the model still learns *when to stop generating*.

::: diagram ch07-batch-masking "Inputs are padded with 50256; targets are the inputs shifted right by one. All padding tokens in the target except the first are replaced with -100, which the loss ignores."
:::

::: callout analogy "Don't grade the blank parts of the test"
Imagine grading a stack of exams where students wrote answers of different lengths, and you padded every page to the same size with blank lines. You wouldn't *penalize* a student for the blank lines — they're just filler to make the pages line up. The `-100` value is your instruction to the grader: "skip these blanks; only score the real content." We keep one end-of-text marker, though, because the model genuinely needs to learn where an answer *ends*.
:::

```python title="Listing 7.5 — Implementing a custom batch collate function"
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (                                  # pad to batch_max_length
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])          # truncate last token for inputs
        targets = torch.tensor(padded[1:])          # shift +1 to the right for targets

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:                     # keep the first padding token,
            targets[indices[1:]] = ignore_index     # replace the rest with -100

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]     # optional length cap
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
```

Running it on the sample batch shows the result — inputs unchanged, targets with all-but-the-first padding turned into `-100`:

```python title="Testing the final collate function"
inputs, targets = custom_collate_fn(batch)
print(targets)
# tensor([[    1,     2,     3,     4, 50256],
#         [    6, 50256,  -100,  -100,  -100],
#         [    8,     9, 50256,  -100,  -100]])
```

Why does `-100` work? A tiny experiment makes it concrete. Compute cross-entropy on two tokens, then add a third — the loss changes. But replace that third target with `-100` and the loss is *identical* to the two-token case:

```python title="Why -100 is ignored by cross-entropy"
import torch
logits_2 = torch.tensor([[-1.0, 1.0], [-0.5, 1.5], [-0.5, 1.5]])

targets_2 = torch.tensor([0, 1, 1])
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)   # 0.7936

targets_3 = torch.tensor([0, 1, -100])
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)   # 1.1269
# loss_3 equals the original two-token loss — the -100 entry was skipped entirely
```

The default `cross_entropy(..., ignore_index=-100)` simply drops any target labeled `-100` from the calculation.

::: callout note "Masking the instruction too?"
A common extra step is to also mask the *instruction* tokens (not just padding) with `-100`, so the loss is computed only over the response. The idea: train the model to *generate answers* rather than memorize prompts. But researchers are divided — the 2024 paper "Instruction Tuning With Loss Over Instructions" found that *not* masking instructions can actually help. The book leaves instruction masking as an optional exercise and does **not** apply it.
:::

## Creating data loaders for an instruction dataset

With `InstructionDataset` and `custom_collate_fn` in hand, we just plug both into PyTorch `DataLoader`s. One detail: the collate function moves tensors to the device itself (CPU, `"cuda"`, or `"mps"` on Apple Silicon). Doing the transfer inside the collate function lets it run as a background process, so it doesn't block the GPU during training.

We use `functools.partial` to pre-fill the `device` and `allowed_max_length` arguments. Setting `allowed_max_length=1024` caps sequences at GPT-2's context length:

```python title="Pre-configuring the collate function"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.backends.mps.is_available():
#     device = torch.device("mps")          # uncomment for Apple Silicon
print("Device:", device)

from functools import partial
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)
```

Now the three loaders. Only the training loader shuffles and drops the last partial batch:

```python title="Listing 7.6 — Initializing the data loaders" collapsible
from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)
```

Inspecting the batch shapes confirms the per-batch padding — note how the token dimension varies from batch to batch (61, then 76, then 73…):

```python title="Batches have different lengths"
print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)
# torch.Size([8, 61]) torch.Size([8, 61])
# torch.Size([8, 76]) torch.Size([8, 76])
# torch.Size([8, 73]) torch.Size([8, 73])
# ...
```

## Loading a pretrained LLM

Now we load the model to fine-tune. Unlike the spam classifier, we use the **medium-sized GPT-2 with 355 million parameters** instead of the 124M model. Smaller models simply lack the capacity to learn the nuanced behaviors that good instruction-following requires. The loading code is identical to chapters 5 and 6 — we just pass `"gpt2-medium (355M)"`.

::: callout warning "This download is ~1.42 GB"
GPT-2 medium needs roughly 1.42 GB of storage — about three times the small model. Training it is also more compute-intensive: two epochs take ~0.9 minutes on an A100 GPU but ~16 minutes on an M3 MacBook Air CPU. If hardware is tight, you can fall back to `"gpt2-small (124M)"`.
:::

```python title="Listing 7.7 — Loading the pretrained model" collapsible
from gpt_download import download_and_load_gpt2
from chapter04 import GPTModel
from chapter05 import load_weights_into_gpt

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

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();
```

Before any fine-tuning, it's worth checking how the raw pretrained model does on an instruction. We format the first validation example ("Convert the active sentence to passive…") and generate with the same `generate` function from chapter 5. Since `generate` returns the input *plus* the output, we slice off the input to isolate the response:

```python title="Baseline: the pretrained model before fine-tuning"
from chapter05 import generate, text_to_token_ids, token_ids_to_text

torch.manual_seed(123)
input_text = format_input(val_data[0])
token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)
response_text = generated_text[len(input_text):].strip()
print(response_text)
```

The output is telling: the model creates a `### Response:` section but just **repeats the input sentence and part of the instruction** instead of converting to passive voice. It clearly doesn't yet follow instructions — which is exactly the gap we're about to close.

## Fine-tuning the LLM on instruction data

The heavy lifting is done. Because targets are next-token IDs (with padding masked out), we can reuse the *exact same* loss and training functions from pretraining in chapter 5 — `calc_loss_loader` and `train_model_simple`.

First, a baseline loss to confirm everything is wired up:

```python title="Initial loss before training"
from chapter05 import calc_loss_loader, train_model_simple

model.to(device)
torch.manual_seed(123)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
print("Training loss:", train_loss)      # 3.8259...
print("Validation loss:", val_loss)      # 3.7619...
```

Now the training run. We use `AdamW` with a small learning rate, just **two epochs**, and pass the first validation instruction as `start_context` so we can watch the model's responses improve as it trains:

```python title="Listing 7.8 — Instruction fine-tuning the pretrained LLM"
import time

start_time = time.time()
torch.manual_seed(123)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.00005, weight_decay=0.1
)
num_epochs = 2
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
```

The loss drops steeply and steadily:

```text title="Training progress"
Ep 1 (Step 000000): Train loss 2.637, Val loss 2.626
Ep 1 (Step 000005): Train loss 1.174, Val loss 1.103
Ep 1 (Step 000010): Train loss 0.872, Val loss 0.944
...
Ep 2 (Step 000230): Train loss 0.300, Val loss 0.657
Training completed in 0.87 minutes.
```

And the printed sample responses show the payoff. By the end of training, the model correctly converts "The chef cooks the meal every day." into its passive form: **"The meal is cooked every day by the chef."** — a task it completely failed before fine-tuning.

::: callout tip "Why only two epochs?"
The loss falls fast and then flattens, which means the model has already learned the instruction-following pattern. Training longer risks **overfitting** — memorizing the training answers at the expense of generalizing to new instructions. With a small dataset, restraint pays off.
:::

A loss curve (via the chapter 5 `plot_losses` helper) confirms the story: a sharp initial drop on both train and validation, then a gentler decline as the model converges. But loss is only a proxy — what we really care about is *response quality*, which we turn to next.

## Extracting and saving responses

To judge quality properly we generate responses on the **held-out test set** the model never saw during training, and save them for analysis. First, a quick side-by-side on three examples (input, correct answer, model answer):

```python title="Comparing model responses to reference answers"
torch.manual_seed(123)
for entry in test_data[:3]:
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
        .replace("### Response:", "")
        .strip()
    )
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")
```

The results are encouraging. For "Rewrite the sentence using a simile" the model answers *"The car is as fast as a bullet"* (reference: "as fast as lightning") — different wording, equally valid. It nails the author of *Pride and Prejudice* (Jane Austen). It only slips on one: it says thunderstorms come from *cumulus* clouds when the answer is *cumulonimbus* — close but not quite right.

Next we run generation over the **entire** test set and attach each response to its entry, saving everything to JSON:

```python title="Listing 7.9 — Generating test set responses"
from tqdm import tqdm

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
        .replace("### Response:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text

with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)        # indent for pretty-printing
```

Finally, we save the fine-tuned weights so we can reuse the model later without retraining:

```python title="Saving the fine-tuned model"
import re
file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"   # gpt2-medium355M-sft.pth
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")
# reload later with: model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))
```

## Evaluating the fine-tuned LLM

Eyeballing three responses doesn't scale to 110 (let alone thousands). Instruction-following models are typically evaluated three ways: **multiple-choice benchmarks** like MMLU (general knowledge), **human preference** comparisons like the LMSYS Chatbot Arena, and **automated conversational** scoring where another LLM grades the answers (e.g., AlpacaEval). Since we care about conversational quality, we'll use the automated approach.

::: callout analogy "An automated grader with an answer key"
Hand-grading every response is like a teacher reading 110 essays by hand — accurate but exhausting. Instead we hire an *automated grader*: a separate, larger LLM that reads the instruction, the reference answer, and our model's response, then assigns a score. Like a human TA with the answer key in hand, it can even award partial credit when an answer is close but imperfect.
:::

We use Meta's instruction-tuned **8-billion-parameter Llama 3** as the judge, run locally with the open-source **Ollama** app. Ollama wraps the `llama.cpp` library to run LLMs efficiently on a laptop — it's an *inference* tool only (no training).

::: callout note "Setting up Ollama"
Install Ollama from [ollama.com](https://ollama.com), then either launch the app or run `ollama serve` in a separate terminal. Pull and try the model with `ollama run llama3` (a 4.7 GB download; needs ~16 GB RAM — use `phi3` for ~8 GB, or `llama3:70b` if you have a powerful machine). Keep Ollama running for the rest of the chapter.
:::

::: diagram ch07-llm-judge "The instruction, the reference answer, and the model's response are sent to Llama 3 via Ollama, which returns a quality score from 0 to 100."
:::

A small guard confirms Ollama is alive before we lean on it:

```python title="Verifying Ollama is running"
import psutil

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_if_running("ollama")
if not ollama_running:
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
print("Ollama running:", check_if_running("ollama"))
```

We talk to Llama 3 through Ollama's REST API. `temperature: 0` and a fixed `seed` make responses as deterministic as possible:

```python title="Listing 7.10 — Querying a local Ollama model" collapsible
import urllib.request

def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat"
):
    data = {                                  # build the request payload
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {                          # settings for deterministic output
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }
    payload = json.dumps(data).encode("utf-8")    # JSON string -> bytes
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )
    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:   # stream the reply
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data
```

The scoring prompt feeds Llama 3 the formatted input, the correct output, and our model's response, asking for a 0–100 score. On the three test examples, Llama 3 gives nuanced, well-reasoned grades: **85** for the "as fast as a bullet" simile (valid but slightly less vivid than "lightning"), **40** for the cumulus/cumulonimbus mix-up (acknowledging partial correctness), and **95** for the Jane Austen answer (correct, if a touch verbose). The judge clearly understands partial credit.

For a single summary number, we tweak the prompt to return only an integer and average across the whole test set:

```python title="Listing 7.11 — Evaluating the instruction fine-tuning LLM"
def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."     # just the number
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue
    return scores

scores = generate_model_scores(test_data, "model_response")
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}")
# Number of scores: 110 of 110
# Average score: 50.32
```

Our fine-tuned GPT-2 medium scores **~50** on average. For context, with the same methodology the Llama 3 8B *base* model scores 58.51 and the Llama 3 8B *instruct* model scores 82.6. Our small model isn't going to win awards — but going from "ignores instructions entirely" to a measurable ~50 with two epochs of fine-tuning on 935 examples is a real result.

::: callout tip "Pushing the score higher"
To improve from here: tune hyperparameters (learning rate, batch size, epochs), grow and diversify the training set, experiment with prompt formats, or start from a larger pretrained model. Because Ollama isn't fully deterministic across operating systems, running the evaluation a few times and averaging gives more stable numbers.
:::

## What's next — and a final word

That's the whole journey, end to end: you built a GPT architecture from scratch, pretrained it, loaded real OpenAI weights, fine-tuned for classification, and now fine-tuned for instruction-following — the same recipe behind real chatbots and assistants.

Where to go from here? One natural next step is **preference fine-tuning** (e.g., DPO or RLHF), which nudges a model to better match human preferences after instruction tuning — the book's GitHub repo has a DPO notebook. To keep learning, follow new work on [arXiv](https://arxiv.org/list/cs.LG/recent), communities like r/LocalLLaMA, and the author's own [blog](https://magazine.sebastianraschka.com). For real-world projects, tools like Axolotl and LitGPT build on these foundations.

Most importantly: you now understand LLMs from the inside out. Building one from scratch is the surest way to demystify them — and that hard-won intuition will serve you no matter how fast the field moves. Thanks for coming along on the journey.

## Key takeaways

::: takeaways
- **Instruction fine-tuning** adapts a pretrained text-completion LLM into one that follows human instructions and produces desired responses.
- Dataset prep means downloading instruction–response pairs, formatting each with a prompt template (the **Alpaca style**: preamble + `### Instruction` / `### Input` / `### Response`), and splitting into train/val/test.
- A **custom collate function** builds batches: pad to the longest sequence *in each batch* with `<|endoftext|>` (50256), create targets by shifting inputs right by one.
- Padding targets are replaced with **-100**, which PyTorch's cross-entropy ignores — so blank padding never contributes to the loss. One end-of-text token is kept so the model learns when to stop.
- We fine-tune the larger **GPT-2 medium (355M)** because smaller models lack the capacity for quality instruction-following; the training loop is the same one used for pretraining.
- Evaluation is harder than for classification: we extract responses on the test set and score them. The **LLM-as-judge** approach uses another model (Llama 3 via **Ollama**) to rate responses 0–100, then averages for one quality number.
:::

## Additional references

::: refs
- [Chapter 7 code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07) — GitHub · the complete instruction fine-tuning notebooks, dataset, and Ollama evaluation script.
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) — GitHub · the project that popularized the instruction prompt template used in this chapter, with 52K instruction-following examples.
- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) — GitHub · the automated LLM-as-judge benchmark that inspired this chapter's evaluation approach.
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Paper · OpenAI's InstructGPT, the foundational work on instruction tuning + RLHF behind ChatGPT.
- [Ollama](https://ollama.com) — Tool · run open LLMs like Llama 3 locally; used here as the automated evaluation judge.
- [Instruction Tuning With Loss Over Instructions](https://arxiv.org/abs/2405.14394) — Paper · evidence that *not* masking instruction tokens can improve instruction-tuned model performance.
- [Finetuning Large Language Models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) — Blog · the author's overview of fine-tuning strategies, from full fine-tuning to parameter-efficient methods.
:::

## Test your knowledge

```flashcards
Q: Why doesn't a purely pretrained LLM follow instructions well?
A: Pretraining only teaches next-word **completion**. The model produces plausible continuations but isn't trained to recognize a request and respond to it. Instruction fine-tuning adds that behavior with explicit instruction–response examples.
---
Q: What three labeled sections make up the Alpaca prompt template?
A: `### Instruction:`, an optional `### Input:`, and `### Response:` — preceded by a fixed preamble. The `### Input:` section is omitted when the entry's input field is empty.
---
Q: How are target token IDs created from the inputs during batching?
A: Targets are the inputs **shifted one position to the right** (drop the first input token, append a trailing end-of-text token), so each position predicts the next token.
---
Q: What is the padding token for instruction fine-tuning, and what is its ID?
A: The `<|endoftext|>` token, with ID **50256**. It pads every sequence in a batch to the batch's longest length.
---
Q: What does the value -100 do in the target sequences?
A: PyTorch's cross-entropy uses `ignore_index=-100` by default, so any target labeled -100 is **excluded from the loss**. We use it to mask padding tokens so they don't affect training.
---
Q: Why keep the first end-of-text (50256) token in the target instead of masking all of them?
A: So the model learns **when to stop** generating — the first end-of-text token signals the response is complete. Only the *extra* padding tokens after it are replaced with -100.
---
Q: Why does this chapter use GPT-2 medium (355M) instead of the 124M model?
A: Smaller models lack the **capacity** to learn the nuanced behaviors needed for high-quality instruction-following; the 124M model gives unsatisfactory results.
---
Q: How does the LLM-as-judge evaluation work?
A: Another LLM (Llama 3 via Ollama) is given the instruction, the reference answer, and the model's response, and asked to score it 0–100. Averaging scores over the test set yields one quality number.
```

```quiz
1. During batching, sequences are padded to:
   - ( ) the longest sequence in the entire dataset
   - (x) the longest sequence within each individual batch
   - ( ) a fixed length of 1,024 for every batch
   - ( ) the shortest sequence in the batch
   > Padding per-batch (not per-dataset) minimizes wasted compute, which is why different batches can have different token lengths.

2. What is special about the target value -100?
   - ( ) It marks the start of a response
   - ( ) It is the token ID for <|endoftext|>
   - (x) PyTorch's cross-entropy ignores it by default (ignore_index=-100)
   - ( ) It tells the model to generate a longer response
   > The default cross_entropy(..., ignore_index=-100) skips any target labeled -100, so masked padding doesn't affect the loss.

3. Why is one <|endoftext|> token kept in each target rather than masking all of them?
   - (x) So the model learns when to stop generating a response
   - ( ) Because cross-entropy requires at least one padding token
   - ( ) To make all targets the same length
   - ( ) To increase the loss value
   > Retaining the first end-of-text token teaches the model to signal a complete response; only the extra padding after it becomes -100.

4. Which model is used as the automated judge to score responses?
   - ( ) GPT-2 medium, the model being fine-tuned
   - ( ) GPT-4 via the OpenAI API (the only option)
   - (x) An 8B Llama 3 model run locally with Ollama
   - ( ) A human annotator scoring all 110 responses
   > The chapter uses Llama 3 (8B) via Ollama as a local LLM-as-judge; GPT-4 is mentioned only as an optional alternative.

5. Why does the chapter train for just two epochs?
   - ( ) Because GPT-2 medium can only train for two epochs
   - ( ) To make the download smaller
   - (x) The loss flattens quickly, and more epochs risk overfitting on the small dataset
   - ( ) Because the AdamW optimizer requires exactly two epochs
   > Loss drops sharply then plateaus, signaling the pattern is learned; extra epochs mainly invite overfitting.
```

```assignment "Exercise 7.3 — Fine-tune on the original Alpaca dataset" level=intermediate
The Stanford Alpaca dataset is one of the earliest and most popular open instruction datasets, with 52,002 entries — roughly 50× more than the dataset used in this chapter, and with longer examples on average. Swap it in for `instruction-data.json`, fine-tune the model, and compare the resulting evaluation scores against the smaller dataset.

Hint: the dataset is available at https://github.com/tatsu-lab/stanford_alpaca (also mirrored at https://mng.bz/NBnE).
Hint: because it's much larger, strongly prefer a GPU. If you hit out-of-memory errors, reduce `batch_size` from 8 to 4, 2, or 1.
Hint: lowering `allowed_max_length` from 1,024 to 512 or 256 also helps manage memory.
```
