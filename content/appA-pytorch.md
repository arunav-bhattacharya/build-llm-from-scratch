Every line of code in this book runs on **PyTorch** — an open-source, Python-based deep learning library that has been the most popular framework for AI research since 2019. This appendix is a from-zero primer: it teaches just enough PyTorch to build and train large language models, without trying to be an exhaustive reference.

If you're already comfortable with PyTorch — tensors, autograd, `nn.Module`, data loaders, and the training loop — feel free to skim or skip ahead. If any of those terms are new, this is the gentlest on-ramp in the book. Take it slowly; everything else builds on these foundations.

::: objectives "What you'll learn"
- The **three components** of PyTorch: a tensor library, an autograd engine, and deep-learning utilities
- **Tensors** — scalars, vectors, matrices, and higher-dimensional arrays — plus dtypes and core operations
- How PyTorch sees models as **computation graphs**, and how **autograd** computes gradients automatically
- Building neural networks by subclassing **`nn.Module`**
- Feeding data efficiently with **`Dataset`** and **`DataLoader`**
- A **typical training loop**, and how to **save and load** trained models
- Accelerating training on a **GPU** with `.to(device)` — and a peek at multi-GPU training
:::

## What is PyTorch?

PyTorch is popular because it strikes a rare balance: a friendly, Pythonic interface that doesn't sacrifice flexibility or speed. You can prototype quickly *and* drop down to low-level details when you need to. It's really three libraries in one.

::: diagram appA-tensors "A tensor's rank is simply its number of dimensions: a scalar (rank 0), a vector (rank 1), a matrix (rank 2), and a 3-D tensor (rank 3) — a stack of grids."
:::

**The three core components:**

1. A **tensor library** — like NumPy's array programming, but with the crucial ability to run on GPUs, switching seamlessly between CPU and GPU.
2. An **automatic differentiation engine** (*autograd*) — it computes gradients of tensor operations automatically, which is what makes backpropagation easy.
3. A **deep-learning library** — modular building blocks: prebuilt layers, loss functions, optimizers, and pretrained models.

::: callout note "AI vs. machine learning vs. deep learning"
These nest like Russian dolls. **AI** is the broad goal of machines doing tasks that need human intelligence. **Machine learning** is the subfield where systems learn from data instead of being explicitly programmed. **Deep learning** is the subfield of ML that uses deep neural networks — many layers of artificial neurons. LLMs are deep neural networks, and PyTorch is the deep-learning library we'll use to build them.
:::

### Installing PyTorch

PyTorch installs like any Python package. The default command is `pip install torch`; the book uses version 2.4.0, so for guaranteed compatibility:

```python title="Install and verify PyTorch"
# in your terminal:  pip install torch==2.4.0
import torch
torch.__version__          # '2.4.0'
torch.cuda.is_available()  # True if an NVIDIA CUDA GPU is detected
```

If your machine has a **CUDA-compatible NVIDIA GPU**, `pip install torch` automatically installs the GPU-accelerated build. To pin a specific CUDA version, use the installation selector at [pytorch.org](https://pytorch.org). On an **Apple Silicon** Mac (M1/M2/M3+), check `torch.backends.mps.is_available()` instead — `True` means you can accelerate PyTorch with the Apple GPU.

::: callout tip "No GPU? No problem (for now)"
GPUs are **not required** for the early chapters — everything runs on CPU, just slower. When you do want one, Google Colab offers free time-limited GPU access via its *Runtime → Change runtime type* menu.
:::

## Understanding tensors

A **tensor** generalizes numbers, vectors, and matrices to any number of dimensions. Its **rank** (or *order*) is its number of dimensions: a scalar is rank 0, a vector rank 1, a matrix rank 2, and beyond that we just say "3-D tensor," "4-D tensor," and so on. From a programming view, tensors are simply efficient **multidimensional data containers**.

::: callout analogy "Tensors are nested egg cartons"
Picture storage by dimension. A **scalar** is a single egg. A **vector** is one row of an egg carton. A **matrix** is the full carton — a grid of rows and columns. A **3-D tensor** is a *crate* holding several cartons stacked together; a 4-D tensor is a *truck* full of crates. The rank just tells you how deeply the boxes are nested.
:::

PyTorch tensors are deliberately close to NumPy arrays, but with two superpowers NumPy lacks: a built-in **autograd** engine for gradients, and **GPU acceleration**.

```python title="Listing A.1 — Creating tensors of different ranks"
import torch
tensor0d = torch.tensor(1)                      # rank 0 — a scalar
tensor1d = torch.tensor([1, 2, 3])              # rank 1 — a vector
tensor2d = torch.tensor([[1, 2],
                         [3, 4]])                # rank 2 — a matrix
tensor3d = torch.tensor([[[1, 2], [3, 4]],
                         [[5, 6], [7, 8]]])      # rank 3 — a 3-D tensor
```

### Tensor data types

A tensor's `.dtype` tells you the element type. PyTorch follows Python's defaults: whole numbers become 64-bit integers, but floats become **32-bit** by default — a deliberate trade-off, since `float32` gives plenty of precision for deep learning while using less memory and running faster (GPUs are optimized for it). Change a dtype with `.to()`:

```python title="Inspecting and changing dtypes"
torch.tensor([1, 2, 3]).dtype          # torch.int64   (integers)
torch.tensor([1.0, 2.0, 3.0]).dtype    # torch.float32 (floats — the DL default)

floatvec = torch.tensor([1, 2, 3]).to(torch.float32)
floatvec.dtype                          # torch.float32
```

### Common tensor operations

You'll use a small handful of operations constantly. The `.shape` attribute reports dimensions; `.reshape()` or `.view()` rearrange them; `.T` transposes; and `@` (or `.matmul()`) multiplies matrices.

```python title="The operations you'll use most"
tensor2d = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])
tensor2d.shape              # torch.Size([2, 3]) — 2 rows, 3 columns
tensor2d.view(3, 2)         # reshape to 3×2
tensor2d.T                  # transpose → 3×2, flipped across the diagonal
tensor2d @ tensor2d.T       # matrix multiply → 2×2:  [[14, 32], [32, 77]]
```

::: callout note "`.view()` vs. `.reshape()`"
Both change a tensor's shape. The difference is memory: `.view()` only works when the underlying data is **contiguous** in memory (and fails otherwise), while `.reshape()` always works — copying the data if it must. `.view()` is the more common choice in PyTorch code when you know the layout is contiguous.
:::

## Seeing models as computation graphs

Here's the mental model that makes autograd click. A **computation graph** is a directed graph that lays out the sequence of operations needed to compute an output. PyTorch builds one automatically, in the background, as your code runs — and it later walks that graph to compute gradients for backpropagation.

Consider a tiny **logistic regression** classifier (a single-layer network). It multiplies an input by a weight, adds a bias, squashes the result through a sigmoid, and compares the output to the true label to get a loss:

```python title="Listing A.2 — A logistic regression forward pass"
import torch.nn.functional as F

y  = torch.tensor([1.0])   # true label
x1 = torch.tensor([1.1])   # input feature
w1 = torch.tensor([2.2])   # weight parameter
b  = torch.tensor([0.0])   # bias unit

z = x1 * w1 + b            # net input
a = torch.sigmoid(z)       # activation / output
loss = F.binary_cross_entropy(a, y)
```

Don't worry if logistic regression is unfamiliar — the point isn't the classifier, it's the *shape* of the computation. Each operation is a node; data flows along the edges from inputs and parameters, through the operations, to the loss.

::: diagram appA-computation-graph "The forward pass as a computation graph: input x1 times weight w1, plus bias b, gives the net input z; a sigmoid produces output a; comparing a to label y yields the loss. Data flows left to right."
:::

## Automatic differentiation made easy

To **train** a model we need to know how the loss changes when we nudge each parameter — that's the **gradient**. Computing gradients by hand is tedious and error-prone. PyTorch's autograd engine does it for us: whenever a tensor has `requires_grad=True`, PyTorch tracks every operation it touches, building the computation graph needed to compute gradients via the **chain rule** from calculus.

::: callout analogy "Autograd is a receipt that remembers every step"
Imagine every arithmetic operation drops an itemized receipt into a shoebox as it happens — *"multiplied w1 by x1," "added b," "applied sigmoid."* When you finally ask "how did each ingredient affect the total?", PyTorch reads the receipts **backward**, applying the chain rule at each step, and hands you the gradient for every parameter. You never do the calculus yourself — autograd kept the books.
:::

::: callout math "Gradients, partial derivatives, and the chain rule"
A **partial derivative** measures how fast a function changes with respect to *one* of its inputs. A **gradient** is the vector of all such partial derivatives. Backpropagation computes the gradient of the loss with respect to each parameter by applying the **chain rule** from the loss backward to the inputs — also called *reverse-mode automatic differentiation*. All you need to remember: it tells us how to nudge each parameter to reduce the loss.
:::

You can request a single gradient explicitly with `grad`, which is handy for experimentation:

```python title="Listing A.3 — Computing gradients via autograd"
import torch.nn.functional as F
from torch.autograd import grad

y  = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)   # track this parameter
b  = torch.tensor([0.0], requires_grad=True)

z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)  # (tensor([-0.0898]),)
grad_L_b  = grad(loss, b,  retain_graph=True)  # (tensor([-0.0817]),)
```

In practice you almost never call `grad` directly. Instead, call **`.backward()`** on the loss — PyTorch computes the gradients of *every* leaf parameter at once and stores them in each tensor's **`.grad`** attribute:

```python title="The one-liner you'll actually use"
loss.backward()
print(w1.grad)   # tensor([-0.0898])
print(b.grad)    # tensor([-0.0817])
```

::: diagram appA-autograd "Backpropagation runs the graph in reverse: starting at the loss, autograd applies the chain rule through each operation, multiplying partial derivatives until it reaches the gradient of the loss with respect to each parameter."
:::

That's the entire takeaway from the calculus: **PyTorch handles the derivatives for you** via `.backward()`. You will never differentiate by hand in this book.

## Implementing multilayer neural networks

Real networks stack many layers. To define one, subclass **`torch.nn.Module`** — the base class that tracks your layers and parameters and gives you training machinery for free. The pattern is always the same: declare the layers in `__init__`, and describe how data flows through them in `forward`. (You almost never write a `backward` method — autograd derives it from `forward`.)

```python title="Listing A.4 — A multilayer perceptron with two hidden layers"
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
```

`torch.nn.Sequential` chains layers so a single `self.layers(x)` call runs them in order. A `Linear` layer multiplies its input by a weight matrix and adds a bias (a *fully connected* layer); `ReLU` is a nonlinear activation placed *between* hidden layers so the network can model complex, nonlinear relationships.

Printing the model shows its structure, and we can count its trainable parameters:

```python title="Inspecting the model"
model = NeuralNetwork(50, 3)
print(model)
# NeuralNetwork(
#   (layers): Sequential(
#     (0): Linear(in_features=50, out_features=30, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=30, out_features=20, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=20, out_features=3, bias=True)
#   )
# )

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(num_params)   # 2213
```

Each `Linear` layer's weights and biases default to `requires_grad=True`, so they're trainable. You can reach into a layer directly — `model.layers[0].weight` is the first layer's `30 × 50` weight matrix.

::: callout note "Why weights start as small random numbers"
PyTorch initializes weights with small random values to **break symmetry**: if every weight started identical, all neurons would compute the same thing and learn the same updates, and the network couldn't learn anything useful. For reproducible randomness, seed the generator with `torch.manual_seed(123)` before creating the model.
:::

Feeding the model an input runs the forward pass automatically. For **inference** (prediction, no training), wrap the call in `torch.no_grad()` so PyTorch skips building the gradient graph — saving memory and computation:

```python title="Forward pass and inference"
torch.manual_seed(123)
X = torch.rand((1, 50))        # one random 50-dim example
out = model(X)                 # calling model(X) runs forward()
# tensor([[-0.1262, 0.1080, -0.1792]], grad_fn=<AddmmBackward0>)

with torch.no_grad():          # inference: no gradient tracking
    out = model(X)
```

::: callout tip "Return logits, not probabilities"
By convention, models return raw **logits** (the last layer's outputs) *without* a final softmax. PyTorch's loss functions fold the softmax in for numerical stability. When you actually want class probabilities, apply `torch.softmax(out, dim=1)` yourself — the results then sum to 1.
:::

## Setting up efficient data loaders

Training reads data in **mini-batches**, shuffled and reshuffled each pass. PyTorch splits this into two classes: a **`Dataset`** that knows how to fetch one example, and a **`DataLoader`** that wraps a `Dataset` to handle batching, shuffling, and parallel loading.

Let's start with a tiny toy dataset — five training examples with two features each, plus a small test set:

```python title="Listing A.5 — A small toy dataset"
X_train = torch.tensor([[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6],
                        [ 2.3, -1.1], [ 2.7, -1.5]])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test  = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
y_test  = torch.tensor([0, 1])
```

A custom `Dataset` needs exactly three methods: `__init__` (set up the data), `__getitem__` (return one example by index), and `__len__` (report the total count):

```python title="Listing A.6 — A custom Dataset class"
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):           # return one (features, label) pair
        return self.features[index], self.labels[index]

    def __len__(self):                       # total number of examples
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds  = ToyDataset(X_test, y_test)
```

::: callout analogy "The DataLoader is a conveyor belt of shuffled batches"
Think of the `Dataset` as a warehouse shelf — it can hand you any single item if you give it an index. The `DataLoader` is the **conveyor belt** in front of it: it scoops items off the shelf in a shuffled order, packs them into evenly sized boxes (batches), and feeds those boxes to your model one after another. One full pass over the warehouse is an **epoch**.
:::

Now wrap each `Dataset` in a `DataLoader`. We **shuffle** the training data (so the model doesn't learn the order) but not the test data:

```python title="Listing A.7 — Instantiating data loaders"
from torch.utils.data import DataLoader

torch.manual_seed(123)
train_loader = DataLoader(dataset=train_ds, batch_size=2,
                          shuffle=True,  num_workers=0)
test_loader  = DataLoader(dataset=test_ds,  batch_size=2,
                          shuffle=False, num_workers=0)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)
# Batch 1: [[-1.2, 3.1], [-0.5, 2.6]]  [0, 0]
# Batch 2: [[ 2.3,-1.1], [-0.9, 2.9]]  [1, 0]
# Batch 3: [[ 2.7,-1.5]]               [1]
```

With five examples and `batch_size=2`, the last batch holds a lone example. A stray tiny batch can disturb training, so it's common to set **`drop_last=True`** to discard it:

```python title="Listing A.8 — Dropping the last (partial) batch"
train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True,
                          num_workers=0, drop_last=True)
```

::: callout tip "What does num_workers do?"
`num_workers` controls how many background processes prefetch data. With `num_workers=0`, the **main process** loads each batch — fine for tiny data, but on a GPU it leaves the GPU idle while the CPU fetches the next batch. Setting `num_workers=4` (a good default for real datasets) lets workers queue up batches in parallel, keeping the GPU fed. For tiny datasets or Jupyter notebooks, leave it at 0 — extra workers add overhead and can even cause crashes.
:::

## A typical training loop

Now we can train. The loop has a fixed rhythm you'll see in every chapter: for each **epoch**, iterate over the loader's batches, and for each batch run **forward → loss → zero_grad → backward → step**.

::: diagram appA-training-loop "The training loop cycle: for each batch, run a forward pass to logits, compute the loss, zero out old gradients, backpropagate new ones, then step the optimizer to update the weights — and repeat."
:::

```python title="Listing A.9 — A neural network training loop"
import torch.nn.functional as F

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)                  # forward pass
        loss = F.cross_entropy(logits, labels)    # compute loss

        optimizer.zero_grad()                     # clear old gradients
        loss.backward()                           # backpropagate
        optimizer.step()                          # update the weights

        print(f"Epoch {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Loss: {loss:.2f}")
    model.eval()
    # optional: evaluation code here
```

The three lines in the middle are the heart of learning. **`optimizer.zero_grad()`** resets gradients to zero — PyTorch *accumulates* gradients by default, so without this they'd pile up across batches. **`loss.backward()`** fills every parameter's `.grad`. **`optimizer.step()`** uses those gradients to update the weights; here we use **SGD** (stochastic gradient descent) with a **learning rate** of 0.5. The learning rate and number of epochs are *hyperparameters* you tune by watching the loss.

::: callout warning "Always zero your gradients"
If you forget `optimizer.zero_grad()`, gradients from previous batches **accumulate** onto the current ones, corrupting every update. The standard order is **zero_grad → backward → step**, once per batch.
:::

::: callout note "model.train() and model.eval()"
These flip the model between training and evaluation modes. Some layers — notably **dropout** and **batch normalization** — behave differently during training versus inference. Our toy network has neither, so the calls are technically redundant here, but including them is best practice so your code stays correct when you add such layers later.
:::

After training, make predictions in `eval()` mode under `no_grad()`, then turn logits into class labels with **`argmax`**:

```python title="Making predictions and measuring accuracy"
model.eval()
with torch.no_grad():
    outputs = model(X_train)
predictions = torch.argmax(outputs, dim=1)   # pick the highest-scoring class
correct = torch.sum(predictions == y_train)  # count correct predictions
```

This pattern generalizes into a reusable accuracy helper that works on any data loader, batch by batch — so it scales to datasets too large to fit in memory at once:

```python title="Listing A.10 — A reusable accuracy function" collapsible
def compute_accuracy(model, dataloader):
    model = model.eval()
    correct, total_examples = 0.0, 0
    for features, labels in dataloader:
        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()

print(compute_accuracy(model, train_loader))   # 1.0
print(compute_accuracy(model, test_loader))    # 1.0
```

## Saving and loading models

Once a model is trained, you'll want to reuse it without retraining. The recommended way is to save the model's **`state_dict`** — a Python dictionary mapping each layer to its learned weights and biases:

```python title="Save the learned parameters"
torch.save(model.state_dict(), "model.pth")
```

To restore it, recreate an instance of the **same architecture**, then load the saved parameters into it:

```python title="Load the parameters back"
model = NeuralNetwork(2, 2)                       # same architecture as saved
model.load_state_dict(torch.load("model.pth"))
```

::: callout note "Why save the state_dict, not the whole model"
Saving just the `state_dict` (the parameters) — rather than the entire Python object — is portable and robust: it doesn't pickle your class definitions or file paths. The catch is that you must rebuild the *exact* architecture (`NeuralNetwork(2, 2)` here) before loading, since the saved weights only fit a matching network. The `.pth` and `.pt` extensions are the usual conventions.
:::

## Optimizing training performance with GPUs

GPUs dramatically accelerate the matrix-heavy math of deep learning. In PyTorch, a **device** is simply where a tensor lives and where its operations run — the CPU and each GPU are devices. The rule is simple: **all tensors in a computation must be on the same device.**

::: callout analogy "`.to(device)` moves work to a faster workshop"
By default your tensors work in a small home workshop (the CPU) — fine for one-off tasks. `.to("cuda")` ships them to a massive, highly parallel factory (the GPU) built for crunching huge matrices at once. But the workers and the materials must be in the *same* building: if your model is in the factory and your data is still at home, the computation fails. Move both with `.to(device)`.
:::

Transfer tensors with the same `.to()` method you used for dtypes — and the result carries its device with it:

```python title="Moving tensors to the GPU"
tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])
print(tensor_1 + tensor_2)              # tensor([5., 7., 9.])  — on CPU

tensor_1 = tensor_1.to("cuda")
tensor_2 = tensor_2.to("cuda")
print(tensor_1 + tensor_2)              # tensor([5., 7., 9.], device='cuda:0')
```

::: diagram appA-gpu "A tensor created on the CPU is transferred to the GPU with .to('cuda'); the result reports device='cuda:0'. Both the model and its data must live on the same device."
:::

If one tensor is on the CPU and another on the GPU, PyTorch raises a `RuntimeError: Expected all tensors to be on the same device`. The fix is always to put everything on the same device.

### Single-GPU training

Adapting the training loop to a GPU takes **just three lines**: define a `device`, move the model to it, and move each batch to it inside the loop:

```python title="Listing A.11 — A training loop on a GPU"
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
device = torch.device("cuda")          # (1) pick the device
model = model.to(device)               # (2) move the model
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(num_epochs):
    model.train()
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)   # (3) move data
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

::: callout tip "Make your code run anywhere"
Hard-coding `"cuda"` crashes on machines without an NVIDIA GPU. The portable idiom falls back to CPU automatically:

`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

On Apple Silicon, use `"mps" if torch.backends.mps.is_available() else "cpu"`. This is best practice when sharing code.
:::

### Training with multiple GPUs

When one GPU isn't enough, **distributed training** spreads the work across many. PyTorch's simplest strategy is **`DistributedDataParallel` (DDP)**: it places a *copy* of the model on each GPU, feeds each copy a different, non-overlapping slice of every batch (via a `DistributedSampler`), then **synchronizes** the gradients across all GPUs so every replica stays identical.

::: callout analogy "DDP is a team grading the same exam in parallel"
Imagine grading a huge stack of exams. DDP hands each teaching assistant (GPU) an identical answer key (the model) and a different pile of exams (a data shard). Everyone grades simultaneously, then they huddle to **average their feedback** so all the answer keys get updated the same way. With eight assistants you finish roughly eight times faster — minus a little time spent huddling.
:::

The payoff is near-linear speedup: two GPUs can process an epoch in roughly half the time, eight GPUs about eight times faster, aside from minor communication overhead. The setup involves a few extra pieces — spawning one process per GPU with `torch.multiprocessing`, initializing a process group, and wrapping the model in `DDP`:

```python title="Listing A.13 — Core of DDP training (run as a script, not in Jupyter)" collapsible
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)            # rank = the GPU's index/ID

def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size)
    # ... build train_loader with sampler=DistributedSampler(train_ds) ...
    model = NeuralNetwork(num_inputs=2, num_outputs=2).to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model = DDP(model, device_ids=[rank])  # wrap → syncs gradients across GPUs
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)
            # ... forward, loss, zero_grad, backward, step ...
    destroy_process_group()                # clean up at the end

if __name__ == "__main__":
    world_size = torch.cuda.device_count()         # number of GPUs
    mp.spawn(main, args=(world_size, 3), nprocs=world_size)  # one process per GPU
```

::: callout warning "DDP must run as a standalone script"
DDP spawns multiple processes, each needing its own Python interpreter — which **Jupyter notebooks can't provide**. Run DDP code as a `.py` script from the terminal (`python ddp_script.py`), not in a notebook. To restrict which GPUs are used, set the `CUDA_VISIBLE_DEVICES` environment variable, e.g. `CUDA_VISIBLE_DEVICES=0,2 python script.py`.
:::

You won't need multiple GPUs for this book — the early chapters run comfortably on a CPU or a single GPU. But knowing DDP exists is useful: it's how real LLMs are trained at scale.

## Key takeaways

::: takeaways
- PyTorch has **three components**: a NumPy-like **tensor library** with GPU support, an **autograd** engine for automatic gradients, and **deep-learning utilities** (layers, losses, optimizers).
- A **tensor** is a multidimensional array; its **rank** is its number of dimensions. Floats default to `float32`. Reshape with `.view`/`.reshape`, transpose with `.T`, multiply with `@`.
- PyTorch silently builds a **computation graph** of your operations; **autograd** walks it backward (the chain rule) when you call **`loss.backward()`**, storing each parameter's gradient in `.grad`. You never differentiate by hand.
- Build networks by subclassing **`nn.Module`**: declare layers in `__init__`, define data flow in `forward`. Models return raw **logits**; wrap inference in **`torch.no_grad()`**.
- A **`Dataset`** fetches one example; a **`DataLoader`** batches and shuffles them. One full pass is an **epoch**.
- The training loop is a fixed cycle: **forward → loss → `zero_grad` → `backward` → `step`**, per batch, per epoch.
- Save a model's **`state_dict`** to disk; restore by recreating the same architecture and calling `load_state_dict`.
- Move work to a GPU with **`.to(device)`** — model *and* data must share one device. **`DistributedDataParallel`** scales training across many GPUs.
:::

## Additional references

::: refs
- [Appendix A code](https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-A) — GitHub · the author's complete, runnable notebooks and DDP scripts for this appendix.
- [PyTorch Tutorials](https://pytorch.org/tutorials/) — Docs · official beginner-to-advanced walkthroughs, including the 60-minute Blitz.
- [A Gentle Introduction to torch.autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — Tutorial · how PyTorch's automatic differentiation engine works, with diagrams.
- [torch.Tensor documentation](https://pytorch.org/docs/stable/tensors.html) — Docs · the full reference for tensor types and operations.
- [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) — Tutorial · building efficient input pipelines with `Dataset` and `DataLoader`.
- [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) — Tutorial · the recommended `state_dict` workflow and its alternatives.
- [Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) — Tutorial · the official guide to multi-GPU training with DDP.
:::

## Test your knowledge

```flashcards
Q: What are the three core components of PyTorch?
A: A **tensor library** (NumPy-like, with GPU support), an **automatic differentiation engine** (autograd), and a **deep-learning library** of layers, losses, and optimizers.
---
Q: What is the "rank" of a tensor?
A: Its number of dimensions. A scalar is rank 0, a vector rank 1, a matrix rank 2, a 3-D tensor rank 3, and so on.
---
Q: Why does PyTorch default float tensors to float32 instead of float64?
A: float32 gives enough precision for deep learning while using less memory and running faster — and GPUs are optimized for 32-bit math.
---
Q: What does calling loss.backward() do?
A: It walks the computation graph backward (the chain rule) and stores the gradient of the loss with respect to each trainable parameter in that parameter's .grad attribute.
---
Q: When subclassing nn.Module, what goes in __init__ versus forward?
A: __init__ declares the layers; forward defines how input data flows through them. You rarely write backward — autograd derives it from forward.
---
Q: What is the difference between a Dataset and a DataLoader?
A: A Dataset knows how to return one example by index (via __getitem__/__len__); a DataLoader wraps it to handle batching, shuffling, and parallel loading.
---
Q: What three steps form the core of each training iteration, and in what order?
A: optimizer.zero_grad() (clear old gradients), loss.backward() (compute new ones), optimizer.step() (update the weights).
---
Q: How do you move a model and its data to a GPU, and what's the one rule?
A: Call .to(device) on both the model and each batch. The rule: all tensors in a computation must be on the same device.
```

```quiz
1. Which method computes gradients for all trainable parameters at once?
   - ( ) optimizer.step()
   - (x) loss.backward()
   - ( ) torch.no_grad()
   - ( ) model.eval()
   > backward() walks the computation graph in reverse and fills each parameter's .grad. step() then uses those gradients to update the weights.

2. Why must you call optimizer.zero_grad() each iteration?
   - ( ) To move tensors to the GPU
   - ( ) To switch the model into evaluation mode
   - (x) Because PyTorch accumulates gradients by default, so old ones must be cleared
   - ( ) To reshape the input batch
   > Gradients add up across backward() calls unless reset, which would corrupt each update.

3. What does a custom PyTorch Dataset class need to implement?
   - ( ) only forward()
   - (x) __init__, __getitem__, and __len__
   - ( ) train() and eval()
   - ( ) save() and load()
   > __getitem__ returns one example by index and __len__ reports the total; the DataLoader uses both.

4. What is the recommended way to save a trained model in PyTorch?
   - ( ) Pickle the entire model object
   - (x) Save model.state_dict(), then load it into a freshly built model of the same architecture
   - ( ) Save only the optimizer
   - ( ) Copy the .py source file
   > The state_dict holds just the parameters; it's portable but requires recreating the matching architecture before loading.

5. You get "RuntimeError: Expected all tensors to be on the same device." What's the cause?
   - ( ) The learning rate is too high
   - ( ) You forgot to call model.train()
   - (x) Some tensors are on the CPU and others on the GPU
   - ( ) The batch size doesn't divide the dataset evenly
   > Every tensor in a computation must live on the same device; move the model and data together with .to(device).
```

```assignment "Train the toy network end to end" level=beginner
Reproduce the full A.5–A.7 workflow yourself. Define the `NeuralNetwork` class (two hidden layers via `nn.Sequential`), build a `ToyDataset` and a shuffled `DataLoader` from the five-example toy data in Listing A.5, then write a training loop that runs `forward → cross_entropy → zero_grad → backward → step` for 3 epochs with an SGD optimizer (`lr=0.5`). Print the loss each batch and confirm it falls toward 0.

Hint: instantiate the model with `NeuralNetwork(num_inputs=2, num_outputs=2)` — two features in, two classes out.
Hint: get class predictions from logits with `torch.argmax(logits, dim=1)`, and count correct ones with `torch.sum(predictions == labels)`.
Hint: wrap the final prediction in `model.eval()` and `with torch.no_grad():` to skip gradient tracking.
```
