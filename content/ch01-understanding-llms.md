Large language models (LLMs) like the ones behind ChatGPT, Gemini, and Claude can write emails, summarize documents, translate languages, and even program computers. A decade ago, this felt like science fiction. Today, you can build a small version of one yourself — and by the end of this book, you will.

This first chapter is the map before the journey. We won't write code yet. Instead, we'll build a rock-solid mental model of **what an LLM actually is**, **why the transformer architecture changed everything**, and **the three-stage plan** we'll follow to construct a GPT-style model from the ground up.

::: objectives "What you'll learn"
- A plain-English definition of large language models and where they sit in the AI landscape
- What the transformer architecture is, and how the encoder/decoder and self-attention fit together
- The difference between **pretraining** and **fine-tuning**, and why both matter
- How GPT models work at a high level — next-word prediction, autoregression, and emergent abilities
- The three-stage roadmap this book follows to build an LLM from scratch
:::

## What is an LLM?

An **LLM is a neural network trained to understand and generate human-like text.** That's the whole idea in one sentence. The word **"large"** does double duty: these models are large in their number of **parameters** (the adjustable internal numbers the network learns — often tens or hundreds of *billions*), and large in the amount of **text data** they're trained on (sometimes much of the public internet).

When we say a model "understands" language, we mean something specific and modest: it can process and produce text that is coherent and contextually appropriate. It is **not** conscious and does not comprehend the way a person does. It is, at its core, an extraordinarily capable pattern machine.

::: callout analogy "The 'large' in large language model"
Think of parameters as the **knobs on an impossibly complex mixing board**. A simple model classifying Iris flowers might have *two* knobs. GPT-3 has **175 billion**. Training is the slow process of turning every knob a tiny bit until the whole board produces sensible music — in this case, sensible text. More knobs (and more music to learn from) means the model can capture far subtler patterns of language.
:::

What does the model actually learn to do? Something deceptively simple: **predict the next word.** Given "The cat sat on the…", a good model assigns high probability to "mat" and low probability to "helicopter." This is called **next-word prediction**, and it's powerful precisely because language is sequential — to predict the next word well, the model is forced to learn grammar, facts, reasoning patterns, and context. It's surprising to many researchers that such a simple objective produces such capable models.

### Where LLMs sit in the AI family

LLMs didn't appear from nowhere. They're a specific application of **deep learning**, which is a branch of **machine learning**, which is a sub-field of **artificial intelligence**. Each is nested inside the last.

::: diagram ch01-ai-ml-dl-llm "Artificial intelligence contains machine learning, which contains deep learning, which is where LLMs live. Generative AI is the overlapping region that creates new content."
:::

::: callout analogy "Russian nesting dolls"
The relationship is like a set of **matryoshka dolls**. **AI** is the biggest doll — any system that performs tasks needing human-like intelligence. Open it and you find **machine learning**: programs that learn rules *from data* instead of being hand-coded. Inside that is **deep learning**: machine learning using multi-layer neural networks. And tucked inside is the **LLM** — a deep network specialized for language. **Generative AI** is the label for any of these that *creates* new content (text, images, audio).
:::

The leap from traditional machine learning to deep learning is worth pausing on. Imagine building a spam filter:

- **Traditional machine learning** needs a human expert to hand-pick **features** first — "count exclamation marks," "flag the words *prize*, *win*, *free*," "detect ALL-CAPS," "spot suspicious links." The model then learns from those human-designed signals.
- **Deep learning** skips the manual feature engineering. You feed it raw labeled examples (spam / not-spam) and the network *discovers* the useful patterns by itself.

(Both still need someone to provide the *labels* — that distinction becomes important in a moment.) While modern AI is dominated by machine learning and deep learning, the broader field also includes rule-based systems, expert systems, genetic algorithms, fuzzy logic, and symbolic reasoning.

## What can LLMs do?

Because they parse and generate unstructured text so well, LLMs are remarkably general-purpose. Common applications include:

- **Machine translation** between languages
- **Text generation** — fiction, articles, and computer code
- **Sentiment analysis** and **text classification**
- **Summarization** of long documents
- **Chatbots and virtual assistants** (ChatGPT, Gemini, Claude)
- **Knowledge retrieval** and question answering over medical, legal, or technical material

The headline difference from older NLP systems is **breadth**. Earlier models were each built for one narrow task — a translation model could *only* translate. A single modern LLM handles all of the above without being rebuilt for each one.

## How LLMs are built and used

Why would anyone build their *own* LLM when capable ones already exist? Several solid reasons:

- **Performance:** custom models tuned on domain data (legal, medical, finance) often beat general-purpose ones on those specific tasks.
- **Privacy & control:** running your own model keeps sensitive data in-house instead of sending it to a third party.
- **Latency & cost:** a local model can reduce response time and ongoing server bills.
- **Autonomy:** you control updates and behavior instead of depending on a vendor.

Creating an LLM happens in **two big phases**: **pretraining**, then **fine-tuning**.

::: diagram ch01-pretrain-finetune "An LLM is first pretrained on a huge pile of raw, unlabeled text to become a general-purpose 'foundation model,' then fine-tuned on a smaller labeled dataset for a specific job."
:::

**Pretraining** is the initial phase: the model trains on a massive, diverse pile of **raw, unlabeled text** — internet pages, books, Wikipedia, research articles — to develop a broad, general understanding of language. ("Raw" means plain text with no labels attached; some light filtering removes junk formatting or unknown languages.) The result is a **foundation model** (also called a **base model**), like GPT-3. A foundation model can already complete half-written sentences and has limited **few-shot** ability (learning a new task from just a handful of examples).

::: callout note "Self-supervised learning — the answer key is already in the text"
Traditional supervised learning needs humans to label every example. Pretraining doesn't — it uses **self-supervised learning**, where the model creates its *own* labels from the data. The trick: the "label" for any position is simply **the next word that actually comes next** in the text. Because the correct answer is already sitting there in the sentence, we can generate near-infinite training examples from unlabeled text "on the fly." This is what makes training on trillions of words possible.
:::

**Fine-tuning** comes second: you take the pretrained foundation model and train it further on a smaller, **labeled** dataset specific to your goal. The two most common kinds are:

- **Instruction fine-tuning** — the labels are *instruction → answer* pairs (e.g., "Translate this to French" paired with the correct translation). This is how a base model becomes a helpful assistant.
- **Classification fine-tuning** — the labels are *text → category* (e.g., emails tagged "spam" or "not spam").

::: callout analogy "School, then a specialty"
**Pretraining is like a broad education** — years of reading everything, building general literacy and world knowledge. **Fine-tuning is vocational training** — a short, focused course that turns that well-rounded graduate into a paralegal, a translator, or a customer-support agent. The expensive part (the general education) is done once; the specialization is comparatively cheap.
:::

## The transformer architecture

Almost every modern LLM is built on the **transformer**, a neural-network architecture introduced in the landmark 2017 paper *"Attention Is All You Need."* It was originally designed for **machine translation** (English → German/French). To understand LLMs, it helps to understand this original design.

A transformer has **two submodules**:

- An **encoder** reads the input text and compresses its meaning into a series of numerical vectors (rich numerical representations called **embeddings**) that capture context.
- A **decoder** takes those vectors and generates the output text, one word at a time.

::: diagram ch01-transformer "The original transformer translates text: the encoder reads the full input and turns it into context-rich vectors; the decoder uses those vectors to produce the translation one word at a time."
:::

The secret ingredient inside both submodules is the **self-attention mechanism**. Self-attention lets the model weigh how important every word is *relative to every other word* when forming its understanding — capturing long-range relationships that older models struggled with. We'll defer the mechanics to Chapter 3 (it's the heart of this book), but the intuition matters now.

::: callout analogy "Attention is knowing what 'it' refers to"
Read: *"The trophy didn't fit in the suitcase because **it** was too big."* What is "it" — the trophy or the suitcase? You resolved that instantly by paying **attention** to the relevant words. Self-attention gives the model the same ability: for every word, it decides which other words to focus on. Like a guest at a noisy dinner party tuning in to the one conversation that matters, the model learns where to look.
:::

### Two branches of the family: BERT vs GPT

After the original transformer, two influential variants emerged by keeping different halves of it:

::: diagram ch01-bert-vs-gpt "BERT uses the encoder and fills in masked (hidden) words using context from both sides. GPT uses the decoder and generates text left-to-right, one word at a time."
:::

- **BERT** (Bidirectional Encoder Representations from Transformers) is built on the **encoder**. It's trained on **masked word prediction** — random words are hidden and the model fills them in using context from *both* directions. This makes BERT excellent at **classification** tasks like sentiment analysis. (X/Twitter has used BERT to detect toxic content.)
- **GPT** (Generative Pretrained Transformer) is built on the **decoder**. It generates text **left-to-right, one word at a time**, which makes it superb at *generative* tasks: completion, summarization, translation, writing code.

::: callout analogy "The fill-in-the-blank student vs the improv storyteller"
**BERT is a fill-in-the-blank test-taker.** It sees the entire sentence with a few words blanked out and uses everything around the gap to guess the missing words. **GPT is an improv storyteller.** It can only see what came before and must invent the next word, then the next, building the story forward. This book builds a **GPT-style** model.
:::

GPT models are strikingly flexible at solving tasks straight from the prompt, with no retraining:

- **Zero-shot learning** — performing a brand-new task with *no* examples ("Translate this to German:").
- **Few-shot learning** — performing a task after seeing just a *few* examples in the prompt.

::: callout note "Transformers vs LLMs — not the same thing"
The terms get used interchangeably, but precisely: **not all transformers are LLMs** (transformers also power computer-vision models), and **not all LLMs are transformers** (some use recurrent or convolutional architectures, mainly chasing efficiency). In this book, "LLM" means a transformer-based, GPT-like model.
:::

## Why data matters: training on (almost) the whole internet

The capabilities of GPT- and BERT-like models come from the **scale and diversity** of their training data — billions of words spanning countless topics, plus natural and programming languages. Here's the actual recipe used to pretrain **GPT-3**, the base model behind the first ChatGPT:

| Dataset | What it is | Tokens | Share of training |
|---|---|---:|---:|
| CommonCrawl (filtered) | Web crawl data | 410 billion | 60% |
| WebText2 | Web crawl data | 19 billion | 22% |
| Books1 | Internet book corpus | 12 billion | 8% |
| Books2 | Internet book corpus | 55 billion | 8% |
| Wikipedia | High-quality text | 3 billion | 3% |

A **token** is the unit of text a model reads — roughly a word or a piece of a word (Chapter 2 is all about tokenization). Note the totals don't quite add up the way you'd expect: the subsets sum to ~499 billion tokens, but GPT-3 was actually trained on only ~300 billion (the authors never said why). The web-crawl portion alone (CommonCrawl) needs about **570 GB** of storage. Later models like Meta's LLaMA added sources such as Arxiv research papers and StackExchange code Q&As.

::: callout warning "This scale is expensive"
Pretraining GPT-3 from scratch is estimated to have cost around **$4.6 million** in cloud-computing credits. Don't worry — in this book we'll pretrain a *small* model on a *tiny* dataset purely for learning (it runs on an ordinary laptop), and then we'll **load openly available pretrained weights** so we can skip the multi-million-dollar step when we fine-tune.
:::

The good news: many capable pretrained LLMs are released as **open-source models** you can download, use, and fine-tune on modest datasets — getting strong results without the giant upfront cost.

## A closer look at GPT

GPT was introduced in the paper *"Improving Language Understanding by Generative Pre-Training"* (Radford et al., OpenAI). GPT-3 is simply a scaled-up version — more parameters, more data. The original ChatGPT was then created by **instruction-fine-tuning GPT-3** using the method from OpenAI's *InstructGPT* paper.

What's remarkable is that GPT models are pretrained on that one simple objective — **next-word prediction** — yet end up able to spell-check, classify, translate, and more.

::: diagram ch01-next-word "GPT is trained by repeatedly predicting the next word from the words before it. At generation time it feeds its own output back in as input, building text one word at a time."
:::

Because next-word prediction is **self-supervised** (the next word is its own label), GPT can learn from unlimited unlabeled text. Architecturally, GPT is *simpler* than the original transformer: it's essentially **just the decoder, with no encoder**.

::: callout key "Autoregressive generation"
GPT is an **autoregressive** model: each word it generates is appended to the input and fed back in to predict the *next* word. Like a writer composing a sentence word by word — every choice constrained by everything written so far — this feedback loop is what keeps generated text coherent.
:::

How big is "large"? The original transformer stacked its encoder/decoder block **6 times**. GPT-3 stacks **96 transformer layers** and totals **175 billion parameters**. GPT-3 dates from 2020 — ancient by AI standards — yet newer models like Meta's Llama use the *same core concepts* with only minor tweaks. That's why understanding GPT is still the right foundation.

::: callout analogy "Emergent behavior — a happy accident"
GPT was never explicitly taught to translate, yet it can. Abilities that appear without being directly trained are called **emergent behavior**. It's like learning to cook by following thousands of recipes and then discovering you can now *invent* a coherent menu — a skill nobody drilled into you, that emerged from broad exposure. These surprises are a hallmark of large-scale generative models.
:::

## The plan: building an LLM in three stages

Here's the blueprint for the rest of the book. We take the core idea behind GPT and build it in **three stages**:

::: diagram ch01-three-stages "Stage 1: implement the architecture and data pipeline (data prep, attention, the model). Stage 2: pretrain it into a foundation model and learn to evaluate it. Stage 3: fine-tune it into a classifier or a personal assistant."
:::

1. **Stage 1 — Build the engine (Chapters 2–4).** Learn the data-preprocessing steps, code the **attention mechanism** at the heart of every LLM, and assemble the full GPT **architecture**.
2. **Stage 2 — Pretrain it (Chapter 5).** Code the training loop, pretrain a GPT-like model that can generate text, and learn how to evaluate LLMs. We'll train a small model for learning, and also load real pretrained weights.
3. **Stage 3 — Fine-tune it (Chapters 6–7).** Take the pretrained model and fine-tune it — first as a **spam classifier**, then as an **instruction-following assistant**.

By the end, you'll have built — and deeply understood — every component of a working GPT-style LLM.

## Key takeaways

::: takeaways
- An **LLM** is a deep neural network that generates human-like text; "large" refers to both its billions of **parameters** and its enormous **training data**.
- LLMs are a kind of **deep learning**, which is a kind of **machine learning**, which is a kind of **AI**. **Generative AI** is any model that creates new content.
- The core training objective is **next-word prediction**, a form of **self-supervised learning** where the text supplies its own labels — enabling training on unlabeled data at massive scale.
- Modern LLMs are built on the **transformer** and its **self-attention** mechanism, which weighs the relevance of every token to every other token.
- The original transformer has an **encoder** (understands input) and a **decoder** (generates output). **BERT** uses the encoder (great for classification); **GPT** uses the decoder (great for generation).
- LLMs are built in two phases: **pretraining** on huge unlabeled text (→ a **foundation model**) and **fine-tuning** on smaller labeled data for specific tasks.
- GPT is **decoder-only** and **autoregressive**, generating one token at a time. Despite a simple objective, it shows **emergent behaviors** like translation.
- This book builds a GPT-style LLM in **three stages**: architecture & data → pretraining → fine-tuning.
:::

## Additional references

::: refs
- [Build a Large Language Model (From Scratch) — official code repository](https://github.com/rasbt/LLMs-from-scratch) — GitHub · all the book's code, notebooks, and bonus material (the companion to everything on this site).
- [Build an LLM from Scratch — chapter-by-chapter video series](https://www.youtube.com/playlist?list=PLQRyiBCWmqp5twpd8Izmaxu5XRkxd5yC-) — Video · Sebastian Raschka live-codes the book chapter by chapter.
- [But what is a GPT? Visual intro to transformers](https://www.3blue1brown.com/lessons/gpt/) — Video · 3Blue1Brown's gorgeous visual explanation; perfect for visual learners.
- [Intro to Large Language Models (1-hour talk)](https://www.youtube.com/watch?v=zjkBMFhNj_g) — Video · Andrej Karpathy's accessible, big-picture overview of how LLMs work.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Blog · Jay Alammar's classic visual walkthrough of the transformer architecture.
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — Paper · the original transformer paper that started it all.
- [Language Models are Few-Shot Learners (GPT-3, 2020)](https://arxiv.org/abs/2005.14165) — Paper · introduces GPT-3 and its zero-/few-shot abilities.
:::

## Test your knowledge

Flip through the flashcards to lock in the vocabulary, then take the quiz and try the writing challenge.

```flashcards
Q: What does the "large" in *large language model* refer to?
A: Both the model's huge number of **parameters** (learned weights) and the enormous **dataset** it's trained on.
---
Q: What single task are GPT models pretrained on?
A: **Next-word prediction** — predicting the next token given the previous ones.
---
Q: Why is next-word prediction called *self-supervised* learning?
A: The text supplies its own labels — the "correct answer" is just the word that actually comes next — so no human labeling is required.
---
Q: What are the two submodules of the original transformer, and what does each do?
A: An **encoder** that turns input text into context-rich vectors, and a **decoder** that generates output text one word at a time.
---
Q: How do BERT and GPT differ?
A: **BERT** uses the encoder and fills in masked words using both-sided context (good for classification). **GPT** uses the decoder and generates text left-to-right (good for generation).
---
Q: What is the difference between pretraining and fine-tuning?
A: **Pretraining** builds broad language ability from large *unlabeled* text (a foundation model). **Fine-tuning** specializes that model on smaller *labeled* data for a task.
---
Q: What does *autoregressive* mean?
A: The model feeds its own previous outputs back in as input to predict each next token, generating text one step at a time.
---
Q: What is *emergent behavior* in an LLM?
A: An ability the model was never explicitly trained for (e.g., translation) that arises naturally from large-scale training.
```

```quiz
1. Which statement best describes self-supervised learning as used in LLM pretraining?
   - ( ) Humans manually label every training example
   - (x) The model derives its labels from the structure of the text itself (the next word)
   - ( ) The model trains without any data
   - ( ) Two models supervise each other
   > In pretraining, the label for each position is simply the actual next word, so unlabeled text becomes self-labeling training data.

2. GPT is based on which part of the original transformer?
   - ( ) The encoder
   - (x) The decoder
   - ( ) Both encoder and decoder equally
   - ( ) Neither — it's a recurrent network
   > GPT is a decoder-only architecture designed for left-to-right text generation.

3. Roughly how many parameters does GPT-3 have?
   - ( ) 175 million
   - ( ) 1.75 billion
   - (x) 175 billion
   - ( ) 17.5 trillion
   > GPT-3 has about 175 billion parameters across 96 transformer layers.

4. Which task is BERT especially well suited for?
   - (x) Text classification (e.g., sentiment, toxicity detection)
   - ( ) Open-ended story generation
   - ( ) Generating code one token at a time
   - ( ) Real-time speech synthesis
   > BERT's bidirectional, masked-word training makes it strong at classification rather than generation.

5. What is the correct order of the two main training phases?
   - (x) Pretraining on large unlabeled text, then fine-tuning on smaller labeled data
   - ( ) Fine-tuning first, then pretraining
   - ( ) Pretraining only; fine-tuning is never used
   - ( ) They happen simultaneously on the same dataset
   > A model is first pretrained into a general foundation model, then fine-tuned for specific tasks.
```

```assignment "Explain an LLM to a curious friend" level=beginner
Write a short paragraph (5–7 sentences) explaining to a non-technical friend what a large language model is and how it learns. Your explanation must correctly use and define these four terms: **parameters**, **next-word prediction**, **pretraining**, and **fine-tuning**.

Hint: Lean on an analogy — the "broad education, then a specialty" framing works well for pretraining vs fine-tuning.
Hint: Avoid claiming the model "thinks" or "understands" like a human; describe it as recognizing and continuing patterns.
```
