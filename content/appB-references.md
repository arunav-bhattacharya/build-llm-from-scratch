Every chapter of *Build a Large Language Model (From Scratch)* ends with a short list of papers, articles, and code repositories the author drew on. This appendix gathers them all into one browsable place — a **master reading list** for going deeper on any topic the book touches.

To make it genuinely useful as a study companion, the book's own citations are blended with the **community explainers** referenced throughout this site — 3Blue1Brown's visual lessons, Jay Alammar's illustrated guides, Andrej Karpathy's from-scratch videos, the Hugging Face course, and Sebastian Raschka's own blog posts. Each entry is tagged by **Type** (Paper · Blog · Video · Code · Docs · Dataset · Book) with a one-line note on why it's worth your time.

::: callout note "How to use this page"
This is a curated *menu*, not a checklist — you don't need to read everything. When a concept in a chapter intrigues you, jump to that chapter's section here and pick the resource whose **Type** matches how you like to learn: a **Paper** for the primary source, a **Video** or **Blog** for an intuitive walkthrough, **Code** to see it implemented. Every arXiv ID and URL below was transcribed directly from the book or verified against the original source.
:::

::: objectives "What you'll find here"
- The **foundational papers** behind transformers, GPT, and modern LLMs — *Attention Is All You Need*, BERT, GPT-3, InstructGPT, and more
- Per-chapter **further reading** matching the structure of the book
- The best **free community resources** — visual explainers, lecture videos, and code repos — for each topic
- Pointers to **datasets**, alternative architectures, and advanced techniques for when you want to go beyond the book
:::

## Chapter 1 — Understanding LLMs

The big-picture chapter: what LLMs are, the transformer that powers them, and the GPT lineage. These references cover the architectures that started it all and a few that challenge the transformer's dominance.

::: refs
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Paper · Vaswani et al., 2017 · the original transformer architecture; the single most important paper to read.
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Paper · Devlin et al., 2018 · the original encoder-style transformer.
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Paper · Brown et al., 2020 · the GPT-3 paper; the decoder-style template this book builds toward.
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) — Paper · Ouyang et al., 2022 · the InstructGPT paper behind instruction fine-tuning (revisited in chapter 7).
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) — Paper · Dosovitskiy et al., 2020 · the Vision Transformer; transformers aren't just for text.
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — Paper · Touvron et al., 2023 · a popular, openly available GPT-like model.
- [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564) — Paper · Wu et al., 2023 · a custom-built LLM that beats general models on finance tasks.
- [Towards Expert-Level Medical Question Answering with LLMs](https://arxiv.org/abs/2305.09617) — Paper · Singhal et al., 2023 · fine-tuning LLMs to outperform general models in medicine.
- [The Pile: An 800GB Dataset of Diverse Text](https://arxiv.org/abs/2101.00027) — Dataset · Gao et al., 2020 · EleutherAI's publicly available pretraining corpus.
- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) — Paper · Peng et al., 2023 · a non-transformer LLM architecture.
- [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866) — Paper · Poli et al., 2023 · another transformer alternative.
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Paper · Gu and Dao, 2023 · the state-space model challenging transformers.
- [But what is a GPT? Visual intro to transformers](https://www.3blue1brown.com/lessons/gpt) — Video · 3Blue1Brown's gorgeous big-picture explainer.
- [Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models) — Blog · the author's own survey of the LLM landscape.
- [The Hugging Face LLM Course](https://huggingface.co/learn/llm-course/) — Course · a free, hands-on tour of modern NLP and LLMs.
:::

## Chapter 2 — Working with Text Data

Turning text into numbers: tokenization, byte-pair encoding, and embeddings. These references dig into the tokenizers used by real LLMs.

::: refs
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — Paper · Sennrich et al., 2015 · the original byte-pair encoding (BPE) for tokenization.
- [SentencePiece: A Simple and Language-Independent Subword Tokenizer](https://aclanthology.org/D18-2012/) — Paper · Kudo and Richardson, 2018 · an alternative tokenization scheme used by many LLMs.
- [Fast WordPiece Tokenization](https://arxiv.org/abs/2012.15524) — Paper · Song et al., 2020 · the WordPiece tokenizer (used by BERT), made efficient.
- [GPT-2 BPE encoder source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) — Code · OpenAI's open-sourced byte-pair encoder for GPT-2.
- [OpenAI Tokenizer playground](https://platform.openai.com/tokenizer) — Tool · an interactive web UI showing how GPT's BPE splits text.
- [minbpe: A Minimal BPE Tokenizer](https://github.com/karpathy/minbpe) — Code · Andrej Karpathy's minimal, readable BPE implementation to build from scratch.
- [Machine Learning Q and AI](https://leanpub.com/machine-learning-q-and-ai) — Book · Sebastian Raschka, 2023 · more on embedding spaces and vector representations.
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — Video · Karpathy walks through BPE tokenization end to end.
:::

## Chapter 3 — Coding Attention Mechanisms

Self-attention from first principles. These references trace attention from its RNN origins through scaled dot-product attention to the efficient implementations used in production.

::: refs
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Paper · Bahdanau, Cho, and Bengio, 2014 · the original (Bahdanau) attention for RNNs.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Paper · Vaswani et al., 2017 · introduced scaled dot-product and multi-head attention.
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Paper · Dao et al., 2022 · an IO-aware attention implementation; same math, far faster.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) — Paper · Dao, 2023 · better parallelism and work partitioning for attention.
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) — Paper · Srivastava et al., 2014 · the regularization technique applied to attention weights.
- [Simplifying Transformer Blocks](https://arxiv.org/abs/2311.01906) — Paper · He and Hofmann, 2023 · good performance even without the value matrix and projection layer.
- [PyTorch `scaled_dot_product_attention` docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — Docs · PyTorch's built-in attention function (supports FlashAttention).
- [PyTorch `MultiheadAttention` docs](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) — Docs · PyTorch's efficient multi-head attention module.
- [Attention in transformers, step by step](https://www.3blue1brown.com/lessons/attention/) — Video · 3Blue1Brown's stunning visual breakdown of Q/K/V attention.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Blog · Jay Alammar's canonical visual explanation of multi-head attention.
- [Understanding and Coding Self-Attention from Scratch](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) — Blog · the author's deep dive, mirroring this chapter.
- [Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Video · Andrej Karpathy codes attention live.
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — Blog · Lilian Weng's thorough survey of attention variants.
:::

## Chapter 4 — Implementing a GPT Model from Scratch

Assembling the full GPT: layer normalization, GELU feed-forward, shortcut connections, and transformer blocks. These references cover the building blocks and their modern variants.

::: refs
- [Layer Normalization](https://arxiv.org/abs/1607.06450) — Paper · Ba, Kiros, and Hinton, 2016 · the normalization technique used inside each transformer block.
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Paper · Xiong et al., 2020 · Pre-LayerNorm (as in GPT-2) vs. Post-LayerNorm and why placement matters.
- [ResiDual: Transformer with Dual Residual Connections](https://arxiv.org/abs/2304.14802) — Paper · Tie et al., 2023 · combining the benefits of both LayerNorm placements.
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — Paper · Zhang and Sennrich, 2019 · RMSNorm, the efficient LayerNorm variant in many modern LLMs.
- [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) — Paper · Hendrycks and Gimpel, 2016 · the smooth activation function used in GPT's feed-forward layers.
- [Language Models Are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Paper · Radford et al., 2019 · the GPT-2 paper (124M to 1.5B parameters).
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Paper · Brown et al., 2020 · GPT-3; same architecture as GPT-2, 100x larger.
- [OpenAI's GPT-3: A Technical Overview](https://lambdalabs.com/blog/demystifying-gpt-3) — Blog · Lambda Labs · estimates training GPT-3 on one consumer GPU would take 665 years.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Code · Karpathy's minimalist, efficient GPT-2 implementation.
- [In the Long (Context) Run](https://www.harmdevries.com/post/context-length/) — Blog · Harm de Vries · why feed-forward layers dominate compute below ~32k-token contexts.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Blog · Jay Alammar's visual tour of the GPT-2 architecture.
:::

## Chapter 5 — Pretraining on Unlabeled Data

The training loop, loss, evaluation, decoding strategies, and loading GPT-2's weights. These references cover loss functions, pretraining datasets, and sampling methods.

::: refs
- [Pythia: A Suite for Analyzing LLMs Across Training and Scaling](https://arxiv.org/abs/2304.01373) — Paper · Biderman et al., 2023 · dataset, hyperparameter, and architecture details for pretraining.
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838) — Paper · Groeneveld et al., 2024 · a fully open pretraining recipe.
- [Simple and Scalable Strategies to Continually Pre-train LLMs](https://arxiv.org/abs/2403.08763) — Paper · Ibrahim et al., 2024 · warmup and cosine annealing applied to continued pretraining.
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507) — Paper · Zhao et al., 2024 · cut pretraining memory by swapping the optimizer.
- [GaLore code repository](https://github.com/jiaweizzhao/GaLore) — Code · the `galore-torch` package implementing the above.
- [Dolma: An Open Corpus of Three Trillion Tokens](https://arxiv.org/abs/2402.00159) — Dataset · Soldaini et al., 2024 · a large open pretraining corpus.
- [The Pile: An 800GB Dataset of Diverse Text](https://arxiv.org/abs/2101.00027) — Dataset · Gao et al., 2020 · EleutherAI's diverse text corpus.
- [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) — Dataset · Penedo et al., 2023 · curated web-only pretraining data.
- [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833) — Paper · Fan et al., 2018 · the paper that introduced top-k sampling.
- [Diverse Beam Search](https://arxiv.org/abs/1610.02424) — Paper · Vijayakumar et al., 2016 · an alternative decoding algorithm to sampling.
- [Top-p (nucleus) sampling](https://en.wikipedia.org/wiki/Top-p_sampling) — Wiki · the cumulative-probability alternative to top-k.
- [The Illustrated GPT-2: Visualizing Transformer Language Models](https://jalammar.github.io/illustrated-gpt2/) — Blog · Jay Alammar · how GPT-2 generates text token by token.
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — Video · Karpathy pretrains GPT-2 from scratch, end to end.
:::

## Chapter 6 — Fine-Tuning for Classification

Adapting a pretrained LLM into a classifier. These references cover fine-tuning strategies, class imbalance, and encoder-style alternatives.

::: refs
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Paper · Devlin et al., 2018 · encoder-based models can be effective for classification.
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — Paper · Liu et al., 2019 · a stronger BERT for classification tasks.
- [Label Supervised LLaMA Finetuning](https://arxiv.org/abs/2310.01208) — Paper · Li et al., 2023 · removing the causal mask can improve classification fine-tuning.
- [LLM2Vec: LLMs Are Secretly Powerful Text Encoders](https://arxiv.org/abs/2404.05961) — Paper · BehnamGhader et al., 2024 · turning decoder LLMs into strong encoders.
- [Finetuning Large Language Models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) — Blog · the author on which layers to fine-tune and why the last block helps.
- [Understanding Parameter-Efficient Finetuning of Large Language Models](https://magazine.sebastianraschka.com/p/understanding-parameter-efficient) — Blog · a survey of fine-tuning approaches.
- [imbalanced-learn User Guide](https://imbalanced-learn.org/stable/user_guide.html) — Docs · dealing with imbalanced classification datasets.
:::

## Chapter 7 — Fine-Tuning to Follow Instructions

Instruction fine-tuning end to end, plus LLM-based evaluation. These references cover instruction datasets, prompt formats, masking strategies, and automated evaluation.

::: refs
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) — Paper · Ouyang et al., 2022 · InstructGPT, the foundation of instruction tuning.
- [Stanford Alpaca: An Instruction-Following Llama Model](https://github.com/tatsu-lab/stanford_alpaca) — Code · the 52,000-pair dataset used widely for instruction tuning.
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Paper · Zhou et al., 2023 · a small, high-quality instruction dataset rivals large ones.
- [UltraChat / Enhancing Chat LMs by Scaling Instructional Conversations](https://arxiv.org/abs/2305.14233) — Paper · Ding et al., 2023 · an 805,000-pair instruction dataset.
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) — Paper · Abdin et al., 2024 · a 3.8B model rivaling far larger ones (its prompt format appears in exercise 7.1).
- [Magpie: Alignment Data Synthesis from Scratch](https://arxiv.org/abs/2406.08464) — Paper · Xu et al., 2024 · generating instruction data by prompting aligned LLMs.
- [Instruction Tuning with Loss Over Instructions](https://arxiv.org/abs/2405.14394) — Paper · Shi et al., 2024 · when *not* masking instructions helps (relevant to exercise 7.2).
- [Prometheus: Inducing Fine-grained Evaluation Capability](https://arxiv.org/abs/2310.08491) — Paper · Kim et al., 2023 · an open LLM judge for long-form responses.
- [Prometheus 2](https://arxiv.org/abs/2405.01535) — Paper · Kim et al., 2024 · the improved open evaluator model.
- [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/abs/2405.05904) — Paper · Gekhman et al., 2024 · fine-tuning on new facts can increase hallucination.
- [LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) — Blog · the author on preference fine-tuning after instruction tuning.
- [LIMA dataset on Hugging Face](https://huggingface.co/datasets/GAIR/lima) — Dataset · the LIMA instruction-tuning data.
:::

## Appendix A — Introduction to PyTorch

Deep-learning and PyTorch fundamentals. These references are the author's recommended primers for going deeper on tensors, training, and multi-GPU setups.

::: refs
- [Machine Learning with PyTorch and Scikit-Learn](https://www.manning.com/books/machine-learning-with-pytorch-and-scikit-learn) — Book · Raschka, Liu, and Mirjalili, 2022 · a comprehensive deep-learning introduction (ISBN 978-1801819312).
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Book · Stevens, Antiga, and Viehmann, 2021 · a hands-on PyTorch primer (ISBN 978-1617295263).
- [Model Evaluation, Model Selection, and Algorithm Selection in ML](https://arxiv.org/abs/1811.12808) — Paper · Raschka, 2018 · a thorough guide to evaluating ML models.
- [Introducing PyTorch Fully Sharded Data Parallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) — Blog · scaling beyond DDP when a model won't fit on one GPU.
- [PyTorch official tutorials](https://pytorch.org/tutorials/) — Docs · the canonical starting point for learning PyTorch.
:::

## Key takeaways

::: takeaways
- **Start with the primary source for the big ideas.** *Attention Is All You Need* (transformers), the GPT-2 and GPT-3 papers (the architecture this book builds), and InstructGPT (fine-tuning) are the load-bearing papers — read them once you've finished the matching chapter.
- **Match the resource to your learning style.** For intuition, reach for a **Video** (3Blue1Brown, Karpathy) or an **illustrated Blog** (Jay Alammar); for depth, the **Paper**; to internalize it, the **Code**.
- **The author's own blog** (*Ahead of AI* / sebastianraschka.com) closely mirrors the book and is the most natural next step after each chapter.
- **Go beyond transformers when you're ready** — RWKV, Hyena, and Mamba show the architecture isn't the only option, and FlashAttention shows how the same math gets made fast.
- **Keep learning by building.** The book's [GitHub repo](https://github.com/rasbt/LLMs-from-scratch) and Karpathy's nanoGPT are the best places to extend what you've built here.
:::

## Test your knowledge

```flashcards
Q: Which paper introduced the transformer architecture and scaled dot-product attention?
A: **"Attention Is All You Need"** (Vaswani et al., 2017), arXiv 1706.03762.
---
Q: Which paper describes the decoder-style model used as the template for the LLM built in this book?
A: **GPT-3** — "Language Models are Few-Shot Learners" (Brown et al., 2020), arXiv 2005.14165.
---
Q: Which paper is the foundation for instruction fine-tuning (chapter 7)?
A: **InstructGPT** — "Training Language Models to Follow Instructions with Human Feedback" (Ouyang et al., 2022), arXiv 2203.02155.
---
Q: Which technique makes attention faster without changing its math, and what paper introduced it?
A: **FlashAttention** (Dao et al., 2022, arXiv 2205.14135) — an IO-aware implementation that optimizes memory access.
---
Q: Which resources are best for a *visual* intuition of attention and the transformer?
A: **3Blue1Brown's** "Attention, step by step" video and **Jay Alammar's** "The Illustrated Transformer" blog post.
---
Q: Name two LLM architectures in the references that are *not* based on the transformer.
A: Any two of **RWKV**, **Hyena**, and **Mamba** — alternatives explored in the chapter 1 references.
```
