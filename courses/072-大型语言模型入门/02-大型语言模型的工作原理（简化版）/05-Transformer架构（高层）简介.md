# Transformer架构（高层）简介

来源：[原文](https://apxml.com/zh/courses/intro-large-language-models/chapter-2-simplified-mechanics-of-llms/intro-transformer-architecture)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大型语言模型（LLM）根据前文（上下文 (context)）预测文本，并使用词元 (token)和嵌入 (embedding)表示词语。但是，模型如何有效运用该上下文，尤其当相关信息距离很远时？仅凭看最后几个词，通常不足以理解复杂的句子或段落。

这正是一种特定模型结构——**Transformer架构**——发挥作用之处。由谷歌研究人员在2017年一篇名为“Attention Is All You Need”的论文中提出，Transformer已成为此后开发出的许多强大大型语言模型赖以构建的核心。

### 顺序处理的问题

较早的语言模型方法通常严格按顺序逐词处理文本。想象一下，逐字阅读一篇长段落，并在读到末尾时完美记住第一句话。这很难！这些顺序模型难以连接相距较远但语义相关的词语。例如，如果名词在文本中出现得早得多，就可能难以理解代词所指代的名词是哪个。

### 核心思想：注意力机制 (attention mechanism)

Transformer架构引入了一种强大的机制，称为**注意力机制**，具体来说是**自注意力 (self-attention)机制**。模型不再严格地一个接一个地处理词语，注意力机制让模型在考虑任何单个词时，能衡量输入序列中*所有*词的重要性。

可以这样看：当你读到句子“The cat, which chased the mouse, quickly climbed **up** the tall tree”（那只追逐老鼠的猫，迅速爬上了高高的树），为了理解“up”这个词，你的大脑自然会不仅关注它前面的“climbed”，还会把它与“cat”和“tree”联系起来，以获得完整信息。注意力机制让模型能够以计算方式进行类似操作。它学习识别输入中哪些其他词语为理解当前词或预测下一个词提供了最有用的上下文 (context)。

这使得Transformer能够有效处理长距离依赖关系——指文本中相距较远的词语之间的关系。它帮助模型理解文本的精妙之处，解决代词指代，并比早期架构更好地把握整体上下文。

### 结构简化视图

尽管完整的Transformer架构包含多个组成部分，但为了高层理解，我们可以将其简化为两个主要部分：

1. **编码器（Encoder）：** 这部分读取输入文本。它利用自注意力 (self-attention)机制 (attention mechanism)，同时处理（或者说，以一种考虑所有词的方式）所有输入词，并为每个词构建丰富的表示（嵌入 (embedding)），这些表示融入了整个输入序列的上下文 (context)。
2. **解码器（Decoder）：** 这部分一次生成一个词元 (token)作为输出文本。它也使用自注意力机制来考虑已生成的词语。更重要的是，它*也*关注编码器生成的上下文表示。这确保了输出与输入提示相关，并在生成更多文本时保持连贯性。

> 一个简化流程图，展示编码器处理输入和解码器生成输出，并突出显示上下文信息的流动。

### 位置信息

你可能会想：如果模型使用注意力机制 (attention mechanism)同时查看所有词语，它如何得知词语的原始顺序呢？这通过**位置编码 (positional encoding)**来处理。本质上，表示每个词位置（第一、第二、第三等）的额外信息被添加到词的嵌入 (embedding)中。这确保了模型拥有序列顺序信息，即使在使用注意力机制时会衡量词语的重要性而不管其位置。

### Transformer为何表现出色

Transformer架构带来了显著的优势：

- **处理上下文 (context)：** 它的注意力机制 (attention mechanism)在捕捉词语之间的关系方面表现出色，即使这些词在文本中相距甚远。
- **并行处理：** Transformer内部的许多计算，特别是在注意力层中，可以并行执行。这使得在GPU（图形处理单元）和TPU（张量处理单元）等硬件上训练这些大型模型比严格的顺序模型效率高得多。

这种有效处理上下文并能在海量数据集上高效训练的能力，是基于Transformer的大型语言模型如此强大的主要原因。它们需要大量的训练数据，并拥有庞大的模型参数 (parameter)（$P$），正是因为它们需要学习人类语言所有精妙之处中的复杂注意力模式。这种架构提供了有效学习这些模式的能力。

理解注意力计算的细节或编码器和解码器内部的确切分层需要更进一步的学习。目前，主要观点是Transformer架构通过其注意力机制，使得大型语言模型在处理信息和生成输出时，能够智能地考虑输入文本不同部分的关联性。这是它们理解提示并产生连贯、与上下文相关的回复的根本。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构和自注意力机制的原始论文。
- [Stanford CS224N: Natural Language Processing with Deep Learning - Lecture on Attention and Transformers](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture08-transformers.pdf) — Tatsunori Hashimoto (2023)
  提供Transformer、自注意力及其NLP组件的学术解释的讲义。
- [The Transformer architecture - Hugging Face Course](https://huggingface.co/docs/transformers/en/model_anatomy/architecture) — Hugging Face (2024)
  对Transformer架构的清晰易懂的概述，解释了主要组件及其功能。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 12: Applications
  提供深度学习的理论基础，包括关于注意力机制的章节。

---

[上一节](04-%E7%90%86%E8%A7%A3%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0.md) · [下一节](06-%E8%AF%AD%E5%A2%83%E5%A6%82%E4%BD%95%E5%BD%B1%E5%93%8D%E7%94%9F%E6%88%90.md)
