---
course: "introduction-to-transformer-models"
chapter: "self-attention-multi-head-attention"
lesson: "self-attention-idea"
sourceId: 2969
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-2-self-attention-multi-head-attention/self-attention-idea"
title: "自注意力的原理"
description: "说明自注意力如何让输入能够彼此互动，并在同一序列内确定焦点。"
order: 1
plots: []
sourceHash: "a669e4ad25707ad2969fed40b199b1621a05b7954d1b12a5ce3e62b08596929e"
sourceCorrections: []
---

注意力是一种机制，使模型，特别是序列到序列模型，在生成输出时能够关注输入的相关部分。这种方法有助于克服传统循环神经网络 (neural network)（RNN）的一些局限，例如难以捕捉长距离依赖性以及其固有的序列特性，这阻碍了并行处理。

现在，我们将注意力转向*内部*。设想处理一个句子，例如：

“The animal didn't cross the street because **it** was too tired.”

为了弄明白“**它**”指代什么，我们自然地回看前面的词。“它”是“animal”还是“street”？显然，“它”指代“animal”。模型如何学习这种关联？RNNs试图通过其隐藏状态顺序传递信息来捕捉这种语境，但这在较长距离上会变得困难。

此时，**自注意力 (self-attention)**就派上用场了。它是一种机制，让模型在为特定词生成表示时，能够衡量*同一输入序列内*不同词的重要性。不像RNN只看紧邻前一个隐藏状态，自注意力让模型能够同时查看整个序列，并为每个词询问：“在这个序列中，哪些其他词对于理解*这个*词最相关？”

可以将其视为允许输入序列中的每个词与所有其他词（包括它自身）直接进行互动。模型会计算每对词之间的“注意力分数”。词A和词B之间的高分数表示词B对于理解或表示词A非常相关。

考虑句子“The cat sat on the mat”。在处理“sat”这个词时，自注意力可能会认为“cat”和“mat”尤其相关，相比于“the”或“on”，会赋予它们更高的注意力分数。

> 自注意力聚焦于“sat”这个词的图示。箭头表示注意力流向，潜在分数表明了相关性（例如，“cat”和“mat”与“sat”高度相关）。

这与我们在基本的序列到序列模型中看到的注意力不同，在那里解码器通常关注*编码器的输出*。自注意力发生在层*内部*，查看*同*一层（或其下一层）的表示。

为此，每个输入词表示（最初是它的词嵌入 (embedding)）扮演三个角色，我们将在下一节进行数学上的阐述：

1. **查询（Query）：** 代表当前词在询问：“我应该关注哪些信息？”
2. **键（Key）：** 代表所有词（包括当前词）在说：“这是我所代表的信息类型。”
3. **值（Value）：** 代表所有词在说：“如果发现我相关（基于你的查询与我的特定键匹配），这是我贡献的实际信息/表示。”

模型学习如何将输入词嵌入转化为这些查询、键和值表示。被处理词的查询会与序列中所有词的键进行比较。这些比较的结果（注意力分数）随后被用来创建值表示的加权和。这个加权和成为原始词的新的、考虑语境的表示的一部分。

自注意力的强大之处在于它能够直接建模序列中任意两个词之间的关系，不论它们之间的距离。这比序列RNN处理更有效地解决了长距离依赖问题。此外，由于每个词的计算主要涉及基于*所有*词的矩阵乘法，它们可以并行执行，从而显著加快了相对于RNN的训练速度。

我们已经掌握了核心原理：自注意力让序列中的词能够直接评估其他词的相关性，以优化它们自身的表示。接下来，我们将考察Transformers中用于计算这些注意力分数的具体机制：缩放点积注意力。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构和自注意力机制，为许多现代大型语言模型奠定了基础。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2023)
  Publisher: Stanford University; Pages: Chapter 10: Transformers and Pretrained Language Models (3rd edition draft)
  一本内容全面的教科书，详细解释了自然语言处理中的注意力机制和Transformer模型。
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar (2018)
  对Transformer模型进行了高度可视化和直观的说明，特别有助于理解自注意力机制。
- [Stanford CS224N: Natural Language Processing with Deep Learning - Transformers and Self-Attention](https://web.stanford.edu/class/cs224n/slides/cs224n-2024-lecture08-transformers.pdf) — Christopher Manning, Abigail See, and Course Staff (2024)
  Publisher: Stanford University
  来自顶尖大学课程的讲义，提供了关于Transformer架构和自注意力机制的学术视角。
