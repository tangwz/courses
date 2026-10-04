---
course: "introduction-to-transformer-models"
chapter: "sequence-modeling-attention-fundamentals"
lesson: "sequence-to-sequence-challenges"
sourceId: 2953
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-1-sequence-modeling-attention-fundamentals/sequence-to-sequence-challenges"
title: "序列到序列任务的挑战"
description: "了解序列到序列任务中建模长距离依赖的难题。"
order: 1
plots: []
sourceHash: "2ddecae756d4515d2c90523d40cfab7e6814f85c71d10d974904eb45c9d366c6"
sourceCorrections: []
---

自然语言处理及其他许多方面的重要任务都涉及将一个序列转换为另一个序列。我们称这些为\*\*序列到序列（Seq2Seq）\*\*任务。以机器翻译为例：即将一种语言（例如，英语）的词语序列转换为另一种语言（例如，法语）的词语序列。其他例子包括：

- **文本摘要：** 将长文档（输入序列）映射为较短的摘要（输出序列）。
- **问答：** 获取上下文 (context)段落和问题（通常是连接起来或视为序列），并生成答案序列。
- **语音识别：** 将音频序列转换为文本序列。

以下是像翻译这样的通用序列到序列任务的视图：

> 这是一个序列到序列任务的简单示例，它通过一个模型将输入序列（"The"，"cat"，"sat"）映射到输出序列（"Le"，"chat"，"assis"）。

尽管这个概念看起来简单明了，但有效建模这些转换带来了几个重要的难题。

### 表示顺序和上下文 (context)

序列的含义通常很大程度上取决于其元素的顺序。“猫追狗”与“狗追猫”的含义完全不同。一个成功的模型不仅要理解单个元素（这里是词语），还要理解它们的位置和周围元素如何影响整体含义。它需要获取序列内的上下文关系。

### 处理长距离依赖

序列建模中最持续的困难之一是获取**长距离依赖**。这指的是在序列中理解或预测某个元素时，需要早先出现元素的信息的情况。

考虑以下示例：

> "I grew up in a small village in the south of **France**, near the Pyrenees. Although I moved away many years ago, I still visit often. As a result, I speak fluent **French**."

为了正确预测末尾的“French”，模型需要将其与几句前提到的“France”联系起来。如果中间文本更长，这种联系会更难保持。模型需要机制来“记住”或获取序列内可能较长距离的相关信息，避免这些信息随时间或位置的稀释或丢失。传统方法在这方面常常遇到问题，因为它们的“记忆”可能有限。

### 固定上下文 (context)瓶颈

早期处理序列到序列任务的方法通常涉及将整个输入序列总结为一个单一的、固定大小的向量 (vector)表示（通常称为“上下文向量”或“思维向量”）。然后，这个向量被期望包含来自输入序列的所有必要信息，以便模型开始生成输出序列。

想象一下尝试将整个章节总结成一个短句。你将不可避免地丢失很多细节。类似地，将一个复杂的输入序列，尤其是一个长序列，强制压缩到一个固定大小的向量中，会产生**信息瓶颈**。模型很难编码所有重要细节，导致性能下降，尤其是在更长或更复杂的序列上。模型可能会忘记输入的早期部分，或者未能获取细微的关系。

### 可变输入和输出长度

序列到序列任务很少涉及相同长度的输入和输出。一种语言的短语可能翻译成另一种语言的更长句子。一篇长文章可能只被总结成几句话。模型架构必须足够灵活以处理这些变化，接收任意长度$N$的输入并生成任意长度$M$的输出，其中$N$和$M$对于每个示例都可以不同。

这些挑战凸显了对以下架构的需求：能够有效获取序列依赖，处理长距离上下文 (context)而不丢失信息，并管理可变序列长度。了解这些难题推动了注意力等机制的进步，它们直接解决了瓶颈问题并改善了依赖关系的处理，为像Transformer这样的模型开辟了道路。在下一节中，我们将简要回顾循环神经网络 (neural network)（RNN），这是一种用于处理序列数据的早期方法，然后检查它们的具体局限性。

## 参考资料

- [Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c113e317042aee7eb-Paper.pdf) — Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014)
  Journal: Advances in Neural Information Processing Systems 27 (NIPS 2014)
  介绍了使用LSTMs进行序列到序列学习的基础编码器-解码器架构，展示了早期解决这些挑战的方法。
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473) — Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2014)
  Journal: International Conference on Learning Representations (ICLR 2015, poster)
  提出了注意力机制，这是一项重要创新，它通过允许解码器选择性地关注输入部分，解决了序列到序列模型中的固定上下文瓶颈问题。
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017)
  介绍了Transformer模型，该模型完全依赖注意力机制在序列到序列任务中取得了最先进的成果，有效克服了文中讨论的局限性。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2023)
  Publisher: Draft textbook, Stanford University
  提供了自然语言处理概念的全面介绍，包括序列建模、循环神经网络以及处理序列数据固有的挑战。
