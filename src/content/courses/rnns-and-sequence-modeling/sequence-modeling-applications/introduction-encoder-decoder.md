---
course: "rnns-and-sequence-modeling"
chapter: "sequence-modeling-applications"
lesson: "introduction-encoder-decoder"
sourceId: 2642
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-9-sequence-modeling-applications/introduction-encoder-decoder"
title: "编码器-解码器架构简介"
description: "简要介绍使用循环神经网络的序列到序列（Seq2Seq）架构。"
order: 7
plots: []
sourceHash: "b40bf72579d6a0bd629179ad329f7cf8a1895e356fabdbd80d9e2c5f6fc56b65"
sourceCorrections: []
---

尽管LSTMs和GRUs显著提升了循环网络处理长距离依赖的能力，但标准的RNN基本结构（用于分类的多对一，以及输入和输出长度常对齐 (alignment)的预测任务中的多对多）仍存在局限性。以机器翻译等任务为例：输入句子（“Hello world”）和输出句子（“Bonjour le monde”）的长度可能不同，并且词语间的关系也并非总是时间步上简单的多对一映射。类似地，将长文档概括为短段落需要处理长的输入序列以生成短得多的输出序列。

对于这类问题，即输入和输出序列的长度和结构可能不同，标准的RNN方法不足。我们需要一种更灵活的架构。**编码器-解码器**架构（常称作\*\*序列到序列（Seq2Seq）\*\*模型）便由此出现。

### 编码器-解码器思路

核心思想很巧妙：将任务分为两个不同的阶段，由两个独立的循环网络（或网络堆栈）处理：

1. **编码器：** 该网络逐步读取整个输入序列，就像我们研究过的RNN一样。其主要目的不是在每个时间步生成输出，而是将整个输入序列的信息压缩成一个单一的、固定大小的向量 (vector)。这个向量常被称为**上下文 (context)向量**或有时称作“思想向量”。它旨在捕获输入序列的语义核心或概括。通常，编码器RNN的最终隐藏状态（或LSTM的单元状态）充当此上下文向量。
2. **解码器：** 该网络接收编码器生成的上下文向量作为输入（通常用于初始化其自身的隐藏状态）。其作用是逐个元素生成输出序列。在每个步骤中，它生成一个输出元素（例如，目标语言中的一个词）并更新其隐藏状态。当前步骤生成的输出通常会作为下一个步骤的输入反馈回去，使解码器能够根据其目前已生成的内容来条件化其未来的输出。

这种分离使得模型能够处理不同长度的输入和输出序列。编码器将变长输入映射到一个固定大小的上下文，解码器再将该固定大小的上下文映射回变长输出。

### 架构可视化

我们可以将此流程可视化如下：

> 编码器-解码器架构的概览。编码器处理输入序列以生成上下文 (context)向量 (vector)，该向量随后初始化解码器以生成输出序列。

### 工作原理（简化版）

让我们直观地思考这个过程：

1. **编码：** 编码器（一个LSTM或GRU）逐词（或逐字符）读取输入句子。每读取一个词，它就更新其隐藏状态。处理完最后一个词后，最终隐藏状态会封装整个句子的含义。这个最终隐藏状态成为上下文 (context)向量 (vector) $c$。
   
   $$
   h_t^{enc} = \text{RNN}_{enc}(x_t, h_{t-1}^{enc})
   $$
   
   
   $$
   c = h_T^{enc} \quad (\text{T 为输入长度})
   $$
   
2. **解码：** 解码器（另一个LSTM或GRU）被初始化。其初始隐藏状态 $h_0^{dec}$ 通常根据上下文向量 $c$ 设定（例如，$h_0^{dec} = c$）。
3. 解码器随后开始生成输出序列。
   - 它可能接收一个特殊的序列开始标记 (token) `<sos>` 作为其第一个输入。
   - 利用其当前的隐藏状态 $h_{t'-1}^{dec}$ 和输入（最初为 `<sos>`，之后为前一个输出词），它预测第一个输出词 $y_1$ 并将其状态更新为 $h_1^{dec}$。
     
     $$
     (y_{t'}, h_{t'}^{dec}) = \text{RNN}_{dec}(\text{input}_{t'}, h_{t'-1}^{dec})
     $$
     
     (当 $\text{input}_{t'}$ 在 $t'=1$ 时为`<sos>`，在 $t'>1$ 时为 $y_{t'-1}$)
   - 这个预测的词 $y_1$ 随后被用作下一个时间步的输入，以预测 $y_2$，以此类推。
   - 此过程一直持续，直到解码器输出一个特殊的序列结束标记 `<eos>` 或达到预定义的最大长度。

### 常见应用

编码器-解码器架构为许多复杂序列建模任务奠定了基础：

- **机器翻译：** 将文本从一种语言翻译成另一种语言（例如，英语到法语）。
- **文本概括：** 从较长的文本文档中生成简洁的概括。
- **问答：** 根据给定问题和上下文 (context)生成答案序列。
- **聊天机器人：** 生成对话回复。
- **图像描述：** 为图像生成文本描述（在此，编码器通常是卷积神经网络 (neural network)（CNN），它从图像生成上下文向量 (vector)）。

### 局限性与后续内容

虽然功能强大，但基本的编码器-解码器架构存在一个潜在瓶颈：单一的、固定大小的上下文 (context)向量 (vector) $c$。对于非常长的输入序列，将所有必要信息压缩到这个单一向量中可能存在困难。解码器在生成整个输出时，只能依靠这一个摘要进行工作。直观来说，解码器在生成输出的不同部分时，能够选择性地关注输入序列的不同部分会很有帮助。

这一局限性促使了**注意力机制 (attention mechanism)**的发展，它使得解码器在生成过程的每个步骤中，能够动态地回顾编码器隐藏状态中的相关部分（而不仅仅是最终状态）。我们将在下一节简要介绍注意力机制，因为它显著提升了序列到序列模型的性能，特别是对于较长序列。

## 参考资料

- [Sequence to Sequence Learning with Neural Networks](https://doi.org/10.48550/arXiv.1409.3215) — Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 27); Pages: 3104-3112; DOI: [10.48550/arXiv.1409.3215](https://doi.org/10.48550/arXiv.1409.3215)
  这篇开创性论文介绍了使用深度LSTM的序列到序列（Seq2Seq）架构，用于机器翻译等任务，证明了其在映射不同长度输入序列到输出序列方面的有效性。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179/) — Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1724-1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  这篇论文同时提出了RNN编码器-解码器框架并引入了门控循环单元（GRUs），展示了如何学习机器翻译中短语的连续表示。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  这是一本全面的教材，涵盖了深度学习的理论基础和实际应用，包括对循环神经网络、LSTM、GRU和编码器-解码器架构的详细解释。特别是第10章“序列建模：循环和递归网络”，深入讨论了这些架构。
