---
course: "rnns-and-sequence-modeling"
chapter: "implementing-lstm-gru"
lesson: "understanding-bidirectional-rnns"
sourceId: 2606
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-7-implementing-lstm-gru/understanding-bidirectional-rnns"
title: "理解双向循环神经网络"
description: "解释双向LSTM/GRU的原理，以便在前向和后向两个方向处理序列。"
order: 5
plots: []
sourceHash: "f28e7169ada343aa8060d043a83dd01ee360b202c791fd932511d755bb0eb12a"
sourceCorrections: []
---

标准循环神经网络 (neural network) (RNN)，例如LSTM和GRU，以单向方式处理序列，通常按时间顺序从开始到结束。在任意给定时间步$t$，隐藏状态$h_t$概括了来自过去的输入$x_1, x_2, ..., x_t$的信息。尽管这反映了我们经常感受时间相关现象的方式，但对于某些任务而言可能存在局限。

考虑理解句子中一个词的含义。有时，正确解释一个词所需的上下文 (context)会在该词 *之后* 出现。例如，在句子“他与棒球队一起吃了一个**蝙蝠**”中，知道“棒球队”这些词有助于消除“bat”的歧义（更可能是运动器材，而不是动物）。一个从左到右处理的标准RNN，在看到澄清上下文之前就已经处理了“bat”。

在这里，双向循环神经网络（BiRNN）提供了优势。其主要思想简单明了：同时使用两个独立的循环层，以两个方向处理序列。

1. **前向层：** 该层从第一个元素到最后一个元素（$t=1$到$T$）处理输入序列。它在时间$t$的隐藏状态，记为$\overrightarrow{h_t}$，捕捉*过去*上下文（$x_1, ..., x_t$）的信息。
2. **后向层：** 该层反向处理输入序列，从最后一个元素到第一个元素（$t=T$到$1$）。它在时间$t$的隐藏状态，记为$\overleftarrow{h_t}$，捕捉*未来*上下文（$x_t, ..., x_T$）的信息。

这两个层独立运行，每个都维护自己的一组权重 (weight)和隐藏状态。它们可以由简单的RNN、LSTM或GRU单元组成。

### 架构与信息流

在每个时间步$t$，BiRNN会产生一个输出，其中包含来自前向和后向处理的信息。将两个层的信息组合起来的最常见方法是在该时间步连接它们各自的隐藏状态。

The overall hidden state or output representation $y_t$ at time step $t$ can be formed as:


$$
y_t = g([\overrightarrow{h_t} ; \overleftarrow{h_t}])
$$


这里，$[\overrightarrow{h_t} ; \overleftarrow{h_t}]$表示前向隐藏状态$\overrightarrow{h_t}$与后向隐藏状态$\overleftarrow{h_t}$的连接。函数$g$可以是一个恒等函数（直接使用连接后的状态），或者它可能涉及进一步的处理，例如将连接后的向量 (vector)通过一个全连接层，具体取决于模型架构和任务。其他组合方法，如求和或平均，也存在，但不如连接方法常见。

下图展示了这种结构：

> 一个双向RNN使用两个独立的循环层处理输入序列$x_1, ..., x_T$。前向层根据过去信息计算隐藏状态$\overrightarrow{h_t}$，而后向层根据未来信息计算$\overleftarrow{h_t}$。每一步的最终输出$y_t$结合了$\overrightarrow{h_t}$和$\overleftarrow{h_t}$，通常通过连接操作。

### 优点与缺点

BiRNN的主要优点是它们能够整合来自两个方向的上下文 (context)。这通常会提高在那些元素理解依赖于其周围上下文的任务上的表现。例子包括：

- **自然语言处理：** 情感分析、命名实体识别（NER）、词性标注和机器翻译经常受益于双向上下文。
- **语音识别：** 理解音素可能取决于周围的声音。
- **生物信息学：** 分析DNA或蛋白质结构等序列。

然而，BiRNN也伴随着一些考量：

- **对完整序列的依赖：** 后向传递需要从末尾到开头处理序列。这意味着在计算完成之前，必须提供整个输入序列。因此，BiRNN通常不适合实时应用，在这些应用中，预测必须随着数据到达而增量进行，且无法访问未来的输入（例如，仅基于过去价格的实时股票价格预测）。
- **计算成本增加：** 使用两个循环层而非一个，与相同隐藏层大小和类型的单向RNN相比，参数 (parameter)数量和训练及推断所需的计算量大约增加一倍。

### 何时使用双向模型

在以下情况选择双向架构：

1. 任务涉及处理整个序列，其中来自过去和未来元素的上下文 (context)对于在每一步进行预测或分类有益（例如，像命名实体识别这样的序列标注）。
2. 任务需要根据所有元素对整个序列进行分类或概括（例如，对完整评论的情感分析）。
3. 提前需要完整序列的限制对于该应用来说是可接受的。

对于需要真正的在线处理或预测，且在预测时无法获得未来输入的任务，应避免使用双向架构。在此类情况下，标准的单向RNN是合适的选择。

在理解了双向处理的思想和用途之后，我们现在将探讨如何使用流行的深度学习 (deep learning)框架实现标准和双向的LSTM和GRU层。

## 参考资料

- [Bidirectional recurrent neural networks](https://ieeexplore.ieee.org/document/650093) — Mike Schuster and Kuldip K. Paliwal (1997)
  Journal: IEEE Transactions on Signal Processing; Publisher: IEEE; Volume: 45; Pages: 2673-2681; DOI: [10.1109/78.650093](https://doi.org/10.1109/78.650093)
  提出双向循环神经网络架构的开创性论文。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖深度学习理论基础和实际应用的综合教材，包含循环神经网络及其变体的详细章节。
- [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991) — Zhiheng Huang, Wei Xu, Kai Yu (2015)
  Journal: arXiv; DOI: [10.48550/arXiv.1508.01991](https://doi.org/10.48550/arXiv.1508.01991)
  这篇有影响力的论文展示了结合双向LSTM和条件随机场的序列标注任务的有效性，这是双向RNN的常见应用领域。
- [Stanford CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/index.html) — Diyi Yang, Tatsunori Hashimoto (2023)
  Publisher: Stanford University
  知名大学课程的官方讲义，在自然语言处理的背景下，清晰地解释了RNN、LSTM、GRU和双向RNN的概念和实用视角。
