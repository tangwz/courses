---
course: "rnns-and-sequence-modeling"
chapter: "rnn-fundamentals"
lesson: "core-idea-iterative-processing"
sourceId: 2526
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-2-rnn-fundamentals/core-idea-iterative-processing"
title: "核心思想：迭代处理序列"
description: "阐述一步步处理序列元素并维持状态的基本原理。"
order: 1
plots: []
sourceHash: "a83bfc24fec839900f9031e44d046728f0664398292955d155c798083c41a091"
sourceCorrections: []
---

正如我们在上一章讨论的，标准前馈神经网络 (neural network)在处理序列数据时有一个基本局限：它们没有记忆功能。每个输入都被独立处理，不考虑之前输入提供的顺序或语境。想象一下阅读这个句子。为了理解“句子”这个词的含义，你的大脑会使用前面词语“阅读这个”提供的语境。前馈网络无法自然地做到这一点。它们是为固定大小的输入设计的，在这些输入中，顺序本身不以同样的方式带有含义。

循环神经网络 (RNN)（RNNs）的主要思想是，它不一次性处理整个序列，也不单独处理每个元素，而是**一次处理一个元素，迭代地处理序列。**

设想处理一个序列 $x = (x_1, x_2, ..., x_T)$。RNN接收第一个元素 $x_1$，对其进行处理，并生成一个内部状态（通常称为隐藏状态），我们称之为 $h_1$。当处理第二个元素 $x_2$ 时，RNN不仅仅查看 $x_2$。它还会考虑从第一步获得的信息，这些信息归纳在隐藏状态 $h_1$ 中。它将新的输入 $x_2$ 与先前的状态 $h_1$ 结合，以计算下一个隐藏状态 $h_2$。

这个过程对序列中的每个元素都重复：

- 在第 $t=1$ 步：使用 $x_1$ 计算 $h_1$（通常还会使用初始状态 $h_0$，通常设为零）。
- 在第 $t=2$ 步：使用 $x_2$ 和 $h_1$ 计算 $h_2$。
- 在第 $t=3$ 步：使用 $x_3$ 和 $h_2$ 计算 $h_3$。
- ...
- 在第 $t=T$ 步：使用 $x_T$ 和 $h_{T-1}$ 计算 $h_T$。

**隐藏状态 $h_t$** 充当网络的记忆。它捕获了网络认为与处理当前元素 $x_t$ 和未来元素相关的所有先前元素 ($x_1, ..., x_{t-1}$) 的信息。这个状态从一个时间步传递到下一个时间步，从而在网络连接中形成一个循环或递归。

> RNN 处理一步的示意图。输入 $x_t$ 和先前的隐藏状态 $h_{t-1}$ 通过 RNN 操作结合，生成当前的隐藏状态 $h_t$。这个状态 $h_t$ 随后被传递到下一个时间步（处理 $x_{t+1}$）使用，并且可以在这一步生成一个可选输出 $y_t$。

这种带有持久状态的迭代处理与前馈网络本质上不同。前馈网络对每个输入独立应用相同的转换，而RNN在每一步应用相同的转换*规则*（RNN单元内相同的权重 (weight)集合），但其*结果*（隐藏状态 $h_t$ 和输出 $y_t$）既取决于当前输入 $x_t$，也取决于归纳在 $h_{t-1}$ 中的历史信息。

这种维持语境的能力使RNN天生适合处理顺序很重要且信息需要随时间累积的任务，例如理解语言、预测时间序列或分析音频信号。图中“RNN操作”框内使用的具体数学函数规定了先前的状态和当前输入如何结合，我们将在接下来的章节中查看这些细节。

## 参考资料

- [Finding Structure in Time](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1) — Jeffrey L. Elman (1990)
  Journal: Cognitive Science; Publisher: Wiley; Volume: 14; Pages: 179-211; DOI: [10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1)
  一篇基础性论文，介绍了简单循环网络 (SRN)，展示了神经网络如何随着时间发展出序列信息和上下文的内部表示，为迭代处理奠定了基础。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10
  这本权威教材全面介绍了循环神经网络，清晰地阐述了其基本结构、隐藏状态的概念以及它们如何迭代处理序列数据。
- [Recurrent Neural Networks and Language Models](http://web.stanford.edu/class/cs224n/slides/cs224n-2023-lec05-rnnlm.pdf) — Christopher Manning (2023)
  Publisher: Stanford University
  来自一门备受推崇的课程的讲义，提供了对循环神经网络、其迭代处理以及隐藏状态在自然语言处理任务中维持上下文作用的清晰解释和视觉呈现。
