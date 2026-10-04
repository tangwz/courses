---
course: "foundations-transformers-architecture"
chapter: "revisiting-sequence-modeling-limitations"
lesson: "sequential-computation-rnns"
sourceId: 1913
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-1-revisiting-sequence-modeling-limitations/sequential-computation-rnns"
title: "循环网络中的顺序计算"
description: "分析循环神经网络（RNN）固有的顺序处理特性。"
order: 1
plots: []
sourceHash: "1ddcd544ab017016bc9332ec9f67108e61aa8e2669bc052bfec6414511c0f2a2"
sourceCorrections: []
---

循环神经网络 (neural network)（RNN），包括LSTMs和GRUs等更复杂的变体，都遵循逐元素处理序列的原理进行设计。在每个时间步 $t$，RNN 接收输入 $x_t$ 和来自上一个时间步的隐藏状态 $h_{t-1}$，以计算当前隐藏状态 $h_t$。这个过程可以抽象地表示为：


$$
h_t = f(h_{t-1}, x_t; \theta)
$$


其中 $f$ 代表循环函数（例如，涉及矩阵乘法和tanh或sigmoid等激活函数 (activation function)，或者LSTMs/GRUs中更复杂的门控逻辑），并由权重 (weight) $\theta$ 参数 (parameter)化。在每个时间步也可以生成一个可选输出 $y_t$，通常由 $h_t$ 得到。

这种表述显示了循环模型的一个基本特性：一种固有的**顺序依赖**。时间步 $t$ 隐藏状态的计算*必须*等待时间步 $t-1$ 的计算完成后才能进行。这种依赖形成了一个贯穿整个序列长度的链条。

> 一个展开的RNN，说明了信息流的顺序性。隐藏状态 $h_t$ 直接依赖于前一个状态 $h_{t-1}$ 和当前输入 $x_t$。

尽管单个时间步*内部*的计算（例如函数 $f$ 内的矩阵乘法）通常可以利用GPU等并行硬件，但*跨*时间步的计算无法并行化。你不能同时计算 $h_t$ 和 $h_{t+1}$，因为 $h_{t+1}$ 需要 $h_t$ 作为输入。

这种顺序约束对性能有重要影响：

1. **训练时间：** 对于长度为 $L$ 的序列，前向和后向传播的时间复杂度与 $L$ 呈线性关系。处理更长的序列会直接导致更长的训练时间，不论针对步内计算可用的并行处理能力有多强。
2. **推理 (inference)延迟：** 同样，生成输出序列或基于长输入序列进行预测，需要顺序迭代每个时间步，从而增加了延迟。

这与应用于序列的前馈网络或卷积神经网络（CNN）（例如1D卷积）等架构形成鲜明对比。在这些模型中，输入序列不同部分的计算通常可以独立并行地执行，从而带来更高的效率，特别是在专用硬件上。

尽管LSTMs和GRUs引入了更先进的门控机制，以更好地控制信息流并解决梯度问题（接下来会讨论），但它们本质上仍然遵循相同的顺序处理方法。时间 $t$ 的门（输入、遗忘、输出）和单元状态的计算仍然依赖于时间 $t-1$ 的隐藏状态和单元状态。

因此，无法在序列长度维度上并行化计算，是循环网络主要设计中固有的一个基本瓶颈。这个局限是促使我们发展出其他架构（如Transformer）的主要驱动力，这些架构能够更并发地处理序列元素。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: The MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了原始的LSTM架构，概述了其门控机制和序列计算流程。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  提供了循环神经网络的全面解释，涵盖了它们的序列特性和计算图。参见第10章：序列建模：循环和递归网络。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Pages: 15 pages; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构，直接指出循环网络中序列计算的局限性是其一个推动因素。
