---
course: "deep-learning-fundamentals-keras"
chapter: "recurrent-neural-networks-rnns"
lesson: "vanishing-gradient-problem"
sourceId: 4961
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-5-recurrent-neural-networks-rnns/vanishing-gradient-problem"
title: "梯度消失问题"
description: "了解SimpleRNN的局限性，特别是梯度消失难题。"
order: 4
plots: ["plots/4961-0.json"]
sourceHash: "2639bdd4e6e0813e4bfecdac58e13c8ba3e3f91cd5294a4e823c6960efa7455a"
sourceCorrections: []
---

循环网络的核心思想是维持一个隐藏状态，该状态携带来自先前时间步的信息，以影响当前步骤的处理。这种核心思想，通过`SimpleRNN`等层得到体现，对于捕获序列中的上下文 (context)非常有效。然而，训练这些网络，特别是较简单的网络，会遇到一个主要的实际难题，主要被称为梯度消失问题。

回顾一下，神经网络 (neural network)通过根据损失函数 (loss function)的梯度来调整其权重 (weight)进行学习。这个梯度表示每个权重的微小变化会如何影响最终损失。在训练过程中，我们使用反向传播 (backpropagation)来计算这些梯度。在RNN中，由于给定时间步的输出依赖于先前时间步的计算，我们需要反向传播梯度，这不仅通过当前时间步的层进行，而且还 *通过时间* 本身进行。这个过程常被称为时间反向传播（BPTT）。

想象一下将RNN沿序列展开。时间步 $t$ 的计算依赖于 $t$ 时刻的输入和 $t-1$ 时刻的隐藏状态。$t-1$ 时刻的隐藏状态依赖于 $t-1$ 时刻的输入和 $t-2$ 时刻的隐藏状态，以此类推。当计算损失相对于影响早期隐藏状态（例如，在时间步 $t-k$）的权重的梯度时，链式法则要求我们将许多梯度项相乘，每个我们反向传播的时间步都对应一项。

具体来说，梯度计算涉及反复乘以一个时间步的隐藏状态相对于前一个时间步隐藏状态的导数。这个导数通常包含循环权重矩阵以及循环单元中使用的激活函数 (activation function)的导数。

当这些项持续小于1时，问题就出现了。考虑一个简单情况，其中梯度计算的相关部分涉及反复乘以一个值，例如 $0.8$。

- 回溯1步后：梯度被缩放 $0.8$ 倍。
- 回溯2步后：梯度被缩放 $0.8 \times 0.8 = 0.64$ 倍。
- 回溯10步后：梯度被缩放 $0.8^{10} \approx 0.107$ 倍。
- 回溯20步后：梯度被缩放 $0.8^{20} \approx 0.0115$ 倍。

随着你在时间上进一步反向传播，梯度幅度呈指数级缩小，趋近于零。



![BPTT中梯度衰减的示例](plots/4961-0.json)



> 示意图，说明在简单RNN中，梯度幅度在时间步中反向传播时如何呈指数级下降。

为什么这些项可能小于1？

1. **激活函数：** 常见的激活函数，如sigmoid和tanh，在零附近的狭窄范围之外，其梯度小于1（通常远小于1）。当这些梯度是链式法则计算的一部分时，它们会导致缩小效应。
2. **循环权重：** 循环权重矩阵中值的幅度也起到作用。如果权重通常很小，这会加剧缩小。

梯度消失的影响很大：网络难以学习序列中距离较远元素之间的依赖关系。如果与早期时间步相关的梯度变得微乎其微，那么与该时间步相关的权重将无法有效更新。网络基本上无法“记住”或从许多步骤之前看到的信息中学习。对于文本生成或情感分析等任务，理解上下文可能依赖于之前很久看到的词或短语，这是一个主要限制。

`SimpleRNN` 架构特别容易出现这个问题，因为它缺乏机制来长时间调节信息和梯度的流动。隐藏状态在每个步骤中被简单地覆盖，使得难以保留来自遥远过去的重要信息。

虽然在实践中较少见但在数学上可能，相反的问题也可能发生：**梯度爆炸**。如果相关的梯度项持续大于1，它们的乘积可以呈指数级增长，导致巨大的梯度、不稳定的训练和数值溢出（训练期间经常导致 `NaN` 值）。梯度裁剪（将梯度幅度限制在某个阈值）等技术常用于缓解梯度爆炸。

然而，梯度消失问题是简单RNN试图捕获长期依赖关系时更持久的难题。这一局限性是开发更复杂的循环架构（如长短期记忆（LSTM）和门控循环单元（GRU））的主要驱动力，我们接下来将检查这些架构。这些架构包含显式门控机制，旨在更好地控制信息和梯度随时间流动，使它们能够在序列数据中学习更长期的模式。

## 参考资料

- [Learning long-term dependencies with gradient descent is difficult](https://ieeexplore.ieee.org/document/279181) — Yoshua Bengio, Patrice Simard, Paolo Frasconi (1994)
  Journal: IEEE Transactions on Neural Networks; Publisher: IEEE; Volume: 5; Pages: 157-166; DOI: [10.1109/72.279181](https://doi.org/10.1109/72.279181)
  一篇基础论文，首次明确识别并分析了循环神经网络中的梯度消失和梯度爆炸问题。
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了长短期记忆（LSTM）网络的原始论文，该网络通过其门控机制有效地缓解了梯度消失问题。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10
  一本全面的深度学习教科书，涵盖了深度学习的理论基础和实践方面，包括对循环神经网络和梯度消失/爆炸问题的详细解释。
- [On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.html) — Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013)
  Journal: Proceedings of the 30th International Conference on Machine Learning; Publisher: PMLR; Volume: 28; Pages: 1310-1318
  这篇论文深入探讨了循环神经网络中的梯度消失和爆炸问题，并提出了诸如梯度裁剪等实际解决方案，以实现稳定的训练。
