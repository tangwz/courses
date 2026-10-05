# 循环神经网络 (RNN) 概述

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-7-introduction-common-architectures/rnn-overview)

[返回章节目录](README.md) · [返回课程目录](../README.md)

前馈网络是独立处理输入的。然而，许多问题都涉及序列数据，其中顺序很重要，并且先前项的背景信息会影响当前项。例如，理解一个句子、预测股价或转录语音。每个词、价格点或声音片段都依赖于它之前的内容。标准的前馈网络缺乏一种内在机制来‘记住’序列中的过去信息。

这就是循环神经网络 (neural network)（RNN）的作用所在。它们通过引入**循环**的构想，专门设计用于处理序列数据。

### 记忆的构想：隐状态

RNN 的决定性特征是其内部循环。在处理序列的每一步，网络不仅考虑当前输入，还会考虑它从先前步骤中保留下来的信息。这些保留的信息存储在所谓的**隐状态**中。

设想你正在阅读一个句子。你不会孤立地处理每个词。你对当前词的把握会受到你已阅读词语的很大影响。RNN 中的隐状态就像这种运行中的总结或背景信息。它捕获了序列中先前元素的相关信息。

### 逐步处理序列

RNN 一次处理序列中的一个元素（或“时间步”）时。对于每个时间步 $t$：

1. 它接收该时间步的输入，我们称之为 $x_t$。
2. 它还接收来自前一个时间步的隐状态 $h_{t-1}$。
3. 它使用一组学习到的权重 (weight)结合 $x_t$ 和 $h_{t-1}$，以计算新的隐状态 $h_t$。这个新的隐状态现在包含了从所有步骤直到 $t$ 的信息。
4. 可选地，它可以为当前时间步生成一个输出 $y_t$，这通常基于隐状态 $h_t$。

重要的是，**每个时间步**都使用相同的一组权重（结合输入和先前状态以及生成输出的规则）。这种权重共享使得 RNN 效率高，并能使其将模式推广到不同长度的序列。

### 可视化循环：时间上的展开

通常，通过在时间上“展开”RNN 会有所帮助。我们可以绘制一条链来表示网络在每个时间步的状态，而不是绘制循环。

> 一个在时间上“展开”的 RNN。相同的 RNN 单元（代表共享权重 (weight)）处理输入 $x_t$ 和先前的隐状态 $h_{t-1}$，以生成新的隐状态 $h_t$ 和可选的输出 $y_t$。隐状态从一个时间步传递到下一个时间步。

从数学角度看，简单 RNN 单元在时间步 $t$ 内的核心计算通常表示为：

计算新的隐状态 $h_t$：


$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$


计算输出 $y_t$：


$$
y_t = W_{hy} h_t + b_y
$$


这里：

- $x_t$ 是时间步 $t$ 的输入。
- $h_{t-1}$ 是来自前一个时间步的隐状态。
- $h_t$ 是时间步 $t$ 的新隐状态。
- $y_t$ 是时间步 $t$ 的输出。
- $W_{hh}$、$W_{xh}$ 和 $W_{hy}$ 是在训练期间学习到的权重矩阵。它们分别代表了先前隐状态、当前输入和当前隐状态的影响程度。这些权重在所有时间步之间是**共享**的。
- $b_h$ 和 $b_y$ 是偏置 (bias)向量 (vector)，也是学习得到的。
- $\tanh$ 是双曲正切激活函数 (activation function)，常用于简单的 RNN 中以引入非线性。根据具体任务，输出层可以使用其他激活函数（例如，用于分类的 Softmax）。

重要之处在于 $h_t$ 的循环公式，它同时依赖于当前输入 $x_t$ 和先前的隐状态 $h_{t-1}$。正是这种依赖性赋予了 RNN 记忆能力。

### RNN 的应用场景

RNN 在处理序列模式的任务中表现出色：

- **自然语言处理（NLP）：** 语言建模（预测下一个词）、机器翻译、情感分析、文本生成。
- **语音识别：** 将口语音频转换为文本。
- **时间序列分析：** 预测股价、天气预报、传感器数据分析。
- **视频分析：** 理解视频帧中随时间发生的动作。

### 挑战与后续

尽管功能强大，但像上面描述的简单 RNN 在学习长距离依赖时可能会遇到困难。来自早期时间步的信息在通过多个步骤传播时可能会被稀释或丢失，这个问题通常被称为**梯度消失问题**。反之，梯度有时可能会变得过大，这被称为**梯度爆炸问题**。

这些挑战促成了更精密的循环架构的发展，如长短期记忆（LSTM）和门控循环单元（GRU），它们使用门控机制来更好地控制信息流和记忆。本章稍后将简要提及这些内容。

目前，掌握循环的核心思想、隐状态的作用以及逐步处理过程就足够了。在接下来的部分中，我们将了解如何使用 PyTorch 的 `nn.RNN` 模块实现一个基本的 RNN。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10: Sequence Modeling: Recurrent and Recursive Nets
  对循环神经网络进行了广泛的学术讨论，涵盖其架构、训练方法和变体。
- [RNN - PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) — PyTorch Development Team (2024)
  Publisher: PyTorch Foundation
  PyTorch `nn.RNN` 模块的官方文档，展示了如何在 PyTorch 中实现基本的循环层。
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: The MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  这篇论文介绍了长短期记忆 (LSTM) 网络，它提升了 RNN 在长序列中保留信息的能力。
- [Learning long-term dependencies with gradient descent is difficult](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEzT5PbIIOWKbdFz6q7XK0YHYL7n5ZfwNfZqbIF_Yl9s8JiMbpkcnaF1ScLFSKnOy8KRjI9bS94ySTqkNyB5MZcpm3Mjr7ZTQecA_OtpBONNedD573fk2OeWo-I4x6P2QDCnZ0D4IDkClfL3--CaC7e6SnJ6AH6NM9ANwqUB_YGzQM6FNn50SX2oWPVB0rS3K77HRsiQjeD0Dh_e7R9RD5pzDlgozmQiwnWsjCRDkCQtBlM4zs30GXOBTfWo_t2a9jVWhSdPG8Kqg==) — Yoshua Bengio, Patrice Simard, and Paolo Frasconi (1994)
  Journal: IEEE Transactions on Neural Networks; Volume: 5; Pages: 157-166; DOI: [10.1109/72.279181](https://doi.org/10.1109/72.279181)
  一篇开创性论文，指出了循环神经网络在学习长程模式时存在的梯度消失和梯度爆炸问题。

---

[上一节](03-%E7%90%86%E8%A7%A3CNN%E5%B1%82%E7%9A%84%E8%BE%93%E5%85%A5-%E8%BE%93%E5%87%BA%E5%BD%A2%E7%8A%B6.md) · [下一节](05-%E5%9C%A8PyTorch%E4%B8%AD%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84RNN.md)
