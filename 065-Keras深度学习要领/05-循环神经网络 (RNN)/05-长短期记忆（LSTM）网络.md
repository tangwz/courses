# 长短期记忆（LSTM）网络

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-5-recurrent-neural-networks-rnns/long-short-term-memory-lstm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管 `SimpleRNN` 层直观，但在学习跨越长序列的模式时表现不佳。这主要是由于梯度消失问题，其中早期输入的影响在反向传播 (backpropagation)过程中迅速减弱，使得网络难以根据长距离依赖关系调整权重 (weight)。

为克服这一重要局限，Sepp Hochreiter 和 Jürgen Schmidhuber 于 1997 年开发了一种更复杂的循环架构，称为长短期记忆（LSTM）网络。LSTM 专门设计用于长时间记忆信息，使其在多种序列建模任务中非常有效。

LSTM 的核心改进在于其内部结构。与只有一个简单循环连接和变换的 `SimpleRNN` 单元不同，一个 LSTM 单元包含一个**记忆单元**（或称单元状态）和多个**门**。可以将单元状态（常用 $C_t$ 表示）看作网络的长期记忆。它可以使信息在多个时间步中相对不变地传递。这些门是特殊的神经网络 (neural network)组件，它们像控制器一样，调节信息进出此单元状态的流动。它们学习在每个时间步保留哪些信息、丢弃哪些信息以及输出哪些信息。

让我们看看 LSTM 单元在时间步 $t$ 的主要组成部分：

### 门

LSTM 单元通常有三个主要的门来控制信息流动。这些门使用 sigmoid 激活函数 (activation function)，输出介于 0 和 1 之间的值。接近 0 的值表示“不让任何信息通过”，而接近 1 的值表示“让所有信息通过”。它们将当前输入 $x_t$ 和前一个隐藏状态 $h_{t-1}$ 作为输入。

1. **遗忘门 ($f_t$):** 此门决定应从单元状态中移除哪些信息。它查看 $h_{t-1}$ 和 $x_t$，并为前一个单元状态 $C_{t-1}$ 中的每个数字输出一个介于 0 和 1 之间的值。1 表示“完全保留此信息”，而 0 表示“完全清除此信息”。
2. **输入门 ($i_t$):** 此门决定应在单元状态中存储哪些新信息。它由两部分组成：

   - 一个 sigmoid 层决定更新哪些值（输出 $i_t$）。
   - 一个 `tanh` 层创建一个新的候选值向量 (vector) $\tilde{C}_t$，这些值可以被添加到状态中。`tanh` 函数输出介于 -1 和 1 之间的值。
3. **输出门 ($o_t$):** 此门决定下一个隐藏状态 $h_t$ 应是什么。输出将基于过滤后的单元状态。首先，一个 sigmoid 层决定单元状态的哪些部分要输出。然后，当前单元状态 $C_t$ 通过 `tanh`（将值推到 -1 和 1 之间）并乘以 sigmoid 门 ($o_t$) 的输出。这个过滤后的版本成为隐藏状态 $h_t$，它被传递到下一个时间步，也可以用作当前时间步的输出。

### 单元状态 ($C_t$)

单元状态就像信息的传送带。它直接沿着整个时间步链条运行，只有门控制的少量线性交互。遗忘门选择性地移除前一个单元状态 $C_{t-1}$ 中的信息，而输入门选择性地添加新的候选信息 $\tilde{C}_t$。更新方式如下：


$$
C_t = (f_t * C_{t-1}) + (i_t * \tilde{C}_t)
$$


这里， $*$ 表示元素级乘法。第一项 ($f_t * C_{t-1}$) 表示从前一个状态保留的信息，第二项 ($i_t * \tilde{C}_t$) 表示添加的新信息。与 `SimpleRNN` 中重复的矩阵乘法和非线性变换相比，这种结构使得信息更容易长时间保持。

### 隐藏状态 ($h_t$)

隐藏状态 $h_t$ 根据过滤后的单元状态 $C_t$ 计算得出，并由输出门 $o_t$ 控制。


$$
h_t = o_t * \tanh(C_t)
$$


这个 $h_t$ 随后被传递给下一个时间步 ($t+1$) 的 LSTM 单元，也可以用作当前时间步 $t$ 的输出预测。

> 单个 LSTM 单元内部信息流的图示。门（蓝色）控制单元状态（绿色圆柱体）的更新方式以及作为隐藏状态（橙色平行四边形）输出的内容。Sigmoid ($\sigma$) 和 tanh 激活函数 (activation function)用于门内和状态更新。

借由这种门控机制和独立的单元状态路径，LSTM 可以有效地学习在长序列中存储、遗忘和输出哪些信息。这解决了梯度消失问题，并使它们能够捕获可能相隔数百个时间步的依赖关系，从而成为许多序列相关任务的常用选择。

在下一节，我们将了解如何使用 Keras 提供的便捷 `LSTM` 层来实现 LSTM。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍LSTM架构的基础论文，旨在解决循环神经网络中的梯度消失问题。
- [Deep Learning (Chapter 10: Sequence Modeling: Recurrent and Recursive Networks)](https://www.deeplearningbook.org/contents/rnn.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本被广泛引用的教材中的一个章节，提供了循环神经网络的理论解释，包括LSTM的机制及其解决的问题。
- [Keras LSTM Layer](https://keras.io/api/layers/recurrent_layers/lstm/) — Keras Team (2024)
  Keras `LSTM`层的官方文档，详细说明了其参数和在Keras框架中实际实现的用法。

---

[上一节](04-%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E9%97%AE%E9%A2%98.md) · [下一节](06-Keras%E4%B8%AD%E7%9A%84LSTM%E5%B1%82.md)
