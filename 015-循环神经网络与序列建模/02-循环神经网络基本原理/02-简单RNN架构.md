# 简单RNN架构

来源：[原文](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-2-rnn-fundamentals/simple-rnn-architecture)

[返回章节目录](README.md) · [返回课程目录](../README.md)

循环神经网络 (neural network) (RNN)逐步处理序列信息，同时保持内部记忆。实现此功能的基本构件是简单RNN单元。

可将RNN单元视为在序列中每个时间步运行的计算引擎。对于输入序列中的每个元素 $x_t$（在时间步 $t$），该单元接收两个输入：

1. 当前输入元素 $x_t$。
2. 来自上一时间步的隐藏状态 $h_{t-1}$。

利用这些输入，该单元执行计算以生成两个输出：

1. 新的隐藏状态 $h_t$。此状态捕获当前输入 $x_t$ 的信息，*以及*来自所有先前步骤的相关上下文 (context)（概括于 $h_{t-1}$ 中）。它作为网络的记忆，并传递给下一时间步。
2. 可选输出 $y_t$。这是网络专门为当前时间步 $t$ 生成的预测或结果。是否在每一步都产生输出取决于具体任务（例如，预测下一个单词每步需要一个输出，而情感分类可能只需要在整个序列处理完后得到一个输出）。

### RNN单元内部

简单RNN单元内的核心计算包含结合当前输入和先前隐藏状态，使用学习到的权重 (weight)和激活函数 (activation function)。本章引言展示了基本方程：

$h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$
$y_t = g(W_{hy}h_t + b_y)$

我们来逐一分析：

- **$W_{xh}$**: 一个将输入 $x_t$ 进行转换的权重矩阵。
- **$W_{hh}$**: 一个将先前隐藏状态 $h_{t-1}$ 进行转换的权重矩阵。这是*循环*权重矩阵，捕获过去信息的影响。
- **$b_h$**: 一个添加到隐藏状态计算中的偏置 (bias)向量 (vector)。
- **$f$**: 隐藏状态的激活函数。这里的常见选择是双曲正切 (`tanh`)，它将值压缩到 [-1, 1] 的范围。这有助于调节信息流并减轻一些梯度问题（尽管不能完全解决，正如我们稍后将了解的）。
- **$W_{hy}$**: 一个将当前隐藏状态 $h_t$ 进行转换以产生输出 $y_t$ 的权重矩阵。
- **$b_y$**: 一个添加到输出计算中的偏置向量。
- **$g$**: 输出 $y_t$ 的激活函数。$g$ 的选择取决于任务（例如，分类使用 `softmax`，回归使用线性函数）。

此架构的一个重要特点是**权重共享**。相同的权重矩阵 ($W_{xh}$, $W_{hh}$, $W_{hy}$) 和偏置 ($b_h$, $b_y$) 在*每个时间步*都使用。这意味着网络学习一套参数 (parameter)来明确如何处理输入元素并更新其记忆，无论该元素在序列中的位置如何。这使得RNN在参数上具有高效性，并且能够泛化处理不同长度的序列。

### 单元和循环的可视化

我们可以可视化单个RNN单元及其连接：

> 单个RNN单元处理输入 $x_t$ 和先前状态 $h_{t-1}$，以计算得到新状态 $h_t$ 和输出 $y_t$。权重 (weight)矩阵 ($W_{xh}$, $W_{hh}$, $W_{hy}$) 在这些转换过程中应用。

真正的作用体现在将这些单元链接起来，形成循环连接。在时间步 $t$ 计算得到的隐藏状态 $h_t$ 成为时间步 $t+1$ 单元的输入 $h_{t-1}$。这种时间上的“展开”使得信息能够流经序列：

> 一个RNN在三个时间步上展开。每个时间步由单元计算得到的隐藏状态 ($h_t$) 作为输入传递给下一时间步 ($t+1$) 的单元。每个时间步都使用相同的单元参数 (parameter)（权重 $W$ 和偏置 (bias) $b$）。

这种展开视图很有帮助，尤其是在思考梯度在训练期间如何反向流动（时间反向传播 (backpropagation)）时，我们将在接下来讨论这个。然而，请记住，在实际应用中，是*相同*的单元结构（具有相同的权重）被重复应用，而不是每个时间步都有不同的副本。这种结构使得RNN能够使用固定数量的参数处理任意长度的序列。

## 参考资料

- [Finding Structure in Time](https://doi.org/10.1207/s15516709cog1402_1) — Jeffrey L. Elman (1990)
  Journal: Cognitive Science; Publisher: Wiley; Volume: 14; Pages: 179-211; DOI: [10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1)
  一篇基础性论文，介绍了简单循环网络（通常称为“Elman网络”）及其学习时间依赖性的能力。
- [Deep Learning](https://www.deeplearningbook.org/contents/rnn.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10
  第10章详细阐述了循环神经网络，包括基本的RNN架构和其运作方式。
- [CS224N: Natural Language Processing with Deep Learning, Lecture 4: Recurrent Neural Networks](https://www.youtube.com/watch?v=kYJqG5hNn58) — Christopher Manning (2024)
  Journal: Stanford University Course Material; Publisher: Stanford Online
  作为一门备受推崇的课程的一部分，它清晰简洁地解释了循环神经网络，包括其架构、展开和基本计算。

---

[上一节](01-%E6%A0%B8%E5%BF%83%E6%80%9D%E6%83%B3%EF%BC%9A%E8%BF%AD%E4%BB%A3%E5%A4%84%E7%90%86%E5%BA%8F%E5%88%97.md) · [下一节](03-%E9%9A%90%E8%97%8F%E7%8A%B6%E6%80%81%E7%9A%84%E4%BD%9C%E7%94%A8.md)
