---
course: "introduction-to-deep-learning"
chapter: "intro-specialized-architectures"
lesson: "rnn-recurrence-hidden-state"
sourceId: 5077
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-7-intro-specialized-architectures/rnn-recurrence-hidden-state"
title: "循环与隐藏状态"
description: "描述RNN如何使用循环和隐藏状态逐个处理序列。"
order: 7
plots: []
sourceHash: "0258d7fb8a1e8e686dbc5ad0afc13cbf597fa26777b04382f892337358a15d5f"
sourceCorrections: []
---

前馈网络，包括多层感知机（MLP），独立处理输入。如果你两次输入相同的MLP，它会产生相同的输出，不知晓任何之前的交互。这适用于许多任务，但对于顺序很重要的数据呢？考虑预测句子中的下一个词、分析股市趋势或语音转录。含义或预测通常很大程度上依赖于之前的内容。前馈网络缺乏记住序列中过去信息的固有机制。

循环神经网络 (neural network)（RNN）专门设计来处理这类序列信息。其核心原理是**循环**：逐个处理序列元素，同时保持内部**记忆**，通常称为**隐藏状态**。

设想阅读一个句子。你不会孤立地处理每个词。你对当前词的理解受已读词语的影响。RNN模拟了这一过程。在每一步（例如，句子中的每个词或时间序列中的每个点），RNN基于两项内容进行计算：

1. 当前输入元素（在时间步$t$的$x_t$）。
2. 来自上一步的隐藏状态（$h_{t-1}$）。

此计算生成一个新的隐藏状态（$h_t$），它捕获了当前输入的信息*以及*过去的有关上下文 (context)。这个隐藏状态$h_t$随后传递到下一个时间步（$t+1$），作为网络目前所见内容的记忆。

这种“循环”机制，即一步的输出（通过隐藏状态）反馈到下一步的输入中，是使网络**循环**的原因。重要的一点是，在*每个*时间步都使用相同的权重 (weight)和偏置 (bias)进行计算。这种参数 (parameter)共享使得RNN高效，并使它们能够泛化不同长度序列中不同位置的模式。

我们可以用数学方式表示时间步$t$的隐藏状态更新。一种常用形式是使用激活函数 (activation function)（如`tanh`或`ReLU`），应用于当前输入和前一个隐藏状态的组合：


$$
h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$


其中：

- $h_t$ 是当前时间步 $t$ 的隐藏状态。
- $h_{t-1}$ 是前一时间步 $t-1$ 的隐藏状态。
- $x_t$ 是当前时间步 $t$ 的输入。
- $W_{hh}$ 是应用于前一隐藏状态的权重矩阵。
- $W_{xh}$ 是应用于当前输入的权重矩阵。
- $b_h$ 是隐藏状态计算的偏置项。
- $f$ 是一个非线性激活函数（例如，`tanh`）。

网络在每个时间步也可能产生输出$y_t$，通常基于当前的隐藏状态计算：


$$
y_t = g(W_{hy} h_t + b_y)
$$


其中：

- $y_t$ 是时间步 $t$ 的输出。
- $W_{hy}$ 是连接隐藏状态到输出的权重矩阵。
- $b_y$ 是输出偏置项。
- $g$ 是另一个激活函数（例如，用于分类的`softmax`）。

从视觉上看，我们可以将循环随时间“展开”。下图展示了一个展开为三个时间步的简单RNN。注意隐藏状态$h$如何从一步传递到下一步，沿着序列携带信息。

> 一个RNN单元随时间展开。隐藏状态`h`作为记忆，将信息从一个时间步传递到下一个时间步。相同的权重（`W_hh`、`W_xh`、`W_hy`）应用于每个时间步。

这种循环结构，以演进的隐藏状态为核心，使RNN能够捕获序列中元素之间的依赖关系，使它们适用于涉及自然语言、时间序列数据及其他上下文重要的有序输入任务。

## 参考资料

- [Finding structure in time](https://doi.org/10.1207/s15516709cog1402_1) — Jeffrey L. Elman (1990)
  Journal: Cognitive Science; Volume: 14; Pages: 179-211; DOI: [10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1)
  一篇开创性论文，介绍了简单循环神经网络（通常称为Elman网络），确立了使用“上下文层”（隐藏状态）处理序列信息的概念。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本内容全面的教科书，提供了深度学习的详细理论和数学基础，其中包括专门介绍循环神经网络及其机制的章节。
- [CS224n: Natural Language Processing with Deep Learning - Lecture 6: Recurrent Neural Networks (RNNs) and Backpropagation Through Time (BPTT)](http://web.stanford.edu/class/cs224n/slides/cs224n-spr2024-lecture06-fancy-rnn.pdf) — Stanford University, CS224n Instructors (2023)
  一门领先大学课程的官方讲义，提供了循环神经网络、其架构和操作原理的教学性清晰解释。
