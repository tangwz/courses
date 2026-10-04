# 循环神经网络（RNN）基础

来源：[原文](https://apxml.com/zh/courses/nlp-fundamentals/chapter-5-nlp-sequence-models-intro/rnn-basics)

[返回章节目录](README.md) · [返回课程目录](../README.md)

词袋模型（Bag-of-Words）和TF-IDF等方法生成的文本表示会丢失词语固有的顺序。虽然这对于某些任务很适用，但许多自然语言处理问题非常依赖对序列的理解。例如，以情感分析（“这部电影不好看”）对比（“这部电影还不错，但没那么好”）为例，或考虑机器翻译，词序对意义非常重要。为处理这类序列信息，需要专门为有序数据设计的模型。这就引出了循环神经网络 (neural network)（RNN）。

循环神经网络的核心思想是**循环**（recurrence）：对序列的每个元素执行相同的任务，其中每个元素的输出取决于前面元素的计算结果。这就像阅读句子；你对当前词语的理解受到你已经看到的词语的影响。RNN通过保持一个内部**状态**或**记忆**（$h_t$）来模拟这一点，该状态总结了序列中到目前为止已处理的信息。

### 循环连接

与独立处理固定大小输入的标准前馈网络不同，RNN具有一个循环。这个循环使得信息能够从网络的一个步骤保留到下一个步骤。在每个时间步$t$，RNN接受当前步骤的输入（$x_t$）和前一步骤的隐藏状态（$h_{t-1}$）来计算新的隐藏状态（$h_t$）。这个新状态随后作为处理序列中下一个元素的记忆。

> 一个RNN单元的图示，显示了输入$x_t$、根据$x_t$和前一个隐藏状态$h_{t-1}$计算的隐藏状态$h_t$，以及输出$y_t$。该循环表示递归。

### 按时间展开网络

为更好地理解RNN如何处理序列，将其按时间“展开”或“铺开”可视化会很有帮助。想象为序列中的每个时间步复制一个网络副本。一个时间步的隐藏状态会传递到下一个时间步。重要的是，所有时间步都使用相同的参数 (parameter)集（权重 (weight)和偏置 (bias)）。这种参数共享使得RNN高效，并能应用于不同长度的序列。

考虑处理一个长度为$T$的序列：$x_1, x_2, ..., x_T$。展开后的网络会是这样的：

> 一个RNN按时间步$1, 2, ..., T$展开的示意图。蓝色箭头显示了隐藏状态（$h_t$）从一个时间步到下一个时间步的传递。请注意，RNN单元块代表在每个步骤中应用相同的一组权重。$h_0$是初始隐藏状态，通常设为零。

### 数学表述

让我们对一个简单RNN单元内部的计算进行正式说明。在每个时间步$t$：

1. **计算新的隐藏状态（$h_t$）：** 这通常使用当前输入（$x_t$）和前一个隐藏状态（$h_{t-1}$）来完成。会应用一个激活函数 (activation function)（通常是`tanh`或`ReLU`）。

   
   $$
   h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
   $$
   

   其中：

   - $h_t$ 是时间$t$的隐藏状态向量 (vector)。
   - $h_{t-1}$ 是前一个时间步的隐藏状态向量。
   - $x_t$ 是时间$t$的输入向量。
   - $W_{hh}$ 是循环隐藏状态连接的权重 (weight)矩阵。
   - $W_{xh}$ 是输入到隐藏连接的权重矩阵。
   - $b_h$ 是隐藏状态计算的偏置 (bias)向量。
   - $\tanh$ 是双曲正切激活函数，将值压缩在-1到1之间。
2. **计算输出（$y_t$）：** 时间步$t$的输出通常根据隐藏状态$h_t$计算。具体的计算方法和激活函数取决于任务（例如，分类任务使用softmax）。

   
   $$
   y_t = W_{hy} h_t + b_y
   $$
   

   其中：

   - $y_t$ 是时间$t$的输出向量。
   - $W_{hy}$ 是隐藏到输出连接的权重矩阵。
   - $b_y$ 是输出计算的偏置向量。
   - （根据应用，之后可能会对$y_t$应用softmax等激活函数）。

网络在训练期间学习权重矩阵（$W_{hh}, W_{xh}, W_{hy}$）和偏置向量（$b_h, b_y$），通常使用一种称为“时间反向传播 (backpropagation)”（Backpropagation Through Time, BPTT）的反向传播变体。

### 实际序列处理

想象将句子“RNNs process sequences”输入到一个RNN中，可能一次输入一个词（或其嵌入 (embedding)）。

1. **时间 t=1：** 输入是“RNNs”（$x_1$）。网络使用初始隐藏状态$h_0$（通常为零）和$x_1$来计算$h_1$。此时也可能生成一个输出$y_1$。现在$h_1$包含了一些从“RNNs”获取的信息。
2. **时间 t=2：** 输入是“process”（$x_2$）。网络使用$h_1$（“RNNs”的记忆）和$x_2$来计算$h_2$。现在$h_2$包含了来自“RNNs”和“process”的信息。
3. **时间 t=3：** 输入是“sequences”（$x_3$）。网络使用$h_2$（“RNNs process”的记忆）和$x_3$来计算$h_3$。$h_3$表示处理完整个序列后的状态。

最终隐藏状态（本例中为$h_3$）或每个步骤的输出（$y_1, y_2, y_3$）可以用于各种任务。例如，$h_3$可以输入到分类器中进行整句情感分析，或者输出$y_t$可以表示每个步骤中下一个词的预测。

RNN为自然语言处理中序列数据建模提供了基本架构。它们保持状态的能力使其能够捕获序列中元素之间的依赖关系，克服了简单模型的一个主要局限。然而，正如我们将在下一节看到的，基本RNN在处理序列中长距离依赖时面临挑战。

## 参考资料

- [Backpropagation Through Time: What It Does and How to Do It](https://doi.org/10.1109/5.58337) — Paul J. Werbos (1990)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 78; Pages: 1550-1560; DOI: [10.1109/5.58337](https://doi.org/10.1109/5.58337)
  这篇基础论文介绍并解释了时间反向传播（BPTT）算法，该算法对于训练循环神经网络至关重要。
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Pages: Chapter 9
  在自然语言处理的背景下，全面介绍了循环神经网络，包括其动机、架构和应用。(第三版草案)
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10
  这是一本权威的教科书，为循环神经网络提供了严谨的理论基础，包括其结构、数学公式和训练方法。
- [CS224N: Natural Language Processing with Deep Learning - Lecture Notes on RNNs](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEin-uP6Ae1hGCGSo66QRTKjCVk9Lc4rm9rHeZJZth5ZA8B_ukV9gwHMmJbUk5hfmvwQrul29TIs9AagMSMEc9_B5Oo6roD-Cax5K98JNhvn52lOHnlqw7eaIoupFGSl97pGwxEquJYmVgKzQ4ND5PeJpCnyjTHmKP59Mau) — Milad Mohammadi, Rohit Mundra, Richard Socher, Lisa Wang, Amita Kamath (2019)
  Pages: Lecture 5
  提供清晰简洁的讲义，涵盖循环神经网络的基础知识、架构及其在自然语言处理中的应用。(2023年冬季学期)

---

[上一节](01-%E5%BA%8F%E5%88%97%E6%84%9F%E7%9F%A5%E7%9A%84%E5%BF%85%E8%A6%81%E6%80%A7.md) · [下一节](03-%E7%90%86%E8%A7%A3%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E9%97%AE%E9%A2%98.md)
