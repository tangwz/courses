# 回顾：循环神经网络 (RNN)

来源：[原文](https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-1-sequence-modeling-attention-fundamentals/recap-rnns)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在我们了解驱动Transformer模型的注意力机制 (attention mechanism)之前，了解序列数据传统上是如何处理的是有益的。对于翻译句子或预测文本中下一个词这样的任务，元素的顺序非常重要。前馈神经网络 (neural network)独立处理输入，天生不适合捕获这些序列依赖。这促成了循环神经网络 (RNN) 的出现。

RNN的标志性特点是其内部的“记忆”或*隐藏状态*。与标准前馈网络不同，RNN逐个元素（例如，逐词）处理序列。在每一步，它接收当前输入元素和上一步的隐藏状态，以计算新的隐藏状态。这个隐藏状态就像是目前为止序列中已见信息的持续摘要。

可以将其想象成阅读句子：你逐个词处理，任何时候的理解都取决于你已经读过的词。RNN的隐藏状态试图捕捉这种变化的语境。

### 循环结构

其核心是，RNN单元在每个时间步执行相同的计算，但其内部状态根据输入序列而变化。如果我们有一个输入序列 $x = (x_1, x_2, ..., x_T)$，RNN在时间步 $t$ 更新其隐藏状态 $h_t$，使用当前输入 $x_t$ 和前一个隐藏状态 $h_{t-1}$。这种更新的常见公式是：


$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$


在这里：

- $h_t$ 是时间步 $t$ 的新隐藏状态。
- $h_{t-1}$ 是前一个时间步的隐藏状态（$h_0$ 通常初始化为零）。
- $x_t$ 是时间步 $t$ 的输入向量 (vector)。
- $W_{hh}$ 和 $W_{xh}$ 分别是隐藏到隐藏连接和输入到隐藏连接的权重 (weight)矩阵。这些权重在所有时间步中是*共享*的，这是RNN学习序列模式的根本。
- $b_h$ 是一个偏置 (bias)向量。
- $\tanh$ 是一种常见的激活函数 (activation function)（双曲正切），引入非线性。

RNN还可以选择在每个时间步生成输出 $y_t$，通常根据隐藏状态计算：


$$
y_t = W_{hy} h_t + b_y
$$


这里 $W_{hy}$ 是隐藏到输出的权重矩阵，$b_y$ 是输出偏置。是否在每一步都需要输出取决于具体的任务（例如，在每一步预测下一个词，或对整个序列进行分类）。

### 过程可视化

通过时间“展开”RNN通常更容易理解其运作。想象为每个时间步创建一个独立的网络副本，隐藏状态从一个副本传递到下一个。

> 一个RNN随时间展开的示意图。相同的RNN单元（权重 (weight)共享）处理输入 $x_t$ 和前一个隐藏状态 $h_{t-1}$，生成新的隐藏状态 $h_t$ 和输出 $y_t$。隐藏状态从一个时间步流向下一个时间步。

这种循环结构使得RNN理论上能够模拟序列中任意长度的依赖关系，因为信息可以通过隐藏状态传播。多年来，RNN（以及其更复杂的变体，如LSTMs和GRUs，它们旨在更好地处理长距离依赖）是序列建模任务的常规选择。

然而，正如我们将在下一节中看到，这样纯粹地按顺序处理序列会带来其自身的一些重要挑战，尤其是在处理非常长的序列或复杂依赖关系时。这些局限性促使了替代架构的出现，最终促成了Transformer。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10
  一本全面的教材，其中有一章专门介绍循环神经网络的基础知识、架构和数学公式。
- [CS224n: Natural Language Processing with Deep Learning, Lecture Notes on Recurrent Neural Networks](https://web.stanford.edu/class/cs224n/materials/G_RNNs.pdf) — Stanford University (2019)
  来自一门备受推崇的大学课程的教育材料，对循环神经网络，尤其是在自然语言处理方面的应用，提供了易懂而严谨的介绍。
- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Christopher Olah (2015)
  一篇有影响力且视觉直观的博客文章，它清晰地解释了循环神经网络，作为理解长短期记忆网络的前导知识。

---

[上一节](01-%E5%BA%8F%E5%88%97%E5%88%B0%E5%BA%8F%E5%88%97%E4%BB%BB%E5%8A%A1%E7%9A%84%E6%8C%91%E6%88%98.md) · [下一节](03-%E4%BC%A0%E7%BB%9F%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%96%B9%E6%B3%95%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
