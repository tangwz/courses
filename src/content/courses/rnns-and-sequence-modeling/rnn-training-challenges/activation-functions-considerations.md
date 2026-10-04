---
course: "rnns-and-sequence-modeling"
chapter: "rnn-training-challenges"
lesson: "activation-functions-considerations"
sourceId: 2562
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-4-rnn-training-challenges/activation-functions-considerations"
title: "激活函数考量"
description: "回顾RNN中常用的激活函数（例如，tanh，ReLU）及其特性。"
order: 6
plots: ["plots/2562-0.json"]
sourceHash: "fc8f6d68aec390c409a371e32b1ae57291e44f7f3209d329ff3fd20b66b5951a"
sourceCorrections: []
---

激活函数 (activation function)为神经网络 (neural network)引入非线性，使其能够模拟复杂模式。在循环神经网络 (RNN)中，这个作用尤为重要，因为激活函数在网络处理序列时会在每个时间步重复应用。这种重复应用直接影响隐藏状态的动态变化，最重要的是，也影响通过时间反向传播 (backpropagation)（BPTT）过程中梯度的流动。激活函数的选择可以缓解或加剧梯度消失和梯度爆炸问题。

### 激活函数 (activation function)如何影响RNN训练

回顾基本的RNN状态更新方程：


$$
h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$


其中，$h_t$ 是时间步 $t$ 的隐藏状态，$x_t$ 是输入，$W_{hh}$ 和 $W_{xh}$ 是权重 (weight)矩阵，$b_h$ 是偏置 (bias)，$f$ 是激活函数。在BPTT过程中，损失对早期状态 $h_k$ (其中 $k < t$) 的梯度涉及将梯度通过每个中间步骤相乘。这包括重复乘以激活函数 $f'$ 的导数，该导数在每个步骤的预激活值处求得。

如果 $f'$ 持续很小（小于1），梯度在向后传播时会指数级缩小，导致梯度消失。反之，如果 $f'$ 可以很大，并结合较大的权重值，梯度可能会指数级增长，导致梯度爆炸。

下面我们来分析循环神经网络 (neural network) (RNN)中常用激活函数的特性。

### RNN中常用的激活函数 (activation function)

1. **双曲正切 (tanh)**
   $\tanh$ 函数常在简单RNN中用作隐藏状态的主要激活函数，在LSTM和GRU单元中也普遍存在。

   
   $$
   \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
   $$
   

   它的输出范围是 $(-1, 1)$，并且是零中心化的。相较于非零中心化的函数，这个特性有利于优化。$\tanh$ 的导数是：

   
   $$
   \frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)
   $$
   

   导数的最大值是1（在 $z=0$ 处），当 $|z|$ 增加时它接近0（饱和）。尽管在零附近的梯度比sigmoid函数大，但饱和特性仍使基于$\tanh$ 的RNN容易受到梯度消失问题的影响，尤其是在很多时间步上。不过，它的有界输出有助于阻止隐藏状态值本身变得过大。
2. **Sigmoid (逻辑)**
   Sigmoid 函数将其输入压缩到范围 $(0, 1)$。

   
   $$
   \sigma(z) = \frac{1}{1 + e^{-z}}
   $$
   

   它的导数是：

   
   $$
   \frac{d}{dz}\sigma(z) = \sigma(z)(1 - \sigma(z))
   $$
   

   Sigmoid 导数的最大值是 0.25（在 $z=0$ 处）。因为这个最大值明显小于1，在BPTT期间的重复乘法导致梯度很快消失。因此，在现代RNN中，sigmoid 很少用作隐藏状态更新的主要激活函数。它目前主要用于LSTM和GRU中的门控机制，其 $(0, 1)$ 的输出范围非常适合控制信息流（例如，决定遗忘或传递多少信息）。
3. **修正线性单元 (ReLU)**
   ReLU 在前馈网络和卷积神经网络 (neural network)（CNN）中是一种流行的选择。

   
   $$
   \text{ReLU}(z) = \max(0, z)
   $$
   

   它的导数很简单：当 $z > 0$ 时为1，当 $z \le 0$ 时为0。

   - **优点：** 对于正输入，梯度恒定为1，这有助于解决梯度消失问题。它计算效率也高。
   - **缺点：**
     - **梯度爆炸：** 由于ReLU对正输入无界，在RNN中重复应用并结合较大权重 (weight)可能导致激活值和梯度爆炸。在循环设置中使用ReLU时，梯度裁剪变得尤为重要。
     - **死亡ReLU：** 如果神经元的输入持续低于零，它们可能变得不活跃，导致梯度为零，并有效停止了该神经元的学习。
     - **非零中心化：** 输出始终是非负的。

   尽管在简单RNN中ReLU不如$\tanh$ 常见作为主要的隐藏状态激活函数，但ReLU及其变体（如Leaky ReLU或Parametric ReLU）有时也会使用，特别是在LSTM/GRU的更新机制中，或与仔细的初始化和正则化 (regularization)结合使用。

### 激活函数 (activation function)及其导数的可视化

下面的图表显示了$\tanh$、sigmoid和ReLU函数的形状和导数。请注意输出范围以及导数何时变得很小或为零。



![常用激活函数及其导数](plots/2562-0.json)



> $\tanh$、sigmoid和ReLU激活函数及其一阶导数的比较。请注意$\tanh$ 和sigmoid如何饱和（导数趋近于零），而ReLU的导数对正输入保持不变。

### RNN激活函数 (activation function)的选择

- **Tanh：** 在简单RNN中，它常是主要隐藏状态激活的默认选择，也用于LSTM/GRU的内部计算。其零中心化输出和有界范围提供一定的稳定性。然而，它仍然存在饱和问题，导致深层序列中的梯度消失。
- **ReLU：** 由于其对正输入的非饱和特性，可以缓解梯度消失。然而，它需要仔细处理，因为它可能出现梯度爆炸（需要梯度裁剪）和死亡ReLU问题。它可能会被选择性使用，可能进行Leaky ReLU等修改。
- **Sigmoid：** 通常避免将其用作主要循环激活函数，因为它有导致梯度消失的强烈倾向。其主要用途在于门控机制（我们将在LSTM和GRU中看到），其 $(0, 1)$ 的输出在功能上适合控制信号流。

激活函数的选择与模型设计和训练的其他方面相互关联，包括权重 (weight)初始化（前面已讨论）和使用的特定架构。在简单RNN中，使用这些标准激活函数管理梯度流的固有困难是推动发展更复杂的循环架构（如LSTM和GRU）的重要原因，这些架构包含明确的机制（门，通常使用sigmoid和tanh）以更好地控制长序列上的信息和梯度流。我们将在接下来的章节中分析这些高级架构。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: The MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了长短期记忆（LSTM）网络，该网络通过使用 sigmoid 和 tanh 激活函数的特定门控机制来缓解梯度消失问题。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于激活函数、循环神经网络和梯度流动态的权威教科书，提供了坚实的理论基础。
