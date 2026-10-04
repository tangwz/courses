---
course: "introduction-to-neural-networks"
chapter: "building-first-neural-network"
lesson: "initializing-parameters"
sourceId: 1984
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-5-building-first-neural-network/initializing-parameters"
title: "初始化权重与偏置"
description: "讨论初始化网络参数的常见策略。"
order: 2
plots: []
sourceHash: "a11292c42e8a6e5934576b99609e10a9c80abf05231e5fd1bc9149f62be0ad7c"
sourceCorrections: []
---

为神经网络 (neural network)的参数 (parameter)——权重 (weight)（$W$）和偏置 (bias)（$b$）——设置初始值，是其结构（包括网络层、每层神经元数量和激活函数 (activation function)）确定后的一个基本过程。这看似是个小细节，但初始化这些参数的方式可以很大程度上影响网络能否有效训练或完全无法学习。初始化确定了高维损失函数 (loss function)曲面中的起点，梯度下降 (gradient descent)将从此起点开始寻找最小值。一个糟糕的起点可能导致收敛缓慢、陷入不佳的局部最小值，或在训练过程中出现数值不稳定。

### 为何不从简单的开始？零值或常数初始化的问题

最简单的方法似乎是将所有权重 (weight)和偏置 (bias)初始化为零。我们来考虑一下为什么这会带来问题。如果连接到隐藏层的所有权重（$W$）都为零，那么在第一次前向传播时，该层中的每个神经元都将计算出完全相同的输出，无论输入是什么。这是因为线性变换（$z = Wx + b$）将为该层中的所有神经元产生相同的值（如果偏置非零，则仅为偏置值；如果偏置也为零，则为零）。

因此，当应用激活函数 (activation function)，例如 $a = g(z)$ 时，该层中的所有神经元仍然产生相同的激活值。在反向传播 (backpropagation)过程中，计算出的梯度（$\frac{\partial L}{\partial W}$）对于连接到该层的所有权重也将是相同的。这意味着在权重更新步骤（$W = W - \alpha \frac{\partial L}{\partial W}$）中，所有权重将以相同的量更新。对称性从未被打破；该层中的所有神经元在整个训练过程中都保持相同，有效地将该层的容量降至单个神经元。将所有权重初始化为相同的非零常数值也存在完全相同的对称性问题。

将偏置初始化为零通常可以接受且非常普遍，特别是当权重随机初始化时。然而，将*所有*参数 (parameter)初始化为零会阻止网络有效学习。

### 打破对称性：小随机值

为了克服对称性问题，我们需要用不同的值初始化权重 (weight)。标准方法是随机初始化权重，通常从概率分布中抽取值。一个常见的起点是使用从以下分布中抽取的小随机数：

1. **高斯（正态）分布：** 从均值为0、标准差较小（例如0.01）的高斯分布中采样权重。$W \sim \mathcal{N}(0, \sigma^2)$，其中 $\sigma$ 较小。
2. **均匀分布：** 从以零为中心的小范围（例如 $U[-r, +r]$，其中 $r$ 较小，如0.01）的均匀分布中采样权重。

为什么要使用小值？如果权重过大，Sigmoid或Tanh等激活函数 (activation function)的输入（$z = Wx + b$）也可能变得非常大（正或负）。这些激活函数对于大输入会饱和（趋于平坦），这意味着它们的导数接近于零。在反向传播 (backpropagation)过程中，这些接近零的梯度导致前一层权重的更新变得极其微小，有效地阻止了这些层中的学习。这就是所谓的**梯度消失问题**。尽管用小随机数初始化可以打破对称性并避免立即饱和，但对于更深的网络可能不是最佳选择。

### 更智能的初始化：Xavier/Glorot和He方法

随着网络变得更深，确保信号（前向传播期间的激活值）和梯度（反向传播 (backpropagation)期间的梯度）在所有层中正确流动变得更加重要。如果激活值或梯度层层递减或递增，训练可能会变得不稳定或停滞。Xavier/Glorot和He等初始化方法是专门为解决此问题而开发的。它们旨在使激活值和梯度的方差在各层之间大致保持不变。

这些方法根据神经元或层的输入连接数（扇入， $n_{in}$）和输出连接数（扇出， $n_{out}$）来缩放初始随机权重 (weight)。

> 神经元 $j$ 从上一层中的 $n$ 个神经元接收输入连接（扇入），并向下一层中的 $m$ 个神经元发送输出连接（扇出）。Xavier和He等初始化策略使用这些值来缩放初始权重。

#### Xavier / Glorot 初始化

由Glorot和Bengio（2010）提出，这种方法适用于Tanh和Sigmoid等对称激活函数 (activation function)。它旨在通过基于扇入和扇出缩放权重，使激活值的方差在各层之间保持一致。

- **正态初始化：** 从 $\mathcal{N}(0, \sigma^2)$ 中采样权重，其中 $\sigma^2 = \frac{2}{n_{in} + n_{out}}$。
- **均匀初始化：** 从 $U[-r, +r]$ 中采样权重，其中 $r = \sqrt{\frac{6}{n_{in} + n_{out}}}$。

#### He 初始化

由He等人（2015）提出，这种方法是专门为ReLU激活函数及其变体（如Leaky ReLU）设计的。ReLU不对称，实际上会“杀死”一半的激活值（对负输入输出零）。He初始化通过仅考虑扇入来解决此问题，导致初始权重略大于Xavier，这有助于抵消ReLU引起的方差减小。

- **正态初始化：** 从 $\mathcal{N}(0, \sigma^2)$ 中采样权重，其中 $\sigma^2 = \frac{2}{n_{in}}$。
- **均匀初始化：** 从 $U[-r, +r]$ 中采样权重，其中 $r = \sqrt{\frac{6}{n_{in}}}$。

### 实用指南与偏置 (bias)初始化

- **经验法则：** 使用ReLU或其变体时，请使用He初始化。使用Tanh或Sigmoid激活函数 (activation function)时，请使用Xavier/Glorot初始化。
- **框架默认设置：** 现代深度学习 (deep learning)框架，如TensorFlow和PyTorch，通常将其密集/线性层的默认初始化器实现为这些精密的初始化器（或略有变体），从而使其易于使用。
- **偏置初始化：** 偏置通常初始化为零。有时，特别是使用ReLU激活时，将偏置初始化为小的正常量（例如0.01或0.1）可能是有益的。这确保了ReLU单元最初接收到略微正的输入，使其在早期更有可能被激活并参与学习。

选择正确的初始化策略是为网络成功训练做准备的重要一步。虽然简单的随机初始化可能适用于浅层网络，但Xavier/Glorot和He等方法提供了一种更具原则性的方法，有助于维持信号传播和梯度流动，特别是在更深层次的架构中。随着网络架构的确定和参数 (parameter)的初始化，我们已准备好开始核心训练过程，从前向传播开始，使用这些初始权重 (weight)和偏置生成预测。

## 参考资料

- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Xavier Glorot, Yoshua Bengio (2010)
  Journal: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics; Publisher: PMLR; Volume: 9; Pages: 249-256
  介绍了Xavier（Glorot）初始化方法，以稳定使用sigmoid/tanh激活函数的深度网络中的信号传播。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，全面讨论了神经网络初始化，包括理论和实践方面。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Lecture Notes on Neural Networks Part 3: Learning and Evaluation](https://cs231n.github.io/neural-networks-3/) — Andrej Karpathy, Justin Johnson, and Fei-Fei Li (2023)
  Publisher: Stanford University
  为神经网络中的权重初始化提供了清晰的解释和实践考量。
