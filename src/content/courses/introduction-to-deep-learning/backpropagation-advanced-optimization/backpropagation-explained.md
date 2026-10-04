---
course: "introduction-to-deep-learning"
chapter: "backpropagation-advanced-optimization"
lesson: "backpropagation-explained"
sourceId: 5045
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-4-backpropagation-advanced-optimization/backpropagation-explained"
title: "反向传播算法详解"
description: "说明反向传播算法高效计算梯度的具体步骤。"
order: 3
plots: []
sourceHash: "c46a8c57c69105cd6a5b4e6d1b13a056dfee49c6d8685f733dee411608a6a6e3"
sourceCorrections: []
---

训练神经网络 (neural network)需要调整其权重 (weight)和偏置 (bias)，以最小化误差函数，通常使用梯度下降 (gradient descent)法。梯度下降法的基础要求是误差函数相对于网络中*每个*参数 (parameter)（权重 $w_{ij}^{(l)}$ 和偏置 $b_i^{(l)}$）的梯度。对于具有可能数百万参数的深度网络，直接计算这些梯度将导致计算成本高昂。

这时，**反向传播 (backpropagation)算法**便登场了。它本身并非一种新的优化算法，而是一种计算所需梯度的极其高效的方法。反向传播巧妙地应用了微积分的链式法则，通过网络层从最终误差计算反向进行，以确定每个参数如何对该误差作出贡献。

可以将其理解为对最终误差进行“归因”或“分配”。如果网络的输出与目标相去甚远，反向传播有助于确定哪些连接（权重）和阈值（偏置）对该差异负有主要责任，以及它们应该朝哪个方向调整。

### 反向传播 (backpropagation)的两个主要阶段

该算法主要通过两个阶段运行：

1. **前向传播：** 这是将输入 $x$ 逐层馈送到网络的标准过程，计算每层的加权和 ($z^{(l)}$) 和激活值 ($a^{(l)}$)，直到达到最终输出激活值 $a^{(L)}$ 并计算误差 $E$。在此阶段，存储中间值是必要的，具体来说，是每层 $l$ 的加权和 $z^{(l)}$ 和激活值 $a^{(l)}$，因为它们在反向传播阶段会用到。

   - $z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$
   - $a^{(l)} = g(z^{(l)})$ (其中 $g$ 是激活函数 (activation function))
2. **反向传播：** 这是反向传播的核心。它从输出层开始，将误差的梯度反向传播通过整个网络。

   - **输出层（第 L 层）：** 我们首先计算误差相对于输出激活值 $\frac{\partial E}{\partial a^{(L)}}$ 的梯度。其具体形式取决于所用的误差函数。然后，使用链式法则和存储的 $z^{(L)}$ 值，我们计算误差相对于输出层的预激活值 $z^{(L)}$ 的梯度。这个项通常被称为输出层的“误差” $\delta^{(L)}$：

     
     $$
     \delta^{(L)} = \frac{\partial E}{\partial z^{(L)}} = \frac{\partial E}{\partial a^{(L)}} \odot g'(z^{(L)})
     $$
     

     这里，$g'$ 是输出层使用的激活函数的导数，$\odot$ 表示元素级乘法（Hadamard 积）。这是因为 $\frac{\partial E}{\partial a^{(L)}}$ 和 $g'(z^{(L)})$ 都是与输出层维度相同的向量 (vector)（或张量）。
   - **输出层参数 (parameter)的梯度：** 一旦我们有了 $\delta^{(L)}$，我们就可以计算连接到输出层的权重 (weight) $W^{(L)}$ 和偏置 (bias) $b^{(L)}$ 的梯度：

     
     $$
     \frac{\partial E}{\partial W^{(L)}} = \delta^{(L)} (a^{(L-1)})^T
     $$
     
     
     $$
     \frac{\partial E}{\partial b^{(L)} } = \delta^{(L)}
     $$
     

     （注意：对于 $\frac{\partial E}{\partial W^{(L)}}$，如果 $\delta^{(L)}$ 和 $a^{(L-1)}$ 是向量，这是一个外积。如果使用小批量，则批次维度的求和是隐式的。）
   - **反向传播误差：** 这一步需要计算误差 $\delta^{(L)}$ 如何反向传播到前一层的激活值 $a^{(L-1)}$。我们需要 $\frac{\partial E}{\partial a^{(L-1)}}$ 来继续这个过程。这是通过考虑 $a^{(L-1)}$ 如何通过 $W^{(L)}$ 影响 $z^{(L)}$，以及 $z^{(L)}$ 如何通过 $\delta^{(L)}$ 影响 $E$ 来得到的：

     
     $$
     \frac{\partial E}{\partial a^{(L-1)}} = (W^{(L)})^T \delta^{(L)}
     $$
     
   - **隐藏层（第 l 层）：** 现在，我们将此过程推广到任何隐藏层 $l$（从 $L-1$ 向下到 1 反向进行）。我们假设已经计算了 $\frac{\partial E}{\partial a^{(l)}}$（这是从 $l+1$ 层传播过来的误差）。我们计算当前层的误差 $\delta^{(l)}$：

     
     $$
     \delta^{(l)} = \frac{\partial E}{\partial z^{(l)}} = \frac{\partial E}{\partial a^{(l)}} \odot g'(z^{(l)}) = \left( (W^{(l+1)})^T \delta^{(l+1)} \right) \odot g'(z^{(l)})
     $$
     

     这个方程表明了来自*下一*层（$\delta^{(l+1)}$）的误差，通过连接权重 ($W^{(l+1)}$) 加权后，与激活函数的局部梯度 ($g'(z^{(l)})$) 结合，从而形成*当前*层的误差。
   - **隐藏层参数的梯度：** 使用 $\delta^{(l)}$，我们计算第 $l$ 层参数的梯度：

     
     $$
     \frac{\partial E}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T
     $$
     
     
     $$
     \frac{\partial E}{\partial b^{(l)}} = \delta^{(l)}
     $$
     
   - **继续反向：** 我们重复这个过程，为从 $L-1$ 到 $1$ 的每一层 $l$ 计算 $\delta^{(l)}$、$\frac{\partial E}{\partial W^{(l)}}$ 和 $\frac{\partial E}{\partial b^{(l)}}$。每一步都使用在前一步（即为 $l+1$ 层计算的步骤）中计算出的误差 $\delta^{(l+1)}$。

### 流程图示

下图说明了梯度信息的反向流动。前向传播从左到右计算值。反向传播 (backpropagation)（红色虚线）从误差 $E$ 开始计算梯度，并从右到左移动，在每个阶段计算所需的偏导数（$\partial E / \partial \dots$）。

> 反向传播的流程。实线代表前向传播计算。虚线代表梯度的反向流动计算，从误差 (E) 开始，向左穿过各层。每个虚线边的标签表示在该步骤中计算的梯度。

### 效率提升

反向传播 (backpropagation)的效率来自于计算的复用。请注意，为第 $l$ 层计算的误差 $\delta^{(l)}$ 如何直接用于计算梯度 $\frac{\partial E}{\partial W^{(l)}}$、$\frac{\partial E}{\partial b^{(l)}}$ *以及*前一层所需的误差项 $\delta^{(l-1)}$（通过 $(W^{(l)})^T \delta^{(l)}$）。这避免了如果尝试使用链式法则从头开始独立计算每个 $\frac{\partial E}{\partial W_{ij}}$ 和 $\frac{\partial E}{\partial b_i}$ 直到输入层时可能出现的冗余计算。

本质上，反向传播是应用于神经网络 (neural network)的反向模式自动微分的一种实际应用。它提供了一种结构化的方式，用于计算标量目标函数（误差）相对于其所有输入参数 (parameter)（权重 (weight)和偏置 (bias)）的梯度，其时间复杂度大致与前向传播本身的计算成本成比例。

一旦反向传播阶段完成，我们就得到了所有层 $l$ 的梯度 $\frac{\partial E}{\partial W^{(l)}}$ 和 $\frac{\partial E}{\partial b^{(l)}}$。这些梯度随后被优化算法（如 SGD、Adam 等）用于更新网络的参数：

$W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial E}{\partial W^{(l)}}$
$b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial E}{\partial b^{(l)}}$

其中 $\eta$ 是学习率。这就完成了一个前向传播、误差计算、反向传播和参数更新的循环，构成了神经网络如何学习的基础机制。

## 参考资料

- [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) — David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams (1986)
  Journal: Nature; Publisher: Springer Nature; Volume: 323; Pages: 533-536; DOI: [10.1038/323533a0](https://doi.org/10.1038/323533a0)
  介绍反向传播算法的原始论文，使其在多层神经网络中得到广泛应用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 197-210
  深度学习综合教材，以数学严谨性介绍反向传播及其在深度学习中的作用。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html) — Michael Nielsen (2019)
  Publisher: Determination Press
  一本易于理解的在线书籍，逐步讲解反向传播算法。
- [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) — Stanford CS231n Course Staff (2019)
  Publisher: Stanford University
  课程讲义，提供反向传播的实用和直观解释，包括其推导过程。
