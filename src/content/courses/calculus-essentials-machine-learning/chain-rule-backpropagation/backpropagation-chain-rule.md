---
course: "calculus-essentials-machine-learning"
chapter: "chain-rule-backpropagation"
lesson: "backpropagation-chain-rule"
sourceId: 1416
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-5-chain-rule-backpropagation/backpropagation-chain-rule"
title: "反向传播：链式法则的应用"
description: "说明链式法则如何系统地反向应用于网络层以计算梯度。"
order: 4
plots: []
sourceHash: "6ad5d1bc15792268c4734f6f479c65451a8884245fde7da2fde71ecd33b5db6f"
sourceCorrections: []
---

我们将链式法则与训练神经网络 (neural network)的机制直接关联起来。正如我们之前讨论的，神经网络本质上是一个大型复合函数。最终输出（例如，分类分数或回归值）依赖于前一层的输出，而前一层的输出又依赖于再前一层的输出，依此类推，一直追溯到输入数据。重要的一点是，每个层计算的函数都包含权重 (weight)和偏置 (bias)，这些是我们需要的调整参数 (parameter)。

训练过程包括最小化一个损失函数 (loss function)，我们称之为 $L$，它衡量网络预测值与实际目标值之间的差异。为了使用梯度下降 (gradient descent)（或其变体）最小化 $L$，我们需要计算 $L$ 对网络中每个权重 $W$ 和偏置 $b$ 的偏导数：$\frac{\partial L}{\partial W}$ 和 $\frac{\partial L}{\partial b}$。考虑到这种嵌套结构，直接计算这些似乎很困难。此时，链式法则就变得非常有用。

反向传播 (backpropagation)不是一种新的优化算法；它是一种在神经网络中*计算梯度的高效算法*。它通过从损失函数*反向*工作，系统地应用多元链式法则来计算所有必需的偏导数（对于所有层 $l$、神经元 $i$ 和输入连接 $j$ 的 $\frac{\partial L}{\partial W_{ij}^{[l]}}$ 和 $\frac{\partial L}{\partial b_i^{[l]}}$）。

### 反向传播 (backpropagation)过程详解

设想前向传播过程：输入数据流经网络，层层递进，经历线性变换（乘以权重 (weight)，加上偏置 (bias)）和非线性激活，最终生成输出预测，然后是损失值 $L$。

反向传播则逆转这个流程：

1. **从末端开始：** 该过程始于计算损失 $L$ 对网络最终输出激活值 $a^{[L]}$（其中 $L$ 表示最后一层）的导数。这通常很简单，取决于所使用的具体损失函数 (loss function)和最终激活函数 (activation function)。我们将 $\frac{\partial L}{\partial a^{[L]}}$ 记为 $\delta a^{[L]}$。
2. **关于预激活值 ($z^{[L]}$) 的梯度：** 利用链式法则，我们找到损失对最后一层预激活线性输出 $z^{[L]}$ 的梯度。如果 $a^{[L]} = g(z^{[L]})$，其中 $g$ 是激活函数，那么：
   $\frac{\partial L}{\partial z^{[L]}} = \frac{\partial L}{\partial a^{[L]}} \cdot \frac{\partial a^{[L]}}{\partial z^{[L]}} = \delta a^{[L]} \cdot g'(z^{[L]})$
   我们将 $\frac{\partial L}{\partial z^{[L]}}$ 记为 $\delta z^{[L]}$。项 $g'(z^{[L]})$ 是激活函数在 $z^{[L]}$ 处求得的导数。
3. **关于参数 (parameter) ($W^{[L]}, b^{[L]}$) 的梯度：** 现在我们有了 $\delta z^{[L]}$，我们可以找到最后一层的权重 $W^{[L]}$ 和偏置 $b^{[L]}$ 的梯度。因为 $z^{[L]} = W^{[L]}a^{[L-1]} + b^{[L]}$：
   $\frac{\partial L}{\partial W^{[L]}} = \frac{\partial L}{\partial z^{[L]}} \cdot \frac{\partial z^{[L]}}{\partial W^{[L]}} = \delta z^{[L]} \cdot (a^{[L-1]})^T$
   $\frac{\partial L}{\partial b^{[L]}} = \frac{\partial L}{\partial z^{[L]}} \cdot \frac{\partial z^{[L]}}{\partial b^{[L]}} = \delta z^{[L]} \cdot 1 = \delta z^{[L]}$
   （注意：$\frac{\partial L}{\partial W^{[L]}}$ 的计算涉及矩阵/向量 (vector)运算，结果是一个与 $W^{[L]}$ 同形状的梯度矩阵。对于 $\frac{\partial L}{\partial b^{[L]}}$，我们通常会在批量维度上对 $\delta z^{[L]}$ 求和。）
4. **将梯度传播到前一层：** 真正巧妙的部分是将误差梯度反向传播。我们需要找到 $\frac{\partial L}{\partial a^{[L-1]}}$，即损失对*前一层*的*激活值*的梯度。再次通过 $z^{[L]}$ 使用链式法则：
   $\frac{\partial L}{\partial a^{[L-1]}} = \frac{\partial L}{\partial z^{[L]}} \cdot \frac{\partial z^{[L]}}{\partial a^{[L-1]}} = \delta z^{[L]} \cdot (W^{[L]})^T$
   我们将此记为 $\delta a^{[L-1]}$。这表明了 $L-1$ 层中的激活值对最终损失 $L$ 的影响程度。
5. **对 $L-1$ 层重复：** 现在我们有了 $\delta a^{[L-1]}$。我们可以对 $L-1$ 层重复步骤 2、3 和 4：

   - 计算 $\delta z^{[L-1]} = \delta a^{[L-1]} \cdot g'(z^{[L-1]})$。
   - 计算 $\frac{\partial L}{\partial W^{[L-1]}} = \delta z^{[L-1]} \cdot (a^{[L-2]})^T$。
   - 计算 $\frac{\partial L}{\partial b^{[L-1]}} = \delta z^{[L-1]}$。
   - 计算 $\delta a^{[L-2]} = \delta z^{[L-1]} \cdot (W^{[L-1]})^T$。
6. **持续到输入层：** 这个过程会重复，逐层反向进行，直到我们到达输入层。在每一步 $l$，我们使用传入的梯度 $\delta a^{[l]}$（从 $l+1$ 层计算得到）来计算 $\delta z^{[l]}$，然后计算 $W^{[l]}$ 和 $b^{[l]}$ 的梯度，最后计算梯度 $\delta a^{[l-1]}$ 以传回给前一层。

这种系统性的链式法则反向应用，保证我们能够高效地计算损失 $L$ 对网络中*每个*参数的梯度，并尽可能地重用计算。所有层的梯度 $\frac{\partial L}{\partial W^{[l]}}$ 和 $\frac{\partial L}{\partial b^{[l]}}$ 随后被用于梯度下降 (gradient descent)等优化算法来更新参数：

$W^{[l]} := W^{[l]} - \alpha \frac{\partial L}{\partial W^{[l]}}$
$b^{[l]} := b^{[l]} - \alpha \frac{\partial L}{\partial b^{[l]}}$

其中 $\alpha$ 是学习率。

> 这是一个两层网络中前向传播（计算损失）和反向传播（计算梯度）的简化视图。反向传播从损失开始，逐层应用链式法则，以计算损失对每个参数 ($W^{[l]}, b^{[l]}$) 和激活值 ($a^{[l]}$) 的梯度。红色节点和箭头表示梯度的流动和计算。

理解这种反向流动以及链式法则在每一步中的应用，是掌握神经网络 (neural network)如何学习的基础。下一节将讨论计算图如何帮助更清晰地展示这个过程。

## 参考资料

- [Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0) — David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams (1986)
  Journal: Nature; Publisher: Springer Nature; Volume: 323; Pages: 533-536; DOI: [10.1038/323533a0](https://doi.org/10.1038/323533a0)
  提出了反向传播算法，作为一种有效计算多层神经网络梯度的方法，极大地推动了其训练。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本综合性教科书，其中第6章介绍了深度前馈网络，包括反向传播的严谨数学推导和解释。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael A. Nielsen (2015)
  Publisher: Determination Press
  提供了反向传播工作原理清晰、循序渐进的解释，使基础微积分易于理解和掌握。
- [Backpropagation in Neural Networks (CS231n Convolutional Neural Networks for Visual Recognition)](https://cs231n.github.io/optimization-2/) — Stanford University CS231n Course Staff (2024)
  这些课程笔记详细且实用地解释了反向传播，经常使用与现代深度学习框架相关的向量化表示。
