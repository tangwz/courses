---
course: "introduction-to-neural-networks"
chapter: "forward-propagation"
lesson: "applying-activations"
sourceId: 1951
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-3-forward-propagation/applying-activations"
title: "逐层应用激活函数"
description: "了解激活函数如何在线性变换后引入非线性。"
order: 3
plots: ["plots/1951-0.json"]
sourceHash: "fd8a2827da805aee0a60af32f32412c2bfebb4a0dea948a13fee69a0f8cebcae"
sourceCorrections: []
---

神经网络 (neural network)的每一层都会计算线性组合 $Z = WX + b$，它表示每个神经元的加权输入和偏置 (bias)的和。在此之后，一个主要步骤是应用激活函数 (activation function)。如果忽略这一步，直接将线性组合 $Z$ 传递给下一层，那么整个网络，无论有多少层，都将表现为一个单一的、大型的线性变换。线性函数的堆叠会产生另一个线性函数。这样的网络将无法对图像、文本或复杂的表格数据集等数据中常见的非线性关系进行建模。

### 引入非线性

激活函数 (activation function)为网络引入非线性，使其能够学习更复杂的模式和函数。这种非线性变换是*逐元素*地应用于线性步骤 ($Z$) 的输出。这意味着激活函数对矩阵 $Z$ 中的每个元素独立作用。

如果 $Z$ 是包含某一层所有神经元线性组合的矩阵（其中每列可能代表批量中的一个样本，每行代表一个神经元），并且 $g$ 代表所选的激活函数，那么激活步骤的输出，记作 $A$，计算方式如下：


$$
A = g(Z)
$$


矩阵 $A$ 中的每个元素 $A_{ij}$ 是通过将函数 $g$ 应用于矩阵 $Z$ 中对应的元素 $Z_{ij}$ 而获得的：


$$
A_{ij} = g(Z_{ij})
$$


此操作发生在每个隐藏层以及可能在输出层中。

### 应用常用激活函数 (activation function)

回想第1章中常见的激活函数，如Sigmoid、双曲正切（Tanh）和修正线性单元（ReLU）。每个函数都应用特定的非线性变换：

- **ReLU:** $g(z) = \max(0, z)$。如果输入为正，它直接输出输入值；否则输出零。这种方法计算效率高，并在隐藏层中得到广泛使用。
- **Sigmoid:** $g(z) = \frac{1}{1 + e^{-z}}$。它将输入压缩到0到1的范围。
- **Tanh:** $g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$。它将输入压缩到-1到1的范围。

激活函数的选择影响网络的学习方式和表现。由于其简单性以及在深层网络中与Sigmoid或Tanh相比能够缓解梯度消失问题，ReLU是隐藏层的常用默认选项。

我们来可视化一个层内的变换：输入 $X$（或来自前一层的激活 $A$）经过线性变换产生 $Z$，然后 $Z$ 再通过逐元素的激活函数 $g$ 产生该层的输出激活 $A$。

> 数据从输入流经线性组合计算，然后通过逐元素激活函数产生该层的输出。

考虑ReLU激活函数的效果。它将线性组合 $Z$ 中的所有负值裁剪为零，只允许正值通过。



![ReLU激活函数效果](plots/1951-0.json)



> ReLU函数 $g(z) = \max(0, z)$ 逐元素应用于线性变换的输出 $Z$。

### 层间传播

这种两步过程（线性组合后跟非线性激活）定义了网络单层内的计算。在前向传播过程中，层 $l$ 的输出激活 $A^{[l]}$ 成为下一层 $l+1$ 的输入 $X^{[l+1]}$。


$$
Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}
$$


$$
A^{[l]} = g^{[l]}(Z^{[l]})
$$


这里，$A^{[0]}$ 代表初始输入数据 $X$。这个序列在所有隐藏层中重复，直到达到最终输出层。

### 实现

在实际操作中，使用NumPy等库可以高效地将激活函数 (activation function)一次性逐元素应用于整个矩阵 $Z$。例如，应用ReLU：

```python
import numpy as np

# 假设 Z 是包含线性层输出的 NumPy 数组
# Z = np.dot(W, A_prev) + b

# 逐元素应用 ReLU 激活
A = np.maximum(0, Z)

# 现在 A 包含此层的激活值
```

这种向量 (vector)化操作比单独迭代每个元素快很多。

虽然ReLU、Sigmoid和Tanh等激活函数在隐藏层中很常见，但*输出*层中使用的激活函数通常根据特定任务选择（例如，回归任务使用线性函数，二分类使用Sigmoid，多分类使用Softmax）。这确保了网络的输出处于适当的格式，以便计算损失和进行预测。在讨论最终预测计算时，我们将对此进行进一步说明。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本权威教材，涵盖了深度学习的理论基础和实际应用，包括对激活函数的全面讨论。
- [Deep Sparse Rectifier Networks](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf) — Xavier Glorot, Antoine Bordes, and Yoshua Bengio (2011)
  Journal: Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS); Publisher: Proceedings of Machine Learning Research (PMLR); Volume: 15; Pages: 315-323; DOI: [10.55986/aistats2011.glorot11a](https://doi.org/10.55986/aistats2011.glorot11a)
  介绍了整流线性单元（ReLU）激活函数，并展示了其在深度神经网络中的优势。
- [CS231n: Convolutional Neural Networks for Visual Recognition, Neural Networks Part 2: Setting up the Data and the Layers](https://cs231n.github.io/neural-networks-2/) — Stanford University CS231n Course Staff (2023)
  提供了关于神经网络层（包括激活函数及其引入非线性的作用）的清晰简洁的解释。
