# Dropout介绍：避免协同适应

来源：[原文](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-3-dropout-regularization/dropout-introduction)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Dropout是一种在深度学习 (deep learning)中截然不同的正则化 (regularization)方法。一些正则化方法通过根据网络权重 (weight)的幅值向损失函数 (loss function)添加惩罚来帮助避免过拟合 (overfitting)，而Dropout则是在训练期间以随机方式改变网络自身的结构。其主要目的是避免深度网络训练中一个常见的问题：协同适应。

### 什么是神经元协同适应？

设想一下训练一个深度神经网络 (neural network)。在训练过程中，一层中的神经元学习根据它们从前一层接收到的输入调整自己的权重 (weight)。有时，一个神经元可能会过度依赖其下层少数几个神经元的特定输出。它基本上学会了与*那些特定*的输入配合得非常好，但如果这些输入略有变化或有其他神经元提供信号，则效果会差一些。

类似地，神经元群组也可能学会以非常特定的方式协同工作，互相纠正错误或依赖于该群组产生的特定模式。这种现象称为**协同适应**。虽然这可能有助于网络最小化训练数据上的损失，但通常会导致泛化能力差。网络变得过于专门化于训练集的具体细节和噪声，就像一个团队，成员只有在特定搭档在场时才能正常运作。如果输入数据即使略有变化（就像未见过的新测试数据一样），这些脆弱的协同依赖就会瓦解，性能就会下降。这是一种过拟合 (overfitting)的形式。

> 第L层的一个神经元（N5）变得过度依赖于第L-1层中的某些神经元（N1，N2）。这种紧密连接说明了协同适应。

### 使用Dropout打破依赖关系

Dropout提供了一种简单而有效的方法来打破这些协同适应。核心思想很直接：对于每个训练步骤（在每个小批量数据上），暂时性地随机移除网络中的神经元（及其传入和传出连接）。

设想一下，对于一个给定的训练样本，你随机选择一层中的一部分神经元，并将它们的输出设为零。在下一个训练样本中，你随机选择*不同*的一部分。这意味着一个神经元不能依赖于任何其他特定神经元始终存在，因为任何上游神经元都可能在训练期间的任何时刻被“丢弃”。

> 在应用Dropout的训练步骤中，一些神经元（N2、N4、N6）被随机停用（显示为灰色）。在这次特定的前向和后向传播中，连接到这些神经元和从这些神经元发出的连接都被忽略。

这种随机停用使得每个神经元变得更加独立。它必须学习那些自身有用，或者与其他随机选择的神经元子集结合后有用的特征。网络作为一个整体，学习到更多冗余的表示，这意味着知识分布在整个网络中，而不是集中在某些脆弱的路径上。这大大降低了过拟合 (overfitting)的风险，并提升了模型对新的、未见过的数据的泛化能力。

可以将其视为在隐式地训练大量更小、更稀疏的网络集成。每次处理一个迷你批次时，你实际上是在训练完整网络的一个不同稀疏版本。虽然你没有明确创建和平均这些网络，但这种随机丢弃过程达到了类似的正则化 (regularization)效果。

在接下来的章节中，我们将查看Dropout在训练期间如何应用的具体机制，以及网络在测试（推断）期间如何使用，以及如何在实践中实现它。

## 参考资料

- [Dropout: A Simple Way to Prevent Overfitting in Neural Networks](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 15; Pages: 1929-1958; DOI: [10.5555/2627435.2670313](https://doi.org/10.5555/2627435.2670313)
  介绍Dropout的奠基性论文，详细阐述其防止共适应和提高泛化能力的作用机制。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本标准教材，其中有专门的章节解释Dropout、其理论基础和实际应用。
- [CS231n Convolutional Neural Networks for Visual Recognition: Regularization](http://cs231n.github.io/neural-networks-2/#reg) — Stanford University (2017)
  Publisher: Stanford University
  对Dropout、其在正则化中的作用以及其概念益处提供了清晰易懂的解释。

---

[上一节](../02-%E6%9D%83%E9%87%8D%E6%AD%A3%E5%88%99%E5%8C%96%E6%96%B9%E6%B3%95/09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%B0%86L1-L2%E5%BA%94%E7%94%A8%E4%BA%8E%E7%BD%91%E7%BB%9C.md) · [下一节](02-Dropout%20%E5%9C%A8%E8%AE%AD%E7%BB%83%E6%97%B6%E7%9A%84%E5%B7%A5%E4%BD%9C%E5%8E%9F%E7%90%86.md)
