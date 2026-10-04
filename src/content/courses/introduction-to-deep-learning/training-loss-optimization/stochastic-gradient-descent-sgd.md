---
course: "introduction-to-deep-learning"
chapter: "training-loss-optimization"
lesson: "stochastic-gradient-descent-sgd"
sourceId: 5040
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-3-training-loss-optimization/stochastic-gradient-descent-sgd"
title: "随机梯度下降 (SGD)"
description: "了解随机梯度下降（SGD）和迷你批次梯度下降如何改进标准梯度下降算法。"
order: 7
plots: []
sourceHash: "21425b84281e41aaf039af9b3a255143ca6373daeb009206e0a53012e6166f7f"
sourceCorrections: []
---

在上一节中，我们回顾了标准梯度下降 (gradient descent)算法。该算法涉及使用*全部*训练数据集计算损失函数 (loss function)相对于模型参数 (parameter)的梯度。虽然这能准确地估算出指向最小值的梯度方向，但它带来了显著的计算开销，尤其是在处理大型数据集时。想象一下，如果训练样本有数百万或数十亿个；那么仅为一次参数更新而计算所有样本的梯度将变得极其缓慢且占用大量内存。

面对大型数据集的挑战，**随机梯度下降 (SGD)** 提供了一种有效的解决方案。SGD 背后的主要思想非常简单：它并非基于整个数据集计算梯度，而是在每一步仅使用*一个*随机选取的训练样本来近似梯度，从而显著降低了计算开销。

### SGD 的运作方式

具体过程如下：

1. **洗牌：** 在每个周期（对训练数据进行一次完整遍历）开始时，随机打乱数据集。这可以避免样本顺序引入的任何偏差。
2. **迭代：** 遍历打乱后的数据集，每次取一个样本 $(x^{(i)}, y^{(i)})$。
3. **计算梯度：** 只为这一个样本计算损失函数 (loss function) $J$ 的梯度： $\nabla J(w; x^{(i)}, y^{(i)})$。
4. **更新参数 (parameter)：** 使用这个单样本梯度更新模型参数（权重 (weight) $w$ 和偏置 (bias) $b$）：
   
   $$
   w \leftarrow w - \eta \nabla J(w; x^{(i)}, y^{(i)})
   $$
   
   这里 $\eta$ 是学习率。

此循环对数据集中的所有样本重复执行，完成一个周期。此过程会持续多个周期。

### 为何使用 SGD？其优点与缺点

SGD 具有以下优点：

- **速度：** 参数 (parameter)更新的频率更高（每个样本更新一次，而不是每个周期更新一次）。这通常会使得实际运行时间上的收敛更快，尤其是在大型数据集上，即使总体更新次数更多。
- **大型数据集：** 它可以处理无法完全载入内存的庞大数据集，因为它一次只需要一个样本。
- **逃离局部最小值：** SGD 中的更新是有噪声的，因为梯度是根据单个样本计算的，这可能无法完全表示整体梯度方向。这种噪声实际上是有益的，它可能帮助优化过程跳出较浅的局部最小值或鞍点，并找到更好的解决方案。

然而，SGD 也有其缺点：

- **噪声更新：** 梯度估计中的高方差导致其趋向最小值的路径比批量梯度下降 (gradient descent)更加不稳定。损失值在更新之间可能会显著波动。
- **收敛性：** 由于噪声的存在，SGD 可能不会收敛到精确的最小值，而是在其附近波动。通常需要仔细调整学习率，这常涉及学习率调度（随时间逐渐降低 $\eta$）。

### 小批量梯度下降 (gradient descent)：两全其美的方法

实践中，纯粹的批量梯度下降或纯粹的 SGD 通常都不会单独使用。取而代之的是，一种折衷方法，称为**小批量梯度下降**，是目前最常见的方法。

小批量梯度下降通过基于训练数据的一个小型随机选定子集（即“小批量”）来计算梯度并更新参数 (parameter)，而非整个数据集或单个样本。典型的小批量大小范围从 32 到 256 个样本，但会根据应用和硬件内存限制而变化。

更新规则变为：


$$
w \leftarrow w - \eta \nabla J(w; x^{(i:i+n)}, y^{(i:i+n)})
$$


$n$ 是小批量大小，且 $\nabla J(w; x^{(i:i+n)}, y^{(i:i+n)})$ 是该小批量上的平均梯度。

小批量梯度下降有以下几点优势：

1. **更平滑的收敛：** 梯度估计比 SGD 中的噪声小，从而带来更稳定的收敛路径。
2. **计算效率：** 它通过使用矢量化 (quantization)操作批量处理样本来利用硬件优化（如 GPU），这使其计算速度远快于 SGD 中逐个处理样本的方式。
3. **平衡性：** 它在 SGD 的鲁棒性和批量梯度下降的稳定性之间取得了良好的平衡。

下图展示了批量梯度下降、SGD 和小批量梯度下降在损失曲面上所采取的优化路径差异。

> 批量梯度下降、SGD 和小批量梯度下降的特点比较。N 表示训练样本总数，n 表示小批量大小。

这些变体的选择取决于数据集大小和计算资源。对于当今大多数深度学习 (deep learning)应用而言，小批量梯度下降是默认选择，它提供了一种实用且高效的方式来处理神经网络 (neural network)复杂的损失曲面。尽管在日常交流或甚至库实现中，我们常将优化过程简称为“SGD”，但这几乎总是意味着使用了小批量。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8: Optimization for Training Deep Models
  这本基础教材全面阐述了随机梯度下降法、其理论基础以及在深度学习中的实际考量。
- [Optimization Methods for Large-Scale Machine Learning](https://doi.org/10.1137/16M1080173) — Léon Bottou, Frank E. Curtis, and Jorge Nocedal (2018)
  Journal: SIAM Review; Publisher: Society for Industrial and Applied Mathematics; Volume: 60; Pages: 223-311; DOI: [10.1137/16M1080173](https://doi.org/10.1137/16M1080173)
  这篇综述文章严格地从学术角度回顾了随机梯度下降及其变体，讨论了它们在大型机器学习中的理论属性和表现。
- [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747) — Sebastian Ruder (2016)
  Journal: arXiv:1609.04747 [cs.LG]; DOI: [10.48550/arXiv.1609.04747](https://doi.org/10.48550/arXiv.1609.04747)
  这篇广受引用的论文提供了对各种梯度下降优化算法（包括SGD和mini-batch GD）的易于理解且实用的概述，有助于理解实际实现。
- [CS231n: Convolutional Neural Networks for Visual Recognition, Lecture Notes: Optimization](https://cs231n.github.io/optimization-1/) — Stanford University (2023)
  这份来自著名深度学习课程的官方讲义，清晰地解释了随机梯度下降、mini-batch GD及其影响，并提供了实践指导。
