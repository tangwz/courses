---
course: "deep-learning-regularization-optimization"
chapter: "optimization-refinements-tuning"
lesson: "tuning-hyperparameters-overview"
sourceId: 5004
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-7-optimization-refinements-tuning/tuning-hyperparameters-overview"
title: "调整超参数：学习率、正则化强度、批大小"
description: "讨论调整学习率、正则化参数和批大小等重要超参数的过程。"
order: 7
plots: ["plots/5004-0.json"]
sourceHash: "8ef78902424dcf4cab3eafbc7aa7f081df551a423d839521277ac5b749fdc307"
sourceCorrections: []
---

选择合适的初始化策略和学习率调度能为训练提供良好起点，但深度学习 (deep learning)模型的性能往往很大程度上取决于为几个重要超参数 (parameter) (hyperparameter)找到合适的值。这些设置是在训练过程*开始前*配置的，不同于在训练*期间*学习到的模型参数（权重 (weight)和偏置 (bias)）。调整这些超参数是深度学习工作流程的一个重要环节，通常需要系统化尝试和实验。

主要关注三个影响较大的超参数：学习率（$\alpha$）、正则化 (regularization)强度（$\lambda$）和迷你批大小。理解如何调整这些参数能明显影响模型收敛速度、最终性能以及对新数据的泛化能力。

### 超参数 (parameter) (hyperparameter)是什么？

回顾一下，模型参数是网络中的权重 (weight)和偏置 (bias)，优化算法在训练过程中调整它们以最小化损失函数 (loss function)。另一方面，超参数是定义模型结构或训练过程本身的外部配置。例子包括：

- 优化器的**学习率**（$\alpha$）。
- L1或L2正则化 (regularization)的**正则化强度**（$\lambda$）。
- **Dropout率**。
- 网络中的**层数**。
- 每层**单元数量**。
- **激活函数 (activation function)的选择**。
- **优化器的选择**（SGD、Adam等）。
- **迷你批大小**。
- 学习率调度参数（例如，衰减率）。

找到超参数的良好组合通常更像艺术而非科学，这依靠经验、直觉和迭代实验来指引。

### 调整学习率（$\alpha$）

学习率可以说是最重要的超参数 (parameter) (hyperparameter)。如前几章所述，它控制着梯度下降 (gradient descent)期间的步长。

- **$\alpha$ 过小：** 训练进展非常缓慢，可能陷入糟糕的局部最小值，或者收敛时间过长，不切实际。
- **$\alpha$ 过大：** 训练可能变得不稳定。损失可能剧烈震荡甚至发散（无限增加），因为步长越过了最小值。

寻找有效的学习率通常需要在对数范围内进行搜寻，例如 $10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}$。Adam优化器常用的起始点通常在 $10^{-3}$ 或 $10^{-4}$ 左右，而带动量的SGD可能从 $10^{-2}$ 左右开始。然而，这些只是经验法则，最优值很大程度上取决于数据集、模型架构、优化器选择，甚至批大小。

学习率调度（之前已讨论）通过在训练期间调整 $\alpha$ 提供帮助，但*初始*学习率和调度本身的参数（例如，衰减率、步长）仍需仔细选择。监控训练损失曲线很重要；快速下降但稳定的损失表明学习率良好，而震荡或发散则表示它可能过高。



![学习率对训练损失的影响](plots/5004-0.json)



> 不同学习率下训练损失曲线的示例。选择得当的学习率会显示稳步下降，而过小的学习率收敛缓慢，过大的学习率则导致不稳定或发散。

### 调整正则化 (regularization)强度（$\lambda$）

正则化技术，如L1和L2（权重 (weight)衰减），在第2章中有所提及，它们根据模型权重的量级向损失函数 (loss function)添加一个惩罚项。正则化强度，通常用 $\lambda$ (lambda) 表示，控制着这个惩罚的权重。


$$
\text{总损失} = \text{原始损失（例如，交叉熵）} + \lambda \times \text{正则化项}
$$


- **$\lambda$ 过小：** 正则化效果微乎其微，模型仍可能严重过拟合 (overfitting)。这相当于几乎没有正则化。
- **$\lambda$ 过大：** 对权重的惩罚主导损失函数。优化器过于专注于将权重收缩至零，可能忽略拟合数据的任务，导致欠拟合 (underfitting)（高偏差）。

与学习率类似，$\lambda$ 也常在对数尺度上调整，尝试 $0.1, 0.01, 0.001, 0.0001, 0$ 等值。最优值取决于在没有正则化的情况下观察到的过拟合程度。如果模型严重过拟合（训练和验证损失/准确率之间存在较大差异），则可能需要更大的 $\lambda$。如果模型欠拟合，则应减小 $\lambda$ 或将其设置为零。请记住，Dropout和批归一化 (normalization)等其他正则化技术也会影响最优 $\lambda$。

### 调整批大小

批大小决定了在更新模型权重 (weight)之前处理多少训练样本。它影响着训练动态和计算资源的使用。

- **小批大小（例如，1、8、16、32）：**
  - 在梯度估计中引入更多噪声。这种噪声有时能帮助优化器逃离尖锐的局部最小值，并可能带来更好的泛化（作为一种正则化 (regularization)形式）。
  - 每批需要更少内存。
  - 更新频繁，这可能导致更快的初始收敛，但由于并行硬件（GPU）利用不足，计算效率可能较低。由于频繁且并行化程度较低的更新，整体训练速度可能变慢。
- **大批大小（例如，128、256、512、1024+）：**
  - 提供更准确的梯度估计，带来更平滑的收敛。
  - 能更有效地利用硬件并行性，如果内存允许，可能加快每轮训练时间。
  - 可能收敛到更尖锐的最小值，这有时泛化能力不如通过小批量找到的更平坦的最小值。
  - 需要明显更多内存。

批大小的选择通常受限于GPU内存。常见做法是，从32、64或128等标准大小开始，然后根据性能和内存限制进行调整。同时也要注意批大小和学习率之间的关系，我们将在下一节中阐述。由于硬件内存对齐 (alignment)效率，批大小常选择2的幂次方，但这并非严格要求。

找到这些超参数 (parameter) (hyperparameter)的正确组合对于最大化模型性能很重要。接下来的章节将讨论网格搜索和随机搜索等策略，以更系统地应对这个复杂的调整过程。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，全面涵盖深度学习，包括超参数作用和调优策略的详细解释。
- [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820) — Leslie N. Smith (2018)
  Journal: arXiv preprint arXiv:1803.09820; Publisher: US Naval Research Laboratory; DOI: [10.48550/arXiv.1803.09820](https://doi.org/10.48550/arXiv.1803.09820)
  本文为学习率、批量大小和权重衰减等关键超参数的调优提供了实用的指导和策略，与本节内容直接相关。
- [CS231n Convolutional Neural Networks for Visual Recognition, Lecture Notes: Training Neural Networks, Part 3](https://cs231n.github.io/neural-networks-3/) — Stanford University (2023)
  一个备受推崇的在线资源，为深度学习背景下的超参数调优（包括学习率、正则化和批量大小）提供了实用指导。
