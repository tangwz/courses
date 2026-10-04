---
course: "deep-learning-regularization-optimization"
chapter: "foundational-optimizers"
lesson: "revisiting-gradient-descent"
sourceId: 4970
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-5-foundational-optimizers/revisiting-gradient-descent"
title: "回顾梯度下降"
description: "回顾梯度下降优化的基本原理。"
order: 1
plots: ["plots/4970-0.json"]
sourceHash: "9bdc53ff4003194cf0a3966ca825974227d71b1d8f1d47676efdb591111556e5"
sourceCorrections: []
---

为了训练深度学习 (deep learning)模型，我们的目标是找到一组参数 (parameter)（权重 (weight)和偏差），使其能够使给定数据的特定损失函数 (loss function)最小化。优化算法是我们有效进行这项寻找的工具。最基本的算法，也是构建许多其他算法的基础的算法，就是梯度下降 (gradient descent)。

您可能从机器学习 (machine learning)入门知识中会想起梯度下降。其核心思想直接明了：迭代调整模型参数，使其沿着能最大限度减少损失函数的方向移动。我们如何知道哪个方向能最大限度地减少损失？我们使用梯度。

### 梯度：下坡的指引

损失函数 (loss function)$J(	heta)$衡量模型在当前参数 (parameter)$heta$下的表现。这个函数存在于一个高维空间 (high-dimensional space)中（每个参数对应一个维度）。目标是找到这个空间中对应着最低损失的点。

损失函数关于参数的梯度，表示为$\nabla J(\theta)$，是一个指向损失函数在$\theta$点*最陡峭上升*方向的向量 (vector)。由于我们希望*最小化*损失，我们应该沿着与梯度*相反*的方向移动。

想象您正站在一片雾蒙蒙的山坡上，想要到达山谷底部（最低海拔，即最低损失）。您看不到整个区域，但能感受到脚下的坡度。梯度告诉您最陡峭的上坡路径方向。要下山，您就朝完全相反的方向迈一步。

### 更新规则

梯度下降 (gradient descent)通过一个简单的更新规则将这种直觉形式化。在每一步中，我们按照以下方式更新参数 (parameter)$\theta$：


$$
\theta_{\text{new}} = \theta_{\text{old}} - \alpha \nabla J(\theta_{\text{old}})
$$


我们来分解一下：

- $\theta_{\text{old}}$：模型参数的当前值。
- $\nabla J(\theta_{\text{old}})$：使用*当前*参数$\theta_{\text{old}}$计算的损失函数 (loss function)梯度。这个向量 (vector)告诉我们损失最陡峭增加的方向。
- $\alpha$：**学习率**。这是一个小的正标量超参数 (hyperparameter)，控制我们沿着梯度相反方向迈步的大小。选择一个好的学习率很要紧：太小，训练会非常慢；太大，我们可能会越过最小值甚至发散。
- $\theta_{\text{new}}$：迈出一步后更新的参数值。

我们迭代重复这个更新过程，计算梯度并更新参数，希望收敛到一组参数$\theta$，使得损失函数$J(\theta)$的值较低。



![梯度下降在损失曲面上的步骤](plots/4970-0.json)



> 一个简化的2D损失曲面（等高线代表等损失值）。红线显示了梯度下降所走的路径，它从一个起始点开始，沿着与梯度相反的方向迈步，走向最小值（最深蓝色区域）。

### 批量梯度下降 (gradient descent)

梯度下降的“标准”版本，通常被称为**批量梯度下降**（BGD），有一个特定特点：在每个更新步骤中，为了计算梯度$\nabla J(\theta)$，它会处理*整个*训练数据集。它会在进行一次参数 (parameter)更新之前，计算所有训练样本的平均损失和平均梯度。

这给出了整个数据集真实梯度的非常准确的估计，导致收敛路径平稳地趋向最小值。然而，正如章节引言中提到的，对于深度学习 (deep learning)中常用的海量数据集来说，计算整个数据集的梯度在计算量上是难以承受的。想象一下，仅仅为了对模型权重 (weight)做一次微小的调整，就要计算数百万张图像的预测和梯度！

这种计算负担是促使我们接下来讨论的梯度下降变体（例如随机梯度下降（SGD）和迷你批量梯度下降）产生的主要原因。这些方法提供了更快更新的方式，尽管有一些权衡，但它们使优化过程在大规模深度学习中变得可行。了解批量梯度下降的机制，为理解这些变体为何被开发以及它们如何运作提供了必要的铺垫。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  深度学习领域的奠基性教材，全面涵盖包括梯度下降在内的优化算法。
- [Lecture Notes on Supervised Learning and Gradient Descent](http://cs229.stanford.edu/notes/cs229-notes1.pdf) — Andrew Ng (2022)
  Publisher: Stanford University
  机器学习优化的优秀入门材料，从基本原理层面解释梯度下降。
- [An Overview of Gradient Descent Optimization Algorithms](https://arxiv.org/abs/1609.04747) — Sebastian Ruder (2016)
  Journal: arXiv preprint arXiv:1609.04747; DOI: [10.48550/arXiv.1609.04747](https://doi.org/10.48550/arXiv.1609.04747)
  一篇被广泛引用的综述论文，详细介绍了各种梯度下降优化算法，包括基础的批量梯度下降。
