---
course: "deep-learning-regularization-optimization"
chapter: "optimization-refinements-tuning"
lesson: "learning-rate-schedules-motivation"
sourceId: 5000
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-7-optimization-refinements-tuning/learning-rate-schedules-motivation"
title: "学习率调度的缘由"
description: "解释训练期间调整学习率为何能提升性能。"
order: 3
plots: ["plots/5000-0.json"]
sourceHash: "cf47110967f0d9036e2f0a573616a58bf8922ab6111114ac0f8c41336984a864"
sourceCorrections: []
---

随机梯度下降 (gradient descent)（SGD）、动量（Momentum）、RMSprop和Adam等优化算法是深度学习 (deep learning)中常用的方法。这些算法中的一个主要部分是学习率，通常表示为 $\alpha$。这个超参数 (parameter) (hyperparameter)控制着每次参数更新时的步长，指导着我们下降的方向。通常情况下，学习率被视为训练开始前选定的一个固定值。然而，固定学习率总是最好的方法吗？

思考模型训练的通常过程。早期，模型的参数可能离其最优值较远。在这个区域，损失面可能相对陡峭或复杂。此时，较大的学习率可能有利，使优化器能够迈出大步，快速通过平坦区域，并迅速趋近损失较低的区域。把它想象成在宽广、未知地带中大步前进。

然而，随着训练的进行，当参数趋近一个良好解（最小值）时，大的学习率可能会带来问题。优化器可能会越过最小值，在谷底来回震荡而无法稳定下来。这可能导致损失函数 (loss function)出现波动，并阻碍模型收敛到最优解。在我们的类比中，当你试图在一个小山谷中精确定位最低点时，大步前进会适得其反；需要更小、更谨慎的步子。



![固定学习率的影响](plots/5000-0.json)



> 损失曲线图示：比较不同固定学习率与理想情况（学习率随时间降低）。高学习率会导致震荡，而低学习率则收敛缓慢。

反之，如果我们从一个非常小的学习率开始，训练可能极其缓慢。优化器迈出微小步子，需要大量迭代才能达到一个良好的最小值。此外，持续较小的学习率可能会增加陷入次优局部最小值的风险，或者难以有效处理鞍点。

这带来一个难题：最优学习率在训练过程中似乎会变化。高学习率在初期有益，而较低的学习率在后期更好。这一发现促成了**学习率调度**的运用。

学习率调度是一种在训练期间动态调整学习率 $\alpha$ 的策略，而非保持其固定不变。其主要思路通常是：开始时使用相对较高的学习率，以获得快速的初期进展，然后随着训练的进行逐渐降低学习率。这种降低使优化器能够进行更精细的调整，帮助其更平稳、更准确地收敛到一个良好的最小值，避免过度震荡。

采用学习率调度可以带来以下几点益处：

- 与从一开始就使用小的固定学习率相比，可能实现更快的整体收敛。
- 通过使模型稳定在一个更佳的最小值，达到更低的最终训练和验证损失。
- 提高训练稳定性，尤其是在后期阶段。

尽管像Adam这样的自适应优化器会根据过去的梯度，逐参数调整学习率，学习率调度则随时间（训练轮次或迭代次数）修改*全局基础学习率*。这两种方法并非互相排斥，并且通常可以结合使用。

在接下来的章节中，我们将介绍实现学习率调度的常见策略，例如步进衰减、指数衰减和余弦退火，并讨论如何将其整合到你的训练流程中。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供深度学习优化基础知识，包括学习率的作用及其动态调整的原理。
- [An Overview of Gradient Descent Optimization Algorithms](https://arxiv.org/abs/1609.04747) — Sebastian Ruder (2016)
  Journal: arXiv preprint arXiv:1609.04747; DOI: [10.48550/arXiv.1609.04747](https://doi.org/10.48550/arXiv.1609.04747)
  对梯度下降优化算法的全面综述，其中包含专门讨论学习率调度动机和各种类型的章节。
- [Optimization](https://cs231n.github.io/optimization-2/) — Stanford University CS231n (2023)
  提供对深度学习中优化挑战的清晰直观解释，强调学习率的重要性以及动态控制的优势。
