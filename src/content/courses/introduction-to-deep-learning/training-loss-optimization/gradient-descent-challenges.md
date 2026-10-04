---
course: "introduction-to-deep-learning"
chapter: "training-loss-optimization"
lesson: "gradient-descent-challenges"
sourceId: 5041
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-3-training-loss-optimization/gradient-descent-challenges"
title: "梯度下降的难题"
description: "讨论局部最小值、鞍点和收敛缓慢等问题。"
order: 8
plots: ["plots/5041-0.json", "plots/5041-1.json", "plots/5041-2.json"]
sourceHash: "d8b277b3f035c473e57c37744afa8f438cbcd7c7379c152cef3ee579c291e927"
sourceCorrections: []
---

尽管梯度下降 (gradient descent)为神经网络 (neural network)优化提供了一种核心机制，但损失函数 (loss function)最小化的过程并非总是直截了当。顺利滚下山坡进入最低谷的理想化景象可能具有误导性，尤其是在深度学习 (deep learning)模型常见的高维空间 (high-dimensional space)中。若干问题会阻碍或使训练过程复杂化。

### 局部最小值

神经网络 (neural network)的损失曲面很少是只有一个最低点的简单凸碗状。它通常复杂且非凸，可能包含许多局部最小值。这些点处，损失低于周围区域，导致梯度变为零（$\u2207L = 0$），但它们并非可能损失最低的点（全局最小值）。

梯度下降 (gradient descent)本质上遵循最陡峭的下降路径。如果它恰好落入局部最小值，梯度就会变为零，算法会停止更新权重 (weight)，实际上会“卡住”，即使损失曲面上的其他地方可能存在更好的解决方案。



![局部最小值与全局最小值](plots/5041-0.json)



> 一个简单的一维损失曲面，说明梯度为零的点。如果全局最小值存在于别处，梯度下降可能会停留在任一橙色点（局部最小值），而不是找到可能最低的损失。

虽然局部最小值最初被认为是主要障碍，但研究表明，在深度网络极高维空间 (high-dimensional space)中，大多数临界点（梯度为零处）实际上是鞍点，而且足够深的局部最小值在性能上通常接近全局最小值。

### 鞍点

在高维空间 (high-dimensional space)中比局部最小值更常见的是鞍点。鞍点是梯度为零的位置，但它沿某些维度是最小值，沿其他维度是最大值。想象一下马鞍的形状：它前后方向向下凹陷，但左右方向向上弯曲。

鞍点附近，梯度会变得非常小，导致梯度下降 (gradient descent)显著减速，甚至长时间停滞，之后才可能找到逃逸方向。这种减速会使训练效率低下。



![鞍点可视化](plots/5041-1.json)



> 曲面图说明了 (0, 0, 0) 处的鞍点。损失沿一个轴减小，但沿另一个轴增加。梯度在此点为零。

### 收敛缓慢与峡谷

损失函数 (loss function)曲面也可能具有长而窄的山谷或峡谷，其侧壁陡峭但谷底坡度平缓。在这种情况下，梯度下降 (gradient descent)可能会在峡谷的陡峭壁上反复振荡，同时沿着谷底向实际最小值进展非常缓慢。发生这种情况是因为梯度在横向峡谷方向上比沿峡谷方向陡峭得多。在这里调整学习率可能很困难；适用于平缓坡度的学习率可能对陡峭的侧壁来说过大，导致不稳定。



![狭窄山谷中的梯度下降（等高线图）](plots/5041-2.json)



> 等高线图显示了在 (1, 1) 附近具有狭窄山谷最小值的损失函数。红线说明了标准梯度下降如何可能在山谷中振荡，向着最小值缓慢前进。

### 对学习率和特征缩放的敏感性

如前所述，学习率（$\u03b7$）是一个重要的超参数 (parameter) (hyperparameter)。选择合适的值非常重要：

- **过小：** 收敛非常缓慢，可能需要不可行的时间才能达到一个好的最小值。
- **过大：** 算法可能越过最小值，导致损失剧烈振荡甚至发散（无限增加）。

此外，梯度下降 (gradient descent)对输入特征的尺度敏感。如果特征的范围差异很大（例如，一个特征范围是 0-1，另一个是 1-10000），损失曲面会变得细长，加剧峡谷问题，并使找到一个合适的学习率变得困难。这就是为什么特征缩放（如归一化 (normalization)或标准化，第 5 章中会介绍）是标准的预处理步骤。

这些难题表明，尽管朴素梯度下降是优化方法的根本，但实践中常需要更复杂的算法。动量法、RMSprop 和 Adam 等方法（我们将在下一章中详细阐述）专门设计用于缓解这些问题，使得在复杂损失曲面上更快、更可靠地收敛。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8: Optimization for Training Deep Models
  这本教科书详细解释了深度学习中的优化挑战，包括局部最小值、鞍点、狭长山谷中的慢收敛以及学习率的影响。
- [The Loss Surfaces of Multilayer Networks](https://proceedings.mlr.press/v38/choromanska15.html) — Anna Choromanska, Mikael Henaff, Michael Mathieu, Gérard Ben Arous, Yann LeCun (2015)
  Journal: Proceedings of Machine Learning Research; Publisher: PMLR; Volume: 38; Pages: 192-204
  这篇论文从理论上探讨了深度神经网络损失曲面的几何特性，认为在高维空间中，大多数局部最小值在经验上是好的，主要挑战来自鞍点。
- [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](https://papers.nips.cc/paper/2014/file/b53b3a3d6ab90ce026822923c892d1f9-Paper.pdf) — Yann N. Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems (NIPS 27); Publisher: MIT Press; Pages: 2933-2941
  这篇论文讨论了高维非凸优化中鞍点比局部最小值更普遍的现象，解释了它们如何阻碍基于梯度的优化算法。
- [Optimization Methods for Large-Scale Machine Learning](https://doi.org/10.1137/16M1080173) — Léon Bottou, Frank E. Curtis, Jorge Nocedal (2018)
  Journal: SIAM Review; Publisher: Society for Industrial and Applied Mathematics; Volume: 60; Pages: 223-311; DOI: [10.1137/16M1080173](https://doi.org/10.1137/16M1080173)
  这篇综述文章全面概述了大规模机器学习中使用的优化算法，讨论了与深度学习相关的理论基础和实践方面。
