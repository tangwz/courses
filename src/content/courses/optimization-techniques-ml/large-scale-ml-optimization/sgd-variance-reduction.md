---
course: "optimization-techniques-ml"
chapter: "large-scale-ml-optimization"
lesson: "sgd-variance-reduction"
sourceId: 1277
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-4-large-scale-ml-optimization/sgd-variance-reduction"
title: "随机梯度下降再讨论：方差减小"
description: "分析SGD中固有的方差以及方差减小方法的原因。"
order: 1
plots: ["plots/1277-0.json"]
sourceHash: "5befa6d9750404f99ace3913c853c50935c396419d334b11bc53f98205e66edd"
sourceCorrections: []
---

随机梯度下降 (gradient descent)（SGD）是训练大型数据集模型的主力方法。它的优点在于使用小数据子集（小批量）来估计梯度，与在整个数据集上计算梯度（批量梯度下降）相比，大幅降低了每次迭代的计算成本。然而，这种效率并非没有代价：方差。

每个随机梯度估计， $g_{i_t}(\theta_t) = \nabla f_{i_t}(\theta_t)$ ($f_{i_t}$ 是在迭代 $t$ 时小批量 $i_t$ 上的损失函数 (loss function))，是真实梯度 $\nabla f(\theta_t) = \frac{1}{N} \sum_{i=1}^N \nabla f_i(\theta_t)$ 的无偏估计。这表示，从期望上看，随机梯度指向正确的方向：$\mathbb{E}[g_{i_t}(\theta_t)] = \nabla f(\theta_t)$。但对于任何单次迭代， $g_{i_t}(\theta_t)$ 可能与 $\nabla f(\theta_t)$ 有显著偏离。这种偏离将噪声或方差引入到参数 (parameter)更新中。

### 为什么梯度方差是个问题？

SGD固有的方差带来了几个实际影响：

1. **路径噪声：** 优化路径不像批量梯度下降 (gradient descent)那样平滑地趋向最小值。相反，它倾向于之字形移动，使得优化进展不那么直接。
2. **收敛速度较慢：** 尽管每一步的计算成本低，但噪声更新通常意味着需要更多步才能达到一个好的解，相比于使用完整梯度的方法。即使对于强凸问题，批量梯度下降能达到线性收敛，SGD的收敛速度通常也是次线性的。
3. **精细调整的困难：** 随着参数 (parameter)接近最小值，真实梯度的幅度变得很小。然而，随机梯度的方差不一定以相同的速率减小。这意味着噪声可能压倒信号，导致参数在最小值附近震荡而不是精确收敛到它。这需要仔细调整学习率计划，通常要求学习率随时间衰减以抑制震荡。



![优化路径](plots/1277-0.json)



> 简单二次损失曲面上的优化路径比较。批量梯度下降采取平滑的步长。SGD由于梯度方差而表现出噪声步长。方差减小方法的目标是比SGD实现更快、噪声更小的收敛。

### 方差减小方法的基本思路

方差减小方法的基本思路是修改随机梯度估计使其具有较低方差，同时保持计算效率。我们希望得到一个估计 $\tilde{g}_t(\theta_t)$，它能满足：

1. 它在计算上保持低成本，理想情况下，每次迭代只需求解小批量梯度，类似于SGD。
2. 它具有显著低于标准SGD梯度 $g_{i_t}(\theta_t)$ 的方差。
3. 它仍然是真实梯度 $\nabla f(\theta_t)$ 的无偏（或在某些变体中接近无偏）估计。

实现这一点通常需要将额外信息整合到梯度估计中。例如，方法可以周期性地计算完整梯度，或者维护在不同点评估的过去梯度的移动平均。目标是构建一个梯度估计，它能够将噪声单批量梯度修正到真实的梯度方向。

通过减小方差，这些方法通常能够达到更快的理论收敛速度（对于某些问题类型，可以接近批量梯度下降 (gradient descent)的线性收敛速度），并且与标准SGD相比，表现出更好的实际性能，尤其是在优化后期。

后续章节将介绍具体的算法，例如随机平均梯度（SAG）和随机方差减小梯度（SVRG），它们采用不同的策略来实现这种方差减小，使大规模优化更加高效和有成效。

## 参考资料

- [Accelerating Stochastic Gradient Descent using Stochastic Variance Reduced Gradient](https://proceedings.neurips.cc/paper/2013/file/a76a59bba636be929e79a83b092200f3-Paper.pdf) — Rie Johnson, Tong Zhang (2013)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 26; Pages: 315-323
  介绍了随机方差减少梯度（SVRG）算法，这是随机梯度方法中方差减少领域的一项基础性工作。
- [A stochastic gradient method with an exponential convergence rate for strongly convex objectives](https://proceedings.neurips.cc/paper/2012/file/1543843a4723ed26a08e17005740be57-Paper.pdf) — Nicolas Le Roux, Mark Schmidt, Francis Bach (2012)
  Journal: Advances in Neural Information Processing Systems; Publisher: Neural Information Processing Systems Foundation; Volume: 25; Pages: 2663-2671; DOI: [10.5555/2999335.2999468](https://doi.org/10.5555/2999335.2999468)
  提出了随机平均梯度（SAG）方法，这是一种开创性的方差减少技术，可为强凸问题实现线性收敛速度。
- [Optimization Methods for Large-Scale Machine Learning](https://doi.org/10.1137/16M1080173) — Léon Bottou, Frank E. Curtis, Jorge Nocedal (2018)
  Journal: SIAM Review; Publisher: Society for Industrial and Applied Mathematics; Volume: 60; Pages: 223-311; DOI: [10.1137/16M1080173](https://doi.org/10.1137/16M1080173)
  一份关于大规模机器学习优化算法的综合性综述，涵盖了SGD的理论性质和各种方差减少方法。
