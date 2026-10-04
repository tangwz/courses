---
course: "bayesian-machine-learning"
chapter: "bayesian-deep-learning"
lesson: "inference-challenges-bnns"
sourceId: 3625
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-6-bayesian-deep-learning/inference-challenges-bnns"
title: "贝叶斯神经网络的推断挑战"
description: "分析在深度神经网络中进行精确贝叶斯推断的困难。"
order: 3
plots: []
sourceHash: "e5b3046f1b55aebe16ed1c79119477d4f3d2b5651e432f4456ce110244b012c9"
sourceCorrections: []
---

贝叶斯深度学习 (deep learning)的目标是将神经网络 (neural network)的权重 (weight) $w$ 视为具有分布的随机变量，而非固定点。这种方法定义先验分布 $p(w)$，并使用数据 $\mathcal{D}$ 通过贝叶斯定理得到后验分布 $p(w | \mathcal{D})$。


$$
p(w | \mathcal{D}) = \frac{p(\mathcal{D} | w) p(w)}{p(\mathcal{D})}
$$


听起来很简单。然而，将此定义转化为深度神经网络的实际应用时，会遇到重大的计算和分析障碍。为什么获取甚至近似这个后验分布如此困难呢？让我们来看看主要困难。

### 高维度难题

现代深度神经网络 (neural network)的特点是参数 (parameter)数量庞大。即使是中等规模的架构，也可能包含数百万甚至数十亿的权重 (weight)和偏置 (bias)。参数空间 $w$ 的这种高维度是第一个主要障碍。

考虑贝叶斯定理中的分母，即证据或边缘似然：


$$
p(\mathcal{D}) = \int p(\mathcal{D} | w) p(w) dw
$$


这是对网络权重 *所有可能* 配置的积分。在一个拥有数百万或数十亿维度的空间中，除了最简单的情况，解析地计算这个积分是不可能的。数值积分方法，例如基于网格的求积，其计算量随维度数量呈指数增长，几乎会立即变得计算上不可行。蒙特卡洛积分（我们将在后面介绍）提供了一个可能的解决办法，但空间本身的巨大规模使得高效采样变得困难。

此外，后验分布 $p(w | \mathcal{D})$ 本身也存在于这个高维空间 (high-dimensional space)中。可视化或描述这样一个分布并非易事。它很可能高度复杂，可能呈多峰状（对应于不同可信网络配置的多个明显峰值），并且参数之间具有复杂的关联结构。

### 后验归一化 (normalization)的难以处理性

如上所述，证据 $p(\mathcal{D})$ 是后验分布的归一化常数。它的难以处理性意味着我们无法计算给定 $w$ 的后验密度 $p(w | \mathcal{D})$ 的精确值。我们通常可以计算分子 $p(\mathcal{D} | w) p(w)$，它与后验成比例（有时被称为“unnormalized posterior”），但没有分母，我们就无法得到真实的概率密度。

无法计算归一化常数阻碍了从后验分布中直接采样，并使直接优化或分析变得困难。许多推断技术专门用于解决这个限制，或者通过寻找在不知道 $p(\mathcal{D})$ 的情况下从分布中采样的方法（例如MCMC），或者通过用一个易于处理的分布近似后验（例如变分推断）。

### 复杂且非凸的目标函数形态

对于典型数据集和神经网络 (neural network)架构，似然函数 $p(\mathcal{D} | w)$ 相对于权重 (weight) $w$ 是高度非凸的。这意味着高维权重空间中由似然（以及随之而来的后验）定义的“曲面”是崎岖不平的，包含大量的局部最优、鞍点以及潜在的大片平坦区域。

- **局部最优：** 基于优化的推断方法（例如变分推断中使用的方法或寻找最大后验估计的方法）很容易陷入次优的权重配置。
- **鞍点：** 这些是梯度为零但既非局部最小值也非局部最大值的点。它们在高维空间 (high-dimensional space)中尤其常见，并且可以显著减慢基于梯度的优化方法的速度。
- **采样困难：** 像MCMC这样的采样方法也难以应对复杂的形态。马尔可夫链可能被困在局部模式中，导致对完整后验分布的采样不充分，并产生有偏差的估计。设计能够高效处理这些复杂、高维曲面的采样器是当前研究的一个重要方向（这促成了HMC等方法）。

### 显著的计算成本

即使是为解决上述困难而设计的近似推断方法，也仍然计算成本高昂，特别是对于大型数据集和深度架构。

- **MCMC 方法：** 像哈密顿蒙特卡洛（HMC）这样的技术通常需要在每一步计算对所有参数 (parameter)的对数似然梯度。虽然高效，但这仍然需要通过网络进行反向传播 (backpropagation)。生成数千或数百万个样本以获得准确的后验近似，可能需要大量时间和计算资源（CPU/GPU 小时）。收敛诊断也会增加额外开销。
- **变分推断：** VI 将推断重新定义为优化问题。虽然通常比MCMC快，但它仍然需要迭代的、基于梯度的优化，可能涉及多次遍历数据集（epoch）。此外，设计近似分布族并有效地优化其参数可能很复杂，并且需要仔细调整。

这些困难表明，将标准贝叶斯技术直接应用于深度学习 (deep learning)模型是不可行的。模型的规模及其目标函数的复杂性使得需要专门的、高级的推断技术。接下来的章节将介绍MCMC和VI方法，这些方法专门适用于应对贝叶斯神经网络 (neural network)背景下的这些困难。

## 参考资料

- [Probabilistic Machine Learning: Advanced Topics](https://mitpress.mit.edu/books/probabilistic-machine-learning-advanced-topics) — Kevin P. Murphy (2023)
  Publisher: MIT Press; Pages: Chapter 27: Bayesian Neural Networks
  这本全面的教材权威地介绍了机器学习中的贝叶斯方法，包括贝叶斯神经网络及其推理相关的计算挑战的专门章节。
- [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/abs/1701.02434) — Michael Betancourt (2017)
  Journal: arXiv preprint arXiv:1701.02434; DOI: [10.48550/arXiv.1701.02434](https://doi.org/10.48550/arXiv.1701.02434)
  本教程以易懂且严谨的方式解释了哈密顿蒙特卡洛，阐明了其机制以及在从复杂、高维概率分布中采样时所面临的困难，这与BNN推理直接相关。
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  这篇基础性论文介绍了变分自编码器，通过引入重参数化和随机梯度变分贝叶斯，使变分推理适用于深度神经网络，解决了计算难题。
- [A Principled Evaluation of Approximate Bayesian Inference in Neural Networks](https://proceedings.mlr.press/v119/wenzel20a.html) — Florian Wenzel, Kevin Roth, Thomas Bretschneider, Joerg Liebig, and Lena Schmid (2020)
  Journal: International Conference on Machine Learning (ICML); Publisher: Proceedings of Machine Learning Research; Volume: 119; Pages: 10168-10178; DOI: [10.5555/3455712.3455850](https://doi.org/10.5555/3455712.3455850)
  本文批判性地评估了神经网络中各种近似贝叶斯推理方法，比较了它们的性能，并指出了在高维环境中准确捕捉后验分布的实际挑战。
