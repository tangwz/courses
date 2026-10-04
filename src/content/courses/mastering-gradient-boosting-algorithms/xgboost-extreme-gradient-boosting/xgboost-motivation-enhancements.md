---
course: "mastering-gradient-boosting-algorithms"
chapter: "xgboost-extreme-gradient-boosting"
lesson: "xgboost-motivation-enhancements"
sourceId: 1977
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-4-xgboost-extreme-gradient-boosting/xgboost-motivation-enhancements"
title: "GBM 的原理及优化"
description: "了解 XGBoost 相较于标准 GBM 带来的主要优化。"
order: 1
plots: []
sourceHash: "7f9edbf17ac2edafbe4e81acc4d8547d4868ff53f8c15760c84c38e4ca475969"
sourceCorrections: []
---

标准梯度提升机（GBM）框架为构建预测模型提供了一种强效方法。然而，其实际应用显现出显著的优化空间，尤其在计算效率和过拟合 (overfitting)控制方面。陈天奇开发XGBoost（极端梯度提升）正是受这些实际考量驱动，旨在构建一个更具扩展性、高效且正则化 (regularization)的梯度提升库。

标准GBM实现在处理超大数据集时常遇难题。在每个节点对每个特征遍历潜在分割点可能成为计算瓶颈。此外，尽管收缩（shrinkage）和子采样（subsampling）等技术有助于防止过拟合，但它们常作为启发式方法应用，而非与构建每棵树的优化目标紧密关联。

XGBoost 通过几项精心设计的优化解决了这些局限性：

### 形式化正则化 (regularization)

XGBoost 的一个主要区别是其**正则化学习目标**。与标准GBM中正则化常依赖于定义基本提升步骤*之后*（如限制树的深度或使用收缩）应用的约束不同，XGBoost 将正则化项直接纳入到构建每棵树时优化的目标函数中。具体来说，它在损失函数 (loss function)中增加了类似于 $L_1$ (Lasso) 和 $L_2$ (Ridge) 正则化的惩罚项。

这种集成方法意味着，在树构建过程中，分割的选择和叶子值的计算明确考虑了模型的复杂性。目标函数平衡了损失最小化（模型与数据拟合的程度）与新添加树的复杂性最小化（通过叶子数量和叶子权重 (weight)的幅值衡量）。与仅仅依赖最大深度等启发式方法相比，这种形式化方法提供了一种更有原则的过拟合 (overfitting)控制方法。在下一节中我们将分析这个目标函数的数学细节。

### 先进的分割点查找算法

为降低寻找最优分割点的计算成本，尤其是在特征或实例数量很多的情况下，XGBoost 实现了先进且高效的算法：

1. **精确贪婪算法：** 尽管仍是贪婪的，但它能高效地评估分割点。
2. **近似算法：** 对于枚举所有可能分割点不可行的真正庞大数据集，XGBoost 使用基于特征值百分位数（分位数）的近似算法。它根据特征值的分布提出候选分割点，并仅在这些点评估分割，从而大幅减少了计算量。
3. **稀疏感知分割点查找：** XGBoost 内置的缺失值（稀疏性）处理是一项值得注意的特色。XGBoost 无需事先进行插补，而是在训练期间为每个节点的缺失值学习一个默认方向。它通过评估将包含缺失值的实例放入左子节点与右子节点所获得的增益，并选择使增益最大的方向来实现。与手动插补相比，这种方法既方便又常能带来更好的模型性能。

### 系统效率优化

除了算法上的改进，XGBoost 在设计时就考虑了性能和可扩展性：

- **并行化：** XGBoost 在树构建过程中可以利用多个 CPU 核心进行并行计算。这种并行性主要发生在分割点查找阶段，此时不同特征或潜在分割点的计算可以被分配处理。
- **缓存感知访问：** 其内部数据结构和算法旨在优化 CPU 缓存的使用，最大程度地减少耗时的内存访问延迟。数据以与缓存行对齐 (alignment)的特定块结构存储。
- **核外计算：** XGBoost 可以通过从磁盘分块处理数据来处理不适合主内存的数据集，使其能够扩展到超大数据集。

这些优化共同使得 XGBoost 比许多传统 GBM 实现明显更快、更具扩展性，同时其集成的正则化 (regularization)常能带来更好的泛化性能。它代表了梯度提升技术的一大进步，将理论上的改进与精心的系统设计结合起来。在后续章节中我们将更详细地分析这些特性。

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) — Tianqi Chen and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  描述了XGBoost的核心算法，包括其正则化目标函数、高级分裂查找算法以及用于可伸缩性和效率的系统优化。
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/) — XGBoost Developers (2024)
  提供XGBoost设计原则、正则化技术和效率特性的最新信息，通过实用细节补充了基础论文。
- [Hands-On Machine Learning with Scikit-Learn, Keras, & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供了梯度提升概念的易懂解释，并详细介绍了XGBoost的改进，如正则化和稀疏性处理，提供实用视角。
