---
course: "mastering-gradient-boosting-algorithms"
chapter: "xgboost-extreme-gradient-boosting"
lesson: "xgboost-sparsity-aware"
sourceId: 1985
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-4-xgboost-extreme-gradient-boosting/xgboost-sparsity-aware"
title: "稀疏感知分裂查找"
description: "XGBoost 如何有效率地应对缺失值和稀疏数据。"
order: 5
plots: []
sourceHash: "ebb4ce255ba08a78874de82440390df4c1f39b64703a940e21ad2ae14e589121"
sourceCorrections: []
---

数据集经常包含缺失值，这给许多机器学习 (machine learning)算法带来难题。标准决策树的实现常需要预处理步骤，例如填充（补全缺失值），或者以基本方式处理缺失数据，这可能丢弃有用的信息或引入偏差。XGBoost 包含一种巧妙而表现良好的内置机制，称为稀疏感知分裂查找，用于在树构建时直接处理这些情况。

主要思想出乎意料地简单：在树节点中的每个可能分裂点，XGBoost 会为那些缺少待分裂特征值的实例学习一个 *默认方向*。XGBoost 不要求填充或简单地忽略缺失值，而是明确考虑这些实例应该去哪里（左子节点或右子节点），以获得最大的增益，并将此决策直接纳入分裂查找过程。

### 机制：学习默认方向

回想上一节（“分裂查找算法：精确贪心”）中提到的，XGBoost 通过计算在当前节点中依据特征 $j$ 和分裂值 $v$ 划分实例 ($I$) 所带来的增益来衡量可能的分裂。增益的计算用到节点中实例损失函数 (loss function)的一阶和二阶梯度统计量 ($g_i$ 和 $h_i$)。

在衡量分裂候选 $(j, v)$ 时，稀疏感知算法按以下步骤进行：

1. **初步划分：** 只列举当前节点中特征 $j$ 的值 *非缺失* 的实例。将这些实例划分为两个集合：$I_L = \{i \in I | x_{ij} < v\}$ 和 $I_R = \{i \in I | x_{ij} \ge v\}$。仅根据非缺失实例，计算这两个集合的梯度和 ($G_L = \sum_{i \in I_L} g_i$, $G_R = \sum_{i \in I_R} g_i$) 和 Hessian 和 ($H_L = \sum_{i \in I_L} h_i$, $H_R = \sum_{i \in I_R} h_i$)
2. **识别缺失实例：** 令 $I_{missing} = \{i \in I | x_{ij} \text{ 缺失}\}$。计算这些实例的梯度和 $G_{missing} = \sum_{i \in I_{missing}} g_i$ 和 Hessian 和 $H_{missing} = \sum_{i \in I_{missing}} h_i$。
3. **衡量默认左方向：** 计算如果 $I_{missing}$ 中所有实例都发送到 *左* 子节点时的潜在增益。在这种情况下，左右子节点的总梯度和 Hessian 分别为 $(G_L + G_{missing}, H_L + H_{missing})$ 和 $(G_R, H_R)$。使用常用增益公式（包含正则化 (regularization)项 $\lambda$ 和 $\gamma$）计算增益：
   
   $$
   \text{增益}_{\text{默认左}} = \frac{1}{2} \left[ \frac{(G_L + G_{missing})^2}{H_L + H_{missing} + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R + G_{missing})^2}{H_L + H_R + H_{missing} + \lambda} \right] - \gamma
   $$
   
4. **衡量默认右方向：** 计算如果 $I_{missing}$ 中所有实例都发送到 *右* 子节点时的潜在增益。总梯度和 Hessian 将分别为 $(G_L, H_L)$ 和 $(G_R + G_{missing}, H_R + H_{missing})$。计算增益：
   
   $$
   \text{增益}_{\text{默认右}} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{(G_R + G_{missing})^2}{H_R + H_{missing} + \lambda} - \frac{(G_L + G_R + G_{missing})^2}{H_L + H_R + H_{missing} + \lambda} \right] - \gamma
   $$
   
5. **选取最佳方向和增益：** 比较 $\text{增益}_{\text{默认左}}$ 和 $\text{增益}_{\text{默认右}}$。产生更高增益的方向，即成为此节点缺失值的学习默认方向，*适用于当前分裂候选 $(j, v)$*。这两个增益的最大结果被视为此分裂候选所实现的目标函数减少量（增益）。

XGBoost 对所有特征的每个潜在分裂点都进行此默认方向衡量。在节点中，选定能带来整体最大化增益的分裂（特征 $j$、值 $v$ 以及为缺失值确定的默认方向）。

### 稀疏感知处理的优点

这种集成方法提供了一些显著优点：

- **无需预先填充：** 避免了训练前可能繁琐且包含假设的数据填充步骤。
- **数据驱动的处理方式：** 模型直接依据使损失函数 (loss function)达到最优，学习每种分裂处理缺失值的最佳方式，而不是采用均值/中位数填充等启发式方法。
- **高效：** 该算法设计高效。它通过先处理非缺失值，然后将缺失值的两个潜在方向作为一个整体进行衡量，从而减少了冗余计算。这通常比遍历所有可能的缺失数据点赋值要快得多。
- **应对不同的稀疏模式：** 它自然匹配特征间不同的缺失模式。

### 缺失值预测

在预测时，当一个实例遇到分裂节点，且其分裂特征的值缺失时，它只需遵从在训练期间为该节点学习并存储的默认方向（左或右）。

稀疏感知分裂查找是一种计算上有效率且统计学上合理的方法，它对 XGBoost 的性能和稳固性有很大助益，特别是在缺失值普遍存在的数据集上，例如来源于调查、传感器读数或某些类型的特征工程的数据集。它是让 XGBoost 与常规梯度提升算法与众不同的优化之一。

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785) — Tianqi Chen and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16); Publisher: Association for Computing Machinery; Pages: 785–794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  这篇基础论文介绍了XGBoost算法，详细阐述了稀疏感知分裂查找机制作为处理缺失数据的一项核心优化。
- [Handling Missing Values](https://xgboost.readthedocs.io/en/stable/treemethod.html#sparsity-aware-split-finding) — XGBoost Contributors (2023)
  官方文档提供了XGBoost如何实现稀疏感知分裂查找和处理缺失值的实用细节和最新说明。
- [Classification and Regression Trees](https://doi.org/10.1201/9781315139470) — Leo Breiman, Jerome Friedman, Richard Olshen, and Charles Stone (1984)
  Publisher: Chapman and Hall/CRC; Pages: Chapter 4; DOI: [10.1201/9781315139470](https://doi.org/10.1201/9781315139470)
  这本基础著作介绍了CART算法，其中包括决策树中处理缺失值的早期方法（如替代分裂），为理解XGBoost的独特方法提供了背景。
