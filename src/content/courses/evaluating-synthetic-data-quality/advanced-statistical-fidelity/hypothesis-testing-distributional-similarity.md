---
course: "evaluating-synthetic-data-quality"
chapter: "advanced-statistical-fidelity"
lesson: "hypothesis-testing-distributional-similarity"
sourceId: 5646
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-2-advanced-statistical-fidelity/hypothesis-testing-distributional-similarity"
title: "分布相似性的假设检验"
description: "用于分布匹配的高级假设检验（例如，KS检验变体，卡方检验扩展）。"
order: 2
plots: []
sourceHash: "bfa0112f5d76b129f87f43885ac00582b9a0c4fa9073671878bf297abbfc8585"
sourceCorrections: []
---

虽然比较边际分布和基本统计量可以初步了解数据的逼真度，但这不足以验证合成数据是否真正抓住了真实数据中变量间的关联。我们需要一些方法来评价*联合*概率分布的相似性，$P_{real}(X_1, ..., X_n)$ 和 $P_{synthetic}(X_1, ..., X_n)$。假设检验为这种评估提供了一个正式的统计框架。

在这种情形下，通常的设置包括：

- **零假设 ($H_0$)**: 真实数据集和合成数据集来自同一基础概率分布。 ($P_{real} = P_{synthetic}$)
- **备择假设 ($H_A$)**: 真实数据集和合成数据集来自不同的概率分布。 ($P_{real} \neq P_{synthetic}$)

检验的目的是确定是否有足够的统计证据拒绝 $H_0$ 而支持 $H_A$。较小的p值（通常低于预设的显著性水平，例如0.05）表明，如果数据集确实来自同一分布，那么观察到的数据集间差异不太可能是偶然发生的，这会使我们拒绝 $H_0$ 并认为这些分布是不同的。

有多种统计检验可以进行调整或专门设计，用于比较两个多元样本。下面我们来看一些适用于合成数据评估的重要方法。

### 传统检验的扩展：高维度下的挑战

Kolmogorov-Smirnov (KS)检验和卡方（$\chi^2$）检验等传统检验在单变量分布比较方面已得到充分确立。

- **Kolmogorov-Smirnov (KS) 检验：** 标准的一维KS检验比较两个样本的累积分布函数（CDF）。其检验统计量衡量经验CDF之间的最大绝对差异。虽然对单变量数据有效，但将其直接扩展到多维（>1）时会遇到问题。定义一个独特的多元CDF并计算最大差异变得复杂，并且随着维度增加，检验的统计功效往往迅速下降。实际操作中有时会将一维KS检验应用于数据的投影（例如主成分）或边际分布，但这并不能完全捕捉联合结构。
- **卡方（$\chi^2$）检验：** 卡方拟合优度检验或同质性检验可以通过将数据空间划分为若干区间（bins），并比较落入每个区间的观测数据点频率来比较分布。然而，由于“维度灾难”问题，其在高维度上的有效性显著降低。为了保持每个区间内有合理数量的样本，所需区间的数量随特征数量呈指数增长，这使得它对于大多数多变量数据集而言不切实际，除非维度非常低或数据本身是分类的。区间划分策略的选择也严重影响结果。

考虑到这些局限性，专门为多元比较设计的方法在评估合成数据逼真度时通常更受青睐。

### 最大平均差异 (MMD)

最大平均差异（MMD）是一种非参数 (parameter)度量，它根据样本在再生核希尔伯特空间（RKHS）中的均值嵌入 (embedding)来衡量分布之间的距离。

其思路是使用核函数 $k(\cdot, \cdot)$（例如高斯RBF核）将来自每个分布的数据点映射到可能无限维的特征空间中。MMD随后计算这个RKHS中映射样本均值之间的平方距离。

设 $X = \{x_1, ..., x_m\}$ 为来自 $P_{real}$ 的样本， $Y = \{y_1, ..., y_n\}$ 为来自 $P_{synthetic}$ 的样本。MMD平方的经验估计值如下所示：


$$
MMD^2(X, Y) = \frac{1}{m^2} \sum_{i=1}^m \sum_{j=1}^m k(x_i, x_j) - \frac{2}{mn} \sum_{i=1}^m \sum_{j=1}^n k(x_i, y_j) + \frac{1}{n^2} \sum_{i=1}^n \sum_{j=1}^n k(y_i, y_j)
$$


如果 $P_{real} = P_{synthetic}$，则 $MMD(P_{real}, P_{synthetic}) = 0$。值越大表示差异越大。

**使用MMD进行假设检验：**
可以使用计算出的MMD值作为检验统计量来构建假设检验。显著性（p值）通常通过置换检验来确定：

1. 计算原始样本的 $MMD^2(X, Y)$。
2. 汇集真实数据和合成数据。
3. 重复打乱汇集后的数据，并将其分成两个大小分别为 $m$ 和 $n$ 的新样本。计算这些置换样本的 $MMD^2$。
4. p值是 $MMD^2$ 大于或等于原始 $MMD^2(X, Y)$ 的置换比例。

MMD的优点在于它不需要分箱，在高维度下效果良好，并且根据所选核函数可以捕捉复杂的差异。核函数及其参数（例如RBF核的带宽 $\gamma$）的选择很重要，会影响敏感性。

### 能量距离

能量距离提供了另一个强大的非参数 (parameter)框架，用于检验多元分布的相等性。它基于统计能量思想，与牛顿引力势能相关。E统计量比较来自两个不同样本的点*之间*的距离与每个样本点*内部*的距离。

设 $X = \{x_1, ..., x_m\}$ 来自 $P_{real}$， $Y = \{y_1, ..., y_n\}$ 来自 $P_{synthetic}$。E统计量（与平方能量距离有关）基于欧几里得距离 $|| \cdot ||$ 计算：


$$
E(X, Y) = \frac{2}{mn} \sum_{i=1}^m \sum_{j=1}^n ||x_i - y_j|| - \frac{1}{m^2} \sum_{i=1}^m \sum_{j=1}^m ||x_i - x_j|| - \frac{1}{n^2} \sum_{i=1}^n \sum_{j=1}^n ||y_i - y_j||
$$


与MMD类似，$E(X, Y) \ge 0$，且当且仅当 $P_{real} = P_{synthetic}$ 时（在矩条件成立下），$E(X, Y) = 0$。值越大意味着分布差异越大。

**使用能量距离进行假设检验：**
这里也常用置换检验。针对原始样本计算E统计量 $E(X, Y)$，并将其值与通过置换样本获得的E统计量分布进行比较，以计算p值。

能量距离具有旋转不变性，不需要选择核函数，这使得在某些情况下它比MMD更容易应用，同时在高维度下仍然有效。

### 分类器双样本检验 (C2ST)

检验两个样本是否来自同一分布的一个非常直接的方法是，看机器学习 (machine learning)分类器能否可靠地将它们区分开来。这就是分类器双样本检验（C2ST）的核心思路。

**方法论：**

1. **标记 (token)：** 将真实数据集（$X$）中的所有数据点标记为（例如，0），将合成数据集（$Y$）中的所有数据点标记为另一个标签（例如，1）。
2. **合并与打乱：** 将这些数据集合并成一个包含特征和标签的单一数据集。打乱这个合并后的数据集。
3. **训练/测试拆分：** 将合并后的数据集拆分为训练集和测试集。
4. **训练分类器：** 在训练集上训练一个二元分类模型（例如，逻辑回归、SVM、随机森林、梯度提升），根据特征预测标签（真实 vs. 合成）。
5. **评估：** 使用适当的指标，如准确率或ROC曲线下面积（AUC），评估分类器在保留测试集上的表现。

**使用C2ST进行假设检验：**
分类器在测试集上的表现用作检验统计量。

- 如果 $P_{real} = P_{synthetic}$，则特征不提供关于标签的信息，分类器表现应不佳，理想情况下接近随机猜测（准确率 \u2248 0.5，AUC \u2248 0.5）。
- 如果 $P_{real} \neq P_{synthetic}$，一个足够强大的分类器应该能够区分这些样本，取得明显优于随机机会的表现（准确率 > 0.5，AUC > 0.5）。

p值可以通过对标签进行置换检验或在零假设下分析性能指标的分布来估计。显著高于0.5的高准确率或AUC提供了拒绝 $H_0$ 的证据。

> 图示分类器双样本检验（C2ST）的工作流程。真实数据和合成数据被标记、合并、拆分，并用于训练和评估分类器。高分类表现表明分布是不同的。

C2ST具有灵活性，因为它可以使用多种分类器。此外，如果分类器表现良好，特征重要性分析有时可以提供关于真实数据集和合成数据集之间*哪些*特征或关联差异最大的信息。

### 实际考虑与解释

在使用假设检验评估分布相似性时：

- **检验选择：** 选择取决于数据特征（维度、样本量、连续/分类特征）和计算成本。MMD、能量距离和C2ST通常是中高维度连续或混合数据的良好选择。
- **显著性水平 ($\\alpha$)：** 事先选择一个显著性水平（例如，$\\alpha = 0.05$）。如果p值小于 $\\alpha$，则拒绝 $H_0$。
- **p值解释：** 请记住，p值是在*零假设为真*的情况下，观测到至少与测量结果一样极端数据的概率。
  - **较小的p值**（$< \\alpha$）提供了*反对* $H_0$ 的证据，表明分布是不同的。
  - **较大的p值**（$\ge \\alpha$）表示您*未能拒绝* $H_0$。这并**不**证明分布是相同的，而只是表明在所选显著性水平和给定数据下，检验未能找到足够的证据得出它们不同的结论。没有证据并不意味着不存在。
- **样本量：** 样本量越大，假设检验越敏感。对于非常大的数据集，即使 $P_{real}$ 和 $P_{synthetic}$ 之间存在微小且实际无关紧要的差异，也可能导致统计学上显著的p值（拒绝 $H_0$）。因此，除了p值之外，还应始终考虑差异的大小（效应量），这通常由检验统计量本身表示（例如，MMD值、E统计量或分类器AUC）。统计学上显著的结果可能在实践中不显著。
- **背景重要：** 将结论与预期的使用场景联系起来。敏感检验检测到的统计显著差异是否真正影响下游效用或隐私目标？C2ST AUC为0.6可能在数据量足够大的情况下具有统计显著性，但可能表示数据集在实际应用中仍相当相似。

假设检验提供了一种严谨的方法，可以超越简单的统计数据，评估联合分布的逼真度。它们提供了量化 (quantization)证据，但应仔细解读，同时考虑样本量、效应量和整体评估目标。

## 参考资料

- [A Kernel Two-Sample Test](http://jmlr.org/papers/v13/gretton12a.html) — Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, Alexander Smola (2012)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 13; Pages: 723-773
  介绍最大均值差异（MMD）作为基于核函数的双样本检验，用于比较分布的开创性论文。
- [E-distance: A new method for testing equality of distributions](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-1/issue-2/E-distance-A-new-method-for-testing-equality-of-distributions/10.1214/07-AOAS161.full) — Gábor J. Székely, Maria L. Rizzo (2007)
  Journal: The Annals of Applied Statistics; Publisher: Institute of Mathematical Statistics; Volume: 1; Pages: 653-672; DOI: [10.1214/07-AOAS161](https://doi.org/10.1214/07-AOAS161)
  介绍能量距离及其在检验多元分布相等性方面的确切论文。
- [How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models](https://arxiv.org/abs/2102.08921) — Ahmed M. Alaa, Boris van Breugel, Evgeny Saveliev, Mihaela van der Schaar (2021)
  Journal: Proceedings of the 39th International Conference on Machine Learning; Publisher: PMLR; Volume: 162; Pages: 290-306
  对评估合成表格数据质量的指标和考虑因素的全面评论，包括MMD、能量距离和基于分类器的方法。
