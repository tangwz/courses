---
course: "mastering-gradient-boosting-algorithms"
chapter: "catboost-gradient-boosting"
lesson: "catboost-ordered-ts"
sourceId: 2009
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-6-catboost-gradient-boosting/catboost-ordered-ts"
title: "有序目标统计 (Ordered TS)"
description: "详细解释 CatBoost 处理类别特征的主要技术。"
order: 2
plots: []
sourceHash: "7e36031bb89a85d33fb2a66835b114c7b45457cedaa617960a1cb3e0157d81d9"
sourceCorrections: []
---

有效处理类别特征是许多机器学习 (machine learning)任务中面临的一个主要难题，特别是在增强（boosting）框架内。传统方法常有不足。独热编码虽然直接，但在处理高基数特征（具有许多独特类别的特征）时，计算成本高昂，并可能导致稀疏性问题。另一种常见方法是标准目标编码（也称为均值编码），它将每个类别替换为训练数据中该类别的平均目标值。然而，此方法存在一个主要缺陷：**目标泄漏**。通过使用一个数据点的目标值来计算该数据点的特征编码，我们无意中将目标信息引入特征，导致训练时性能估计过于乐观，并且对未见过的数据泛化能力较差。

CatBoost 引入了\*\*有序目标统计（Ordered TS）\*\*作为一种精巧且有效的解决方案来处理此问题。其核心思想是在计算每个数据点的目标统计时*不*使用其自身的目标值，从而避免直接泄漏。这通过对数据施加特定顺序来实现。

### 有序目标统计机制

有序目标统计并非使用整个数据集来计算给定类别的目标统计，而是使用对训练数据施加的人工“时间”或顺序。其工作方式如下：

1. **随机排列：** 首先，训练数据集被随机打乱，生成数据索引 $\{1, ..., n\}$ 的一个排列 $\sigma$。这个排列定义了一个顺序，类似于时间序列，其中如果 $k > j$，则样本 $\sigma(k)$ 出现在样本 $\sigma(j)$ “之后”。
2. **有序计算：** 对于此序列中的每个样本 $x_{\sigma(k)}$ 及其每个类别特征 $i$，目标统计 $TS_{k,i}$ 仅使用排列中其*前面*的样本（即 $j < k$ 的样本）的目标值 $y_{\sigma(j)}$ 来计算。非常重要的一点是，当前样本的目标值 $y_{\sigma(k)}$ 从此计算中**排除**。

样本 $k$ 和特征 $i$ 的有序目标统计公式通常包含平滑处理：


$$
TS_{k,i} = \frac{\sum_{j=1}^{k-1} [x_{\sigma(j), i} = x_{\sigma(k), i}] \cdot y_{\sigma(j)} + \alpha \cdot \text{prior}}{\sum_{j=1}^{k-1} [x_{\sigma(j), i} = x_{\sigma(k), i}] + \alpha}
$$


我们来分解一下这个公式：

- $[x_{\sigma(j), i} = x_{\sigma(k), i}]$: 这是一个指示函数。如果排列中第 $j$ 个样本的第 $i$ 个类别特征与第 $k$ 个样本的第 $i$ 个特征具有相同的类别值，则它等于 1，否则等于 0。
- $y_{\sigma(j)}$: 排列中第 $j$ 个样本的目标值。
- $\sum_{j=1}^{k-1} ...$: 求和只进行到 $k-1$，确保只考虑在排列 $\sigma$ 中出现在当前样本 $k$ *之前*的样本。
- $\alpha$: 一个平滑参数 (parameter)（正权重 (weight)）。
- $\text{prior}$: 一个先验值，通常设置为整个训练数据集的总体平均目标值。

项 $\alpha \cdot \text{prior}$ 起到正则化 (regularization)器的作用。它在排列初期（$k$ 较小）或对于稀有类别尤其重要，因为这些情况下可能只有很少（或没有）具有相同类别值的先行样本。平滑处理可以防止编码值变得过于嘈杂，或过度依赖一两个之前的例子。当局部信息稀缺时，它将估计值“拉向”总体平均目标值。

> 样本 $x_{\sigma(k)}$ 的有序目标统计计算的依赖流程。该计算仅使用排列 $\sigma$ 中先行样本（$j < k$）中与特征 $i$ 共享相同类别的目标值（$y$），并结合先验值进行平滑处理。目标值 $y_{\sigma(k)}$ 未被使用。

### 通过多次排列增强稳定性

依赖单个随机排列可能会根据生成的特定顺序引入自身的偏差。为了减轻这种情况，CatBoost 通常在训练集成模型时生成多个随机排列。在构建增强模型中的不同树时，会使用不同的排列。这种跨各种顺序的平均效果使生成的目标统计数据更加稳定。此机制与\*\*有序增强（Ordered Boosting）\*\*密切相关，我们将在下一节中讨论，因为它确保了每一步训练的模型不会因编码过程引起的预测偏移而受到影响。

### 处理测试数据

在对新的、未见过的数据（测试集）进行预测时，我们不能简单地将其附加到训练排列中并计算统计数据，因为这会违反仅使用过去信息的原则。相反，CatBoost 通常使用在训练期间学习到的目标统计数据。对于测试样本中观察到的给定类别，其编码可能使用属于该类别的*所有*训练样本（通常仍然包含先验和平滑处理）来计算，或者通过使用在训练阶段存储的预计算统计数据来计算。具体方法可能取决于 CatBoost 的实现细节和参数 (parameter)。

### 有序目标统计的优点

- **减少目标泄漏：** 通过在编码计算中明确排除当前样本的目标值，有序目标统计显著缓解了标准目标编码固有的目标泄漏问题。
- **对高基数有效：** 它为类别特征提供了具有数值意义的编码，而无需诉诸于独热编码等高维稀疏表示。
- **集成方法：** CatBoost 将这种精巧的编码策略直接整合到算法中，简化了用户的预处理流程。

有序目标统计代表了梯度增强模型从类别数据中学习方式的重大改进，为 CatBoost 的强大性能奠定了基础，尤其是在富含此类特征的数据集上。

## 参考资料

- [CatBoost: Unbiased Gradient Boosting with Categorical Features](https://proceedings.neurips.cc/paper/2018/hash/f678ed707a0279ee66b4d32049d5224fa-Abstract.html) — Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin (2018)
  Journal: Advances in Neural Information Processing Systems 31 (NeurIPS 2018); Publisher: NeurIPS; Volume: 31
  本文介绍了CatBoost及其处理分类特征的方法，包括有序目标统计。
- [Categorical Features Processing](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGf6FKI64e9isHfL4pauKx3D6NP4dXOkEek6x1WYXJwP2OrkOB9FBeSvjRS8d8POKMOAzy1ZqT3QgYzjtuv2ibkQwDd8NzhFGBF4CVtHFQSfdj7Nv1QiTFt1tSrpZ5sJSeTP7Pk5PFX2B5BTznnkTrQ_DWIUbwmheDDSAR_rb6Np-jo5A==) — CatBoost Developers (2024)
  Publisher: Yandex
  官方文档，描述了CatBoost将分类特征转换为数值特征的策略，包括有序目标统计。
