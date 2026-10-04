---
course: "model-interpretability-explainability"
chapter: "shap-additive-explanations"
lesson: "shap-values-overview"
sourceId: 4075
sourceUrl: "https://apxml.com/zh/courses/model-interpretability-explainability/chapter-3-shap-additive-explanations/shap-values-overview"
title: "SHAP 值：Shapley 值与模型特征的关联"
description: "了解 SHAP 如何调整 Shapley 值，以分配预测对模型特征的作用。"
order: 2
plots: []
sourceHash: "668cb0e3a4d8d061ca8956a2e918c1ce62899a42290f65652ed41b9716e5e62d"
sourceCorrections: []
---

合作博弈论中的 Shapley 值是一种公平地在合作参与者之间分配“报酬”的方法。SHAP (SHapley Additive exPlanations) 框架有效地将这种思想应用于机器学习 (machine learning)模型解释的背景。

试想一下，如何解释你的训练模型对某个特定输入实例所做的一个预测。该实例的每个特征对最终预测值贡献了多少，与某个基准相比？SHAP 通过将此问题设定为一个合作博弈来给出回答：

- **博弈：** 生成特定实例 $x$ 预测的过程。
- **参与者：** 实例 $x$ 的各个特征（例如，年龄、收入、邮政编码）。
- **报酬：** 模型对实例 $x$ 的实际预测（记作 $f(x)$）与整个训练数据集（或代表性背景数据集）的平均预测（记作 $E[f(X)]$）之间的差值。这个差值，$f(x) - E[f(X)]$，表示实例 $x$ 所有特定特征的总贡献，这些特征使预测值偏离了平均值。

目标是将总报酬（$f(x) - E[f(X)]$）公平地分配给“参与者”（特征）。一个特征的贡献，即其 SHAP 值，表示该特定特征的值在该实例其他特征值的背景下，使预测值产生了多大程度的偏移。

正式地，特征 $i$ 和实例 $x$ 的 SHAP 值 $\phi_i(x)$ 被计算为该特征在所有不包含特征 $i$ 的可能特征子集（联盟）中的边际贡献的加权平均值。特征 $i$ 对特定联盟 $S$（$S$ 是不包含特征 $i$ 的特征子集）的“边际贡献”是模型在将特征 $i$ 添加到该联盟后的预期输出与仅有联盟 $S$ 时的预期输出之间的差值：

$E[f(X) | X_S \cup \{x_i\}] - E[f(X) | X_S]$

在这里，$E[f(X) | X_S]$ 表示仅给定子集 $S$ 中特征值的预期预测。计算这种条件期望通常是最复杂的部分，并且需要对特征独立性做出假设或使用特定的计算方法，我们将在讨论 KernelSHAP 和 TreeSHAP 时介绍这些。

SHAP 值从 Shapley 值继承了一个非常好的属性，称为**可加性**。这表示给定实例 $x$ 所有特征的 SHAP 值之和等于该实例的预测值 $f(x)$ 与基准值（平均预测值 $E[f(X)]$）之间的差值。我们可以将这种关系表示为：


$$
f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)
$$


这里：

- $f(x)$ 是模型对实例 $x$ 的预测。
- $\phi_0 = E[f(X)]$ 是基准值，代表背景数据集上的平均模型预测。
- $\phi_i(x)$ 是特定实例 $x$ 的特征 $i$ 的 SHAP 值。
- $M$ 是特征的总数量。

这个等式提供了对预测的清晰分解。$\phi_0$ 给出起点（平均预测），每个 $\phi_i(x)$ 说明特征 $i$ 的值如何相对于这个基准值将预测推高（如果 $\phi_i(x)$ 为正）或推低（如果 $\phi_i(x)$ 为负）。大小 $|\phi_i(x)|$ 表明特征对该特定预测影响的强度。

与 LIME 不同，LIME 构建了一个局部、可解释的*代理*模型，而 SHAP 基于扎实的博弈论原则直接计算特征贡献，旨在获得更具理论支撑的贡献划分。然而，对于除了最简单的模型和最小的特征集之外的所有情况，计算所有可能特征联盟的这些精确贡献在计算上是不可行的。这种计算挑战促使了高效近似算法（如 KernelSHAP 和 TreeSHAP）的提出，我们将在下文介绍这些算法。

## 参考资料

- [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) — Scott M. Lundberg, Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 4765-4774; DOI: [10.5555/3295222.3295230](https://doi.org/10.5555/3295222.3295230)
  介绍 SHAP 框架、其博弈论特性以及 KernelSHAP 和 TreeSHAP 等近似算法的奠基性论文。
- [A Value for n-person Games](https://www.rand.org/content/dam/rand/pubs/papers/2006/P295.pdf) — Lloyd S. Shapley (1953)
  Journal: Contributions to the Theory of Games, Volume II; Publisher: Princeton University Press; Volume: II; Pages: 307-317; DOI: [10.1515/9781400881970-018](https://doi.org/10.1515/9781400881970-018)
  介绍合作博弈论中 Shapley 值的原始论文，SHAP 正是基于此。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/shap.html) — Christoph Molnar (2024)
  一本全面的开源书籍，其中有一章专门详细介绍 SHAP 值、其属性和实际应用。
- [SHAP (SHapley Additive exPlanations) GitHub Repository](https://github.com/shap/shap) — Scott Lundberg, et al. (2024)
  SHAP 官方开源 Python 库，提供各种模型和算法的实现以及文档。
