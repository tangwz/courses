# Shapley 值概述

来源：[原文](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-3-shap-additive-explanations/shapley-values-introduction)

[返回章节目录](README.md) · [返回课程目录](../README.md)

SHapley Additive exPlanations (SHAP) 提出了一种源于合作博弈论的方法，用于模型可解释性。这种方法提供了一种有原则的方式来理解各个特征如何对模型的预测做出贡献。SHAP 的核心在于 **Shapley 值**，它提供了一种公正的方式，来公平分配合作博弈中“收益”给其参与者。

### 合作博弈类比

想象一群人合作完成一个项目，集体努力产生了某种价值或结果。我们如何才能公平地确定每个人对最终结果的贡献大小？这是合作博弈论所要回答的核心问题，而 Lloyd Shapley 在 20 世纪 50 年代提出了一种解决方案，现在被称为 Shapley 值。

核心思想是考虑每个参与者（人）加入其他参与者不同可能小组（联盟）时的边际贡献。一个参与者的 Shapley 值是他们在所有可能的加入顺序或序列中的平均边际贡献。

我们来考虑一个简单的例子。假设有三位朋友，爱丽丝 (A)、鲍勃 (B) 和查理 (C)，他们合作进行一个小型咨询项目。

- 单独工作，A 产生 $10k，B 产生$15k，C 产生 \$8k。
- 两人合作：{A, B} 产生 $30k，{A, C} 产生$25k，{B, C} 产生 \$35k。
- 三人共同工作 {A, B, C} 产生 \$50k。

最终的 \$50k 应该如何分配给每个人？Shapley 值提供了一种计算方式。我们查看他们可能加入“大联盟” {A, B, C} 的所有可能顺序（排列），并计算每个人在每个顺序中的边际贡献。

1. **顺序 (A, B, C):**

   - A 首先加入：贡献 = $10k$（{A} 的价值 - {} 的价值）
   - B 接着加入：贡献 = $20k$（{A, B} 的价值 - {A} 的价值 = $30k -$10k）
   - C 最后加入：贡献 = $20k$（{A, B, C} 的价值 - {A, B} 的价值 = $50k -$30k）
2. **顺序 (A, C, B):**

   - A 首先加入：贡献 = $10k$
   - C 接着加入：贡献 = $15k$（{A, C} 的价值 - {A} 的价值 = $25k -$10k）
   - B 最后加入：贡献 = $25k$（{A, C, B} 的价值 - {A, C} 的价值 = $50k -$25k）

我们会对所有 $3! = 6$ 种可能的顺序重复此过程：(A, B, C), (A, C, B), (B, A, C), (B, C, A), (C, A, B), (C, B, A)。

一个参与者的 Shapley 值是他们在所有这些顺序中边际贡献的平均值。我们为爱丽丝 (A) 计算一下：

| 顺序 | A 的边际贡献 |
| --- | --- |
| (A, B, C) | $10k$ |
| (A, C, B) | $10k$ |
| (B, A, C) | $15k$ ($30k - 15k$) |
| (B, C, A) | $15k$ ($50k - 35k$) |
| (C, A, B) | $17k$ ($25k - 8k$) |
| (C, B, A) | $15k$ ($50k - 35k$) |
| **平均值** | **$13.67k$** |

同理，我们可以计算鲍勃和查理的 Shapley 值。这种方法确保了基于平均边际影响的公平分配。

### 连接 Shapley 值与模型预测

这与解释机器学习 (machine learning)模型有何关联？SHAP 框架巧妙地调整了这一方法：

- **博弈：** “博弈”是对*单个样本*的预测任务（例如，预测特定申请人贷款违约的概率）。
- **参与者：** “参与者”是该样本的*特征值*（例如，申请人的收入、信用评分、年龄、贷款金额）。
- **收益：** “收益”是模型对该特定样本的预测与整个数据集（或代表性背景数据集）的*基线*或平均预测之间的差值。这个差值表示所有特征对该特定预测的总贡献，使其偏离平均值。
- **Shapley 值：** 针对给定预测，特定特征（例如“收入”）的 Shapley 值表示该特征对将预测结果推离基线值的贡献。这是将预测差异公平地归因于每个特征的数值。

例如，如果平均预测违约概率是 0.10，而对于特定申请人，模型预测 0.75，则待分配的总“收益”是 $0.75 - 0.10 = 0.65$。SHAP 的目标是计算每个特征（收入、信用评分等）对这个 $0.65$ 差异的贡献大小。特征的 Shapley 值为正表示它使预测值更高（在此例中倾向于违约），而负值则表示它使预测值更低。

### Shapley 值的优势

为什么采用这种博弈论方法？Shapley 值具有一些理想的数学特性（我们将在“SHAP 值的属性”部分进行讨论），这使其在特征归因方面具有理论上的合理性：

1. **效率（局部精确性）：** 给定样本所有特征的 Shapley 值之和等于该样本的预测值与基线预测值之间的差。这些贡献可以完全相加。
2. **对称性：** 如果两个特征对所有可能的联盟贡献相同，它们将获得相同的 Shapley 值。
3. **空特征：** 一个从不改变预测结果的特征（无论它被添加到哪个联盟中）其 Shapley 值为零。
4. **可加性：** 对于预测结果被组合（例如平均或求和）的集成模型，Shapley 值也可以一致地组合。

“计算精确的 Shapley 值需要评估模型对每个可能的特征子集（联盟）的输出。对于具有许多特征 ($M$) 的模型，这涉及 $2^M$ 次评估，很快就会变得计算上难以处理。这就是实用型 SHAP 算法，如 KernelSHAP 和 TreeSHAP 发挥作用的地方。它们提供了有效的方法来*估计*这些具有理论依据的 Shapley 值，使得这种有用的原理可以应用于机器学习 (machine learning)问题。”

在接下来的部分中，我们将了解 SHAP 值是如何使用这些实用算法计算和解释的。

## 参考资料

- [A Value for N-person Games](https://doi.org/10.1515/9781400881970-018) — Lloyd S. Shapley (1953)
  Journal: Contributions to the Theory of Games, Volume II; Publisher: Princeton University Press; Volume: II; Pages: 307-317; DOI: [10.1515/9781400881970-018](https://doi.org/10.1515/9781400881970-018)
  介绍了Shapley值，用于公平分配合作博弈中的收益。
- [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) — Scott M. Lundberg, Su-In Lee (2017)
  Journal: Neural Information Processing Systems; Volume: 30; Pages: 4768-4777; DOI: [10.48550/arXiv.1705.07874](https://doi.org/10.48550/arXiv.1705.07874)
  介绍了SHAP，一个基于Shapley值和高效近似方法的局部模型可解释性统一框架。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/) — Christoph Molnar (2023)
  一本广泛引用的在线教材，提供了SHAP和其他模型可解释性方法的易懂说明。

---

[上一节](../02-%E5%B1%80%E9%83%A8%E5%8F%AF%E8%A7%A3%E9%87%8A%E6%A8%A1%E5%9E%8B%E6%97%A0%E5%85%B3%E8%A7%A3%E9%87%8A%20%28LIME%29/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%94%9F%E6%88%90%20LIME%20%E8%A7%A3%E9%87%8A.md) · [下一节](02-SHAP%20%E5%80%BC%EF%BC%9AShapley%20%E5%80%BC%E4%B8%8E%E6%A8%A1%E5%9E%8B%E7%89%B9%E5%BE%81%E7%9A%84%E5%85%B3%E8%81%94.md)
