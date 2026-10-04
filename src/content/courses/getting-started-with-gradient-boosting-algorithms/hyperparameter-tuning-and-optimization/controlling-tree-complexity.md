---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "hyperparameter-tuning-and-optimization"
lesson: "controlling-tree-complexity"
sourceId: 7607
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-6-hyperparameter-tuning-and-optimization/controlling-tree-complexity"
title: "控制树的复杂度"
description: "学习如何使用 max_depth、min_child_weight 和 min_samples_split 等参数来控制单个树的复杂度并防止过拟合。"
order: 4
plots: ["plots/7607-0.json"]
sourceHash: "ef27450df9cdc46301fe32710220ca953cc0ca18cb337b8eab21309fbc1ac308"
sourceCorrections: []
---

虽然树的数量和学习率决定了学习过程的整体规模和速度，但梯度提升模型的表现很大程度上受其单个弱学习器特点的影响。如果序列中的每棵树过于复杂，它可能迅速对训练数据过拟合 (overfitting)，记住噪声而不是学习可泛化的模式。相反，如果树过于简单，模型可能缺乏捕捉数据中必要关系的能力，导致高偏差。目标是限制每棵树的复杂度，确保它能为整体贡献少量而稳定的提升，同时不引入显著的方差。

### 控制树的深度和大小

管理树复杂度最直接的方法是限制其大小。过深的树是基于树的模型中过拟合 (overfitting)的主要原因。通过限制它们的生长，我们促使模型建立更具泛化性的数据表示。

#### 最大树深度 (`max_depth`)

`max_depth` 超参数 (parameter) (hyperparameter)对任何单个决策树可以生长的深度设置了绝对限制。深度为 $d$ 的树最多可以有 $2^d$ 个叶节点。这个参数对于控制模型复杂度并防止过拟合是最有影响力的之一。

- **低 `max_depth` (例如，2-4)：** 产生更简单的树，它们只捕获数据中最明显的相互作用。这会导致一个正则化 (regularization)程度更高的模型，具有较低的方差但可能较高的偏差。
- **高 `max_depth` (例如，8-16 或更多)：** 允许模型学习高度特定和复杂的模式。这可能导致过拟合，即模型在训练数据上表现极佳，但未能泛化到新的、未见过的数据。

对于大多数应用，`max_depth` 的一个好的起始范围是 3 到 7 之间。

> 深度为 2 的浅层树具有有限的决策规则数量，而更深的树可以捕捉更精细的相互作用，增加过拟合的风险。

随着 `max_depth` 的增加，模型在训练集上的表现几乎总会提高。然而，它在独立验证集上的表现才是真正重要的。下面呈现了这种经典的偏差-方差权衡。



![max_depth 对模型误差的影响](plots/7607-0.json)



> 随着树深度增加，训练误差持续下降。验证误差会改善到某一点（最佳深度），然后随着模型开始过拟合而开始恶化。

#### 其他限制大小的参数

虽然 `max_depth` 很常用，但有些库提供了其他限制树大小的方法：

- `max_leaf_nodes`：限制一棵树中终端节点（叶子）的总数。这可以是一种更直接的控制复杂度的方法，因为它不依赖于树结构是否对称。
- `max_features`：在 Scikit-Learn 中，此参数限制在每个节点寻找最佳分割时考虑的特征数量，这也有助于模型正则化。

### 分割和叶子的最小样本数

另一种有效的正则化 (regularization)策略是为创建或保留分割所需的数据点数量设置一个最小阈值。这可以防止模型基于少量可能异常的数据点创建分支。

#### 分割的最小样本数

- **`min_samples_split`** (Scikit-Learn)：此参数 (parameter)指定了一个内部节点在被考虑分割之前必须拥有的最小样本数。如果一个节点的样本数少于 `min_samples_split`，它将不再被分割，并成为一个叶节点。增加此值会使模型更保守。

#### 叶子中的最小样本数

- **`min_samples_leaf`** (Scikit-Learn)：此参数强制规定任何终端节点（叶子）中必须包含的最小样本数。只有当分割后左右分支中分别至少保留 `min_samples_leaf` 个训练样本时，才会考虑进行分割。这是平滑模型的有效方法，尤其适用于回归问题。
- **`min_child_weight`** (XGBoost)：这是 XGBoost 的等效参数，定义为子节点中所需的实例权重 (weight)（Hessian）的最小总和。对于许多常见的损失函数 (loss function)，这可以大致解释为每个叶子中所需的最小样本数。较大的值可以防止模型学习仅由少量实例支持的关系，从而减少过拟合 (overfitting)。

### 基于最小增益剪枝

一种更精巧的控制树生长的方法是要求任何新的分割必须对模型的损失函数 (loss function)提供足够的改进。如果一个潜在的分割未能将损失至少减少到一定量，它就会被“剪枝”而不被执行。

- **`gamma`** (XGBoost) 或 **`min_split_gain`**：此参数 (parameter)指定进行分割所需的最小损失减少量。只有当分割导致目标函数获得正增益时才添加分割，其中增益计算如下：


$$
\text{增益} = \text{损失}_{\text{父节点}} - (\text{损失}_{\text{左子节点}} + \text{损失}_{\text{右子节点}})
$$


仅当 $Gain > gamma$ 时才执行分割。设置更高的 `gamma` 值会使算法更保守，并产生更简单的树。`gamma` 为 0 意味着此参数不进行正则化 (regularization)。这是一个强大的后剪枝参数，通过剪除价值不大的分支来帮助控制最终模型的复杂度。

通过仔细调整这些超参数 (hyperparameter)，你可以找到一个平衡点，使你的模型既足够复杂以捕获数据中真实的潜在信号，又足够简单以避免拟合噪声。这些参数的理想设置高度依赖于数据集，而我们将接下来介绍的结构化调优过程对于找到它们来说是必不可少的。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  统计学习理论、决策树和梯度提升的经典教材，包括模型复杂度控制。第二版。
- [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785) — Tianqi Chen and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍XGBoost算法的开创性论文，详细阐述了其优化的树构建和正则化技术，包括`gamma`和`min_child_weight`。
- [\`sklearn.ensemble.GradientBoostingClassifier\`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) — Scikit-learn Developers (2023)
  描述Scikit-learn梯度提升中`max_depth`、`min_samples_split`、`min_samples_leaf`、`max_leaf_nodes`和`max_features`等超参数的官方文档。
