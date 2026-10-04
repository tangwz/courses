---
course: "mastering-gradient-boosting-algorithms"
chapter: "regularization-gradient-boosting"
lesson: "tree-constraints-regularization"
sourceId: 1960
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-3-regularization-gradient-boosting/tree-constraints-regularization"
title: "树结构限制：深度、节点和分裂"
description: "使用树结构限制（最大深度，最小分裂样本数）进行正则化。"
order: 2
plots: []
sourceHash: "879927c3c4ea81130ef6b157aa92f8999de1979dfc5eef96ac5316027b17a4d4"
sourceCorrections: []
---

正如章节引言中指出的，梯度提升模型建立在弱学习器之上，通常是决策树。提升的效力在于迭代地添加这些树，每棵树都纠正了之前集成的错误。然而，如果允许单棵树变得任意复杂，它们很容易记住训练数据，包括其中的噪声。限制这些基础学习器的结构是一种基本的正则化 (regularization)技术。

### 限制树的深度（`max_depth`）

限制决策树复杂度的最直接方法之一是限制其最大深度。树的深度是从根节点到叶节点的最长路径的长度。

- **影响：** 较浅的树（小的`max_depth`）复杂度较低。它们只能捕获相对简单的模式和低阶特征交互（涉及由根路径决定的少量特征的交互）。较深的树（大的`max_depth`）可以模拟更复杂的关联和高阶特征交互。
- **正则化 (regularization)效果：** 通过限制`max_depth`，可以防止树生成高度特定的路径，以适应个别或小部分训练样本。这会促使模型找到在数据较大子集中共享的更普遍的模式。
- **权衡：**
  - *过浅：* 模型可能欠拟合 (underfitting)，未能捕获数据中的重要结构（高偏差）。
  - *过深：* 模型很可能过拟合 (overfitting)，捕获噪声并在未见过的数据上表现不佳（高方差）。
- **典型值：** 值通常在3到10之间，但最佳深度高度依赖于数据集大小、维度以及底层函数的固有复杂度。更深的树在构建时计算成本更高，需要更多数据才能可靠地训练而不过拟合。

在提升的背景下，即使是相对较浅的树（例如，深度4-8）也能形成强大的集成，因为复杂度是在多次迭代中累加形成的。

### 每叶节点最小样本数（`min_samples_leaf`）

树的终端节点（叶节点）中必须包含的最小训练样本数是一个重要的结构限制。一个分裂点只有在左右分支中都至少留下`min_samples_leaf`个训练样本时，才被认为是有效的。

- **影响：** 将此参数 (parameter)设置为大于1的值，可以防止模型生成对应于极少数（可能是异常值）训练样本的叶节点。它能有效地平滑模型的预测函数，尤其是在回归任务中。
- **正则化 (regularization)效果：** 较高的`min_samples_leaf`会促使树生成更普遍的数据分区。它阻止模型基于极小部分实例进行高度特定的预测，从而降低方差和过拟合 (overfitting)。
- **权衡：**
  - *过低（例如，1）：* 树可以为单个样本创建叶节点，从而最大化过拟合的可能性。
  - *过高：* 树可能变得过度受限，无法捕获数据中更精细的模式，导致欠拟合 (underfitting)（高偏差）。
- **考量：** 合适的值通常取决于总样本数。它可以指定为绝对数（例如，10个样本），或者有时是总训练样本的某个比例。请注意，像XGBoost这样的库使用`min_child_weight`，它考虑的是叶节点中Hessian权重 (weight)的总和，而不仅仅是样本数量，这提供了更精细的控制，尤其是在加权数据集或特定目标函数下。

### 每次分裂最小样本数（`min_samples_split`）

此参数 (parameter)设置了内部节点进行进一步分裂所需的最小训练样本数。如果一个节点包含的样本少于`min_samples_split`，即使潜在的分裂会提高纯度，该节点也不会被考虑进行分裂。

- **影响：** 它在树构建过程中比`min_samples_leaf`更早起作用。它会更早地停止树的较小分支的划分过程。
- **正则化 (regularization)效果：** 与`min_samples_leaf`类似，它阻止模型尝试划分非常小的样本组，这些样本组可能只反映了训练数据中的噪声。它有助于构建更具泛化性的树。
- **权衡：** 与`min_samples_leaf`适用类似的权衡。设置过高会导致欠拟合 (underfitting)，而默认值（通常为2）允许在最小数据上进行分裂，可能增加过拟合 (overfitting)风险。
- **与`min_samples_leaf`的关系：** 通常，`min_samples_split >= 2 * min_samples_leaf`以确保任何潜在的分裂都能实际产生有效的叶节点。设置`min_samples_split`有助于更早地修剪分支，与仅依赖`min_samples_leaf`相比，这可能会稍微加快训练速度。

### 最大叶节点数（`max_leaf_nodes`）

除了直接控制深度之外，还可以限制树中终端节点（叶节点）的总数。树的生长方式是最大化杂质减少，直到达到最大叶节点数。

- **影响：** 这提供了一种控制树复杂度的替代机制。叶节点数量有限的树可能会非对称生长，在杂质减少显著的一些分支中变得更深，而在其他分支中保持较浅。
- **正则化 (regularization)效果：** 限制了树可以创建的独立预测区域的总数。
- **与`max_depth`的比较：** 限制叶节点有时比限制深度提供更大的灵活性。深度受限的树通常对称生长（在某些实现或概念中是逐层生长），而叶节点受限的树可能更好地适应数据的结构（最佳优先或逐叶生长）。在许多实现中（如LightGBM的默认逐叶生长），`max_leaf_nodes`通常被认为是比`max_depth`更直接的复杂度控制方式。当设置了`max_leaf_nodes`时，`max_depth`可能会变得冗余或影响力较小。

### 最小杂质减少/最小分裂增益（`min_impurity_decrease`、`min_split_gain`，XGBoost中的`gamma`）

树算法选择能最大化杂质度量减少（例如回归任务中的均方误差，或分类任务中的基尼/信息熵）或最大化特定增益标准（如XGBoost目标函数中使用的）的分裂点。此参数 (parameter)为该减少或增益设置了一个阈值。

- **影响：** 只有当分裂提供的改进（杂质减少或目标增益）大于或等于此阈值时，才会执行分裂。
- **正则化 (regularization)效果：** 这是一种预剪枝的形式。它阻止树进行那些只提供微小增益的分裂，这些分裂更可能是拟合训练数据中的噪声，而不是捕获真实的潜在模式。为此参数设置一个正值会导致更保守的树生长。
- **考量：** 此参数的量级很大程度上取决于损失函数 (loss function)和数据集。例如，XGBoost的`gamma`与在叶节点上进行进一步划分所需的最小损失减少量有关。调整此参数通常需要通过实验来确定。

> 示例展示了像`min_samples_leaf=10`这样的限制如何阻止本会创建一个小叶节点（叶节点4有5个样本，现成为叶节点C的一部分，叶节点C有40个样本）的分裂。

这些结构性限制是需要调优的超参数 (hyperparameter)，通常通过交叉验证进行，同时还有其他提升参数，如学习率和树的数量。找到合适的平衡可以防止单棵树变得过度专业化，从而得到一个泛化能力更强的最终集成模型。

## 参考资料

- [Classification and Regression Trees](https://doi.org/10.1201/9781315139470) — Leo Breiman, Jerome Friedman, Richard A. Olshen, and Charles Stone (1984)
  Publisher: Chapman and Hall/CRC; DOI: [10.1201/9781315139470](https://doi.org/10.1201/9781315139470)
  决策树算法的经典著作，详细介绍了树的划分、剪枝和复杂性控制方法。
- [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  介绍了梯度提升机算法，阐述了其使用弱学习器迭代构建集成模型的过程。
- [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785) — Tianqi Chen and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  描述了XGBoost算法，包括用于树复杂性控制的正则化参数，如`gamma`和`min_child_weight`。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本关于统计学习的综合教材，涵盖了决策树、集成方法和正则化技术。
- [Parameters - LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html) — Microsoft and LightGBM Contributors (2024)
  官方文档，详细介绍了LightGBM的超参数，包括`max_depth`和`num_leaves`，以及它们对树复杂性的影响。
