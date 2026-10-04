---
course: "mastering-gradient-boosting-algorithms"
chapter: "catboost-gradient-boosting"
lesson: "catboost-oblivious-trees"
sourceId: 2015
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-6-catboost-gradient-boosting/catboost-oblivious-trees"
title: "遗忘树"
description: "解释 CatBoost 中使用的对称树结构。"
order: 5
plots: []
sourceHash: "cab5f558357890e27f3908b1e42c14a7194f66a04fb65b419744a50328941510"
sourceCorrections: []
---

CatBoost 的一个显著特点，也是其区别于 XGBoost 和 LightGBM 等算法之处，在于它使用*遗忘树*（也称为对称树）作为基学习器。大多数决策树算法构建的是非对称树，其中在相同深度可能出现不同的分割，具体取决于所走的路径，而 CatBoost 则强制执行更严格的结构。

### 遗忘树的结构

在遗忘树中，同一深度级别的所有节点都测试完全相同的特征，并使用完全相同的分割条件（阈值）。这意味着树是完美平衡和对称的。从根到叶节点的每条路径长度相同，并且在每一步遇到的分割标准对于经过该层的所有样本都是一致的。

考虑一个简单例子。如果一个遗忘树的深度为 2：

1. **第 0 层（根）：** 所有数据点都基于一个条件进行分割，例如 `Feature_X < threshold_1`。
2. **第 1 层：** 本层的所有节点（根的左子节点和右子节点）都使用*相同的*分割标准，例如 `Feature_Y > threshold_2`。
3. **第 2 层（叶）：** 包含预测值的终端节点。

这形成了一种结构，其中对于给定的深度，测试的特征和阈值的顺序是固定的，无论数据点遵循哪个分支。

> 深度为 2 的遗忘树结构示例。请注意，第 1 层的所有节点都使用相同的分割条件（`特征 Y < T2`）。

### 遗忘树的优点

为什么要施加这样的结构限制？遗忘树在 CatBoost 框架内提供了多项显著优点：

1. **计算效率：** 对称结构非常适合高效的向量 (vector)化实现。由于所有正在处理的数据点在给定层遇到相同的特征分割，因此评估可以大幅优化。与根据每个数据点的路径进行复杂的条件分支不同，操作通常可以在代表整个数据集（或批次）的数组或张量上执行，从而显著提速，尤其是在 GPU 等并行架构上。CatBoost 高效的 GPU 训练实现就有效发挥了这一特性。
2. **更快的预测：** 评估新实例也非常快。任何实例的路径都由固定的比较序列决定。任何样本的叶子索引都可以根据这些比较的结果直接计算，通常表示为一个简单的二进制字符串或整数索引，而无需通过不规则的树结构进行复杂的指针追踪。
3. **隐式正则化 (regularization)：** 每层固定的分割条件作为一种结构正则化形式。它通过为少量数据子集创建高度特定的路径来防止模型构建过于复杂的单个树，从而避免过拟合 (overfitting)噪声。这种更简单的结构本质上限制了每个基学习器的复杂性，有助于更好的泛化能力。CatBoost 通常通过构建更多的树或依赖其精密的特征组合技术来弥补*单个*遗忘树潜在的较低表达能力。

### 与 CatBoost 其他特性的关系

遗忘树与 CatBoost 的其他创新良好结合。复杂特征组合（分类特征之间的交互）的生成可能导致高维特征空间。遗忘树的简单、规则结构提供了一种即使在考虑这些大量生成的特征时也能构建模型的方法。固有的正则化 (regularization)也有助于管理这些组合带来的潜在复杂性。

### 权衡

主要的权衡是，单个遗忘树可能不如非对称树那样强大，或者需要更大的深度才能直接捕捉某些复杂的交互。非对称树可以专门为到达特定节点的数据子集定制分割。然而，CatBoost 的集成方法，结合了有序提升（Ordered Boosting）和自动特征组合等特性，旨在在模型层面克服这一限制，同时保留对称结构的性能和正则化 (regularization)优势。

本质上，遗忘树是 CatBoost 设计理念的一个核心组成部分，它通过一种独特且结构化的方法来构建决策树集成模型，优先考虑效率、正则化和对分类数据的优化处理。这种结构选择是与其他流行梯度提升实现相比的一个根本区别。

## 参考资料

- [CatBoost: Unbiased Boosting with Categorical Features](https://proceedings.neurips.cc/paper_files/paper/2018/file/0c91741528667a998520fe813337a671-Paper.pdf) — Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin (2018)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 31; Pages: 6639-6649
  这篇基础论文介绍了CatBoost算法，详细阐述了其独特组成部分（包括Oblivious Trees），并解释了它们的结构和优势。
- [CatBoost Documentation: CatBoost Tree Structure](https://catboost.ai/en/docs/concepts/algorithm-main-stages_catboost-tree-structure.html) — Yandex (2023)
  Publisher: Yandex
  官方文档对CatBoost的树结构提供了权威解释，对Oblivious Trees的设计原理和实际影响提供了见解。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  这本广受认可的书籍为决策树、集成方法和梯度提升提供了全面基础，帮助读者理解CatBoost树设计选择的一般背景。
