---
course: "intro-feature-engineering"
chapter: "feature-selection"
lesson: "filter-methods-overview"
sourceId: 1394
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-6-feature-selection/filter-methods-overview"
title: "过滤方法概述"
description: "介绍独立于机器学习算法评估特征的过滤方法。"
order: 2
plots: []
sourceHash: "81ff690ff077584e868951508f20b53de58bb2d755a9c9711906ba52bf2e401f"
sourceCorrections: []
---

在明确了为什么特征选择是构建有效机器学习 (machine learning)模型的重要步骤之后，我们现在考察用于此目的的第一类技术：**过滤方法**。

过滤方法是一类特征选择算法，它们根据特征的内在统计特性以及它们与目标变量的关系来评估特征的价值。过滤方法的显著特点是，这种评估的进行是*独立于*您之后可能选择用于预测的任何特定机器学习算法的。它们作为预处理步骤，在实际模型训练开始*之前*筛选掉特征。

### 过滤方法的工作原理

一般方法包括为每个特征计算一个统计分数。这个分数量化 (quantization)了某些属性，例如：

- **方差**：特征值的离散程度。
- **相关性**：特征与目标变量之间的统计关系。
- **统计显著性**：假设检验的结果（例如，回归的ANOVA F检验或分类的卡方检验），用于评估特征与目标之间的关系。

特征选择中，特征通常根据相关性进行排序。可以根据排名选择前 *k* 个特征，或者舍弃分数低于预设阈值的任何特征。

考虑这个流程：

> 过滤方法的流程：在完整特征集上计算统计指标，特征根据这些指标进行排序和选择/舍弃，所得的精简特征集随后用于模型训练。

### 过滤方法的优点

1. **计算效率**：由于它们不涉及训练机器学习 (machine learning)模型，过滤方法通常非常快速，这使它们适合于高维数据集，在这些数据集中，重复训练模型（如在封装方法中）将不可行。
2. **模型无关性**：特征选择过程不与任何特定模型绑定。由此生成的特征子集可以与多种算法搭配使用。
3. **简洁性**：大多数过滤方法背后的原理基于标准统计量，使得它们相对容易理解和实现。

### 过滤方法的局限

1. **忽略特征依赖性**：过滤方法通常单独评估特征，或评估其与目标的关系，常常忽略特征之间的依赖或共同作用。即使每个特征单独得分不高，一组特征也可能整体上具有很强的预测能力。它们也不直接处理多重共线性（特征*之间*的高度相关性）。
2. **忽略模型偏差**：所选的统计指标可能与您计划使用的学习算法的归纳偏差不完全匹配。由过滤方法认为优化的特征集，对于特定模型可能无法产生最佳表现（例如，基于树的模型与线性模型）。

尽管存在这些局限，过滤方法在许多特征选择流程中作为重要第一步，特别是为了快速降低维度或确立一个基准特征集。在接下来的部分，我们将详细考察具体的过滤技术，例如方差阈值法、单变量统计检验和相关性分析。

## 参考资料

- [Data Mining: Concepts and Techniques](https://www.elsevier.com/books/data-mining/han/978-0-12-381479-1) — Jiawei Han, Micheline Kamber, Jian Pei (2011)
  Publisher: Elsevier
  一本涵盖数据挖掘基本概念的综合性教材，包括数据预处理和过滤方法等特征选择技术。
- [Feature Engineering and Selection: A Practical Approach for Developers and Engineers](https://www.routledge.com/Feature-Engineering-and-Selection-A-Practical-Approach-for-Developers-and/Kuhn-Johnson/p/book/9780367181676) — Max Kuhn and Kjell Johnson (2019)
  Publisher: CRC Press; DOI: [10.1201/9780429188448](https://doi.org/10.1201/9780429188448)
  一本关于特征工程和选择的实用指南，提供了过滤方法的详细解释及其应用。
- [An Introduction to Variable and Feature Selection](https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf) — Isabelle Guyon and André Elisseeff (2003)
  Journal: Journal of Machine Learning Research; Publisher: MIT Press; Volume: 3; Pages: 1157-1182; DOI: [10.1162/153244303322753616](https://doi.org/10.1162/153244303322753616)
  一篇奠基性论文，系统地分类和讨论了不同的特征选择方法，包括过滤、包装和嵌入式方法。
