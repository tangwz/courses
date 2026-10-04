---
course: "intro-feature-engineering"
chapter: "feature-creation"
lesson: "motivation-feature-creation"
sourceId: 1363
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-5-feature-creation/motivation-feature-creation"
title: "创建新特征的动机"
description: "了解如何创建新的、信息量大的特征来捕捉数据中的复杂模式。"
order: 1
plots: []
sourceHash: "be27bd5f0a70899cdb45a3c4dbe79296769af10abe68a58cadc71556ea190c79"
sourceCorrections: []
---

您已经学习了如何清洗、填充、编码和缩放原始数据集中的特征。这些是为机器学习 (machine learning)算法准备数据的必要步骤。然而，有时原始特征，即使经过仔细准备，也未能完全体现模型实现最佳性能所需的底层模式或关系。这正是特征创建发挥作用的地方。

将您最初拥有的特征视为基本构建块。特征创建是通过巧妙地组合或转换这些块来构建新的、信息量更大的特征的过程。为什么要这样做呢？因为目标是让模型更容易完成学习任务。

### 明确模式

机器学习 (machine learning)模型从它们获得的数据中学习模式。有些模型，例如线性回归或逻辑回归，天生擅长发现线性关系。然而，如果特征与目标变量之间的真实关系更复杂或非线性，它们可能会遇到困难。

考虑预测房价。一个简单的线性模型可能会将 `square_footage`（面积）和 `number_of_bedrooms`（卧室数量）作为独立的输入。但也许每平方英尺的价值会根据房屋大小而显著变化。大房子每平方英尺的价格可能低于设计精巧的小房子。原始特征 `square_footage` 和 `number_of_bedrooms` 未能直接表示这种交互。通过创建一个新特征，例如 `price_per_square_foot`（如果训练期间价格已知，尽管它通常是目标）或一个交互项，例如 `square_footage * number_of_bedrooms`，我们使这种潜在关系明确，有助于模型更好地学习。

同样，如果一个关系遵循曲线，例如 $y \approx x^2$，只使用 $x$ 作为特征的线性模型将提供不佳的拟合。创建一个新特征，$x_{平方} = x^2$，使得线性模型能够有效地拟合二次关系。

> 该流程图展示了与仅仅依赖原始特征相比，构建特征如何能为机器学习模型提供更直接的输入。

### 融入领域知识

通常，您掌握着关于问题领域的知识，而这些知识并未直接编码在原始数据变量中。特征创建是将这种理解注入建模过程的主要方式。

例如，如果您有 `transaction_timestamp`（交易时间戳）数据，原始时间戳可能不是许多模型最有用的格式。领域知识表明，像 `day_of_week`（星期几）、`month`（月份）或 `hour_of_day`（一天中的小时）等因素可能是顾客行为的重要预测因素。提取这些组成部分会创建新的特征，明确表示可能与预测任务相关的时间模式。同样，将 `date_of_birth`（出生日期）特征转换为 `age`（年龄）是一个常见的、由领域知识驱动的特征创建步骤。

### 简化模型复杂性

有时，创建合适的特征使得您能够有效地使用更简单、更易于解释的模型。与其强迫一个复杂模型（例如深度神经网络 (neural network)或非常深的决策树）从原始特征中学习复杂的交互，不如预先构建捕捉这些交互的特征。这可能使得更简单的模型，如正则化 (regularization)线性回归，能够实现相似甚至更好的性能，而且通常还具有训练更快、解释更容易的额外好处。

在本章中，我们将介绍几种系统地创建新特征的方法：

- **交互项：** 特征的乘法或加法组合。
- **多项式特征：** 创建现有数值特征的幂。
- **日期/时间提取：** 从时间数据中提取有意义的组成部分。
- **分箱：** 将数值数据分组到类别箱中。

通过掌握这些技术，您通常可以显著提升模型的预测准确性及其泛化到新的、未见数据的能力。让我们首先看一下如何生成交互特征。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2013)
  Publisher: Springer; DOI: [10.1007/978-1-4614-7138-7](https://doi.org/10.1007/978-1-4614-7138-7)
  一本基础统计学习教材，涵盖数据准备和特征工程在模型开发中的关键作用。
- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng and Amanda Casari (2018)
  Publisher: O'Reilly Media
  这本书提供了一份实用而全面的特征工程指南，详细阐述了其重要性及各种技术。
- [CS229 Lecture Notes](https://www.kaggle.com/code/pranavthebest/cs-229-machine-learning-notes-autumn-2018) — Andrew Ng (2018)
  Journal: Stanford University; Publisher: Stanford University
  这些被广泛引用的斯坦福机器学习课程讲义讨论了适当的特征表示对于有效模型学习的必要性。
