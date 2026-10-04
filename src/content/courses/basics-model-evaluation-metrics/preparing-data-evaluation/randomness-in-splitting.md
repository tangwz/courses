---
course: "basics-model-evaluation-metrics"
chapter: "preparing-data-evaluation"
lesson: "randomness-in-splitting"
sourceId: 4031
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-4-preparing-data-evaluation/randomness-in-splitting"
title: "数据分割中的随机性"
description: "解释为何在数据划分前进行混洗通常是必要的。"
order: 6
plots: []
sourceHash: "86da2c2433dfd05e25154df9737171be4a23700de4051101d7128ad2bc576e2b"
sourceCorrections: []
---

我们已经明确，将数据划分为训练集和测试集对于真实了解模型在未见过的新数据上的表现非常必要。训练集用于模型学习，而测试集用于评估。

但是，我们究竟该如何进行这种划分呢？我们是否应该简单地取数据行的前80%用于训练，其余20%用于测试呢？通常情况下，这并不是个好主意。

思考一下原因。数据通常具有一些固有的顺序。它可能按收集日期排序，或者按类别分组。如果你有一个按日期排序的客户注册数据集，将前80%用于训练，后20%用于测试，这意味着你的模型只从老客户那里学习，并且只在新客户上进行测试。任何随时间变化的趋势或变化都会使这种评估变得不可靠。同样，如果在分类问题中，不同类别的数据被分到一起，简单的顺序划分可能导致某些类别只出现在训练集中，而另一些只出现在测试集中。

### 混洗的重要性

为避免这些问题，并确保训练集和测试集都能够代表整体数据分布，我们需要引入**随机性**。标准做法是在划分数据集*之前*对其进行**随机混洗**。

可以把这想象成发牌前洗牌。混洗会打乱原始顺序，确保在划分数据时，训练和测试部分都可能包含整体数据集中存在的类似混合的样本、模式和潜在变体。

我们来想象一下。设想一个用于形状分类的小数据集，最初按形状类型排序：

> 如果我们按顺序划分这种有序数据（例如，80/20），训练集将获得所有圆形和一个正方形，而测试集将只获得正方形。模型将无法有效学习关于正方形的知识，并且测试结果会产生误导。

现在，我们来看看如果先进行混洗会发生什么：

> 混洗后，数据顺序被打乱。现在，当我们划分时，训练集和测试集都更有可能包含圆形和正方形的代表性样本。测试集现在能更好地评估模型的泛化能力。

大多数执行数据划分的机器学习 (machine learning)库和工具（例如 Python 中的 `scikit-learn`）都默认启用混洗功能，以确保数据划分的随机性，避免模型训练和评估时的偏差。

### 可复现性和随机状态

尽管我们希望混洗是随机的，但实验往往需要具有**可复现性**。如果你今天随机划分数据，明天再运行相同的代码，你可能会得到略有不同的随机划分，导致评估结果略有差异。这会使调试或比较不同方法变得困难。

为处理这个问题，划分函数通常允许你设置一个 `random_state`（有时也称为 `seed`，即种子）。这本质上是随机混洗算法的一个起始数字。如果你每次运行划分时都使用相同的 `random_state` 值，你将获得完全相同的混洗顺序和完全相同的训练/测试划分。这确保了你的结果一致，并且可以被其他人（或你自己将来）复现。

总之，在划分数据前进行混洗是一个简单但重要的步骤。它有助于确保你的训练集和测试集是整体数据的无偏样本，从而带来对模型处理新数据、未见过样本能力的更可靠评估。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125964/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 850
  解释了使用 Python 库进行数据拆分和混洗的实用技术，包括使用 random_state 实现可重现性。
- [\`sklearn.model_selection.train_test_split\`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) — scikit-learn developers (2024)
  `train_test_split` 函数的官方文档，详细说明了 `shuffle` 和 `random_state` 等参数。
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani (2021)
  Publisher: Springer
  为将数据拆分为训练集和测试集进行模型评估的必要性提供了统计学基础。
