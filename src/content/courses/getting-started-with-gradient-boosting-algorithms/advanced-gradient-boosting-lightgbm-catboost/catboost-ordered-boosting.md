---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "advanced-gradient-boosting-lightgbm-catboost"
lesson: "catboost-ordered-boosting"
sourceId: 7601
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-5-advanced-gradient-boosting-lightgbm-catboost/catboost-ordered-boosting"
title: "CatBoost的有序提升和对称树"
description: "分析CatBoost用于防止目标泄漏的有序提升方法，以及它为加快预测速度而使用的对称（或称无意识）树。"
order: 4
plots: []
sourceHash: "403126be5e7e6175880946a4800a5b3b9c40b3bbdf0799f5c2c3f8b1548590c4"
sourceCorrections: []
---

CatBoost 包含两种主要的架构设计，以提高其预测准确性和效率：有序提升和对称树的使用。这些特点共同作用，创建一个既强大又不易过拟合 (overfitting)的模型，通常只需少量调优。

### 有序提升：防止目标泄漏的方法

在标准的梯度提升实现中，算法迭代地将新树拟合到先前模型预测的残差或梯度上。当计算一个数据点的残差时，模型可以获取该数据点在初始数据集中的真实目标值。这会引入一个不易察觉但要紧的问题，称为**目标泄漏**。模型无意中从它试图预测的值本身进行学习，这可能导致模型在训练数据上表现出色，但对新的、未见过的数据泛化能力不足。

CatBoost 通过一种称为**有序提升**的技术来缓解这个问题。其主要原理是确保对于任何给定数据点，用于计算其梯度的模型并未使用该数据点的标签进行训练。

为了高效实现这一点，CatBoost 使用训练数据的随机排列。在构建新树时，过程被修改：为了计算排列中样本 $i$ 的残差，它只使用基于前 $i-1$ 个样本训练的模型。这种顺序确保了模型对任何样本的预测都不会受到该样本自身目标值的“偏向”影响。

> 对于随机排列中的每个样本，CatBoost 使用仅基于前序样本训练的模型来计算当前残差。这种有序方法避免了模型在训练过程中从目标变量中泄漏信息。

尽管这个过程看起来复杂，CatBoost 使用多重排列和高效数据结构，使其既快速又有效，这为其在构建泛化能力强的模型方面提供了显著优点。

### 对称树：以简洁实现速度和稳定性

CatBoost 的第二个主要创新是使用**对称树**，也称为*无意识决策树*。大多数梯度提升库，如XGBoost和LightGBM，构建的树可能是不平衡或非对称的。在这些树中，在某个深度对一个特征进行划分，可能导致左右分支出现不同的划分特征和条件。

CatBoost 采取了不同的方法。对称树对给定深度级别的所有节点应用*完全相同的划分准则*。这意味着树的每一层都根据相同的单一特征和条件来划分数据，形成一个完全平衡且结构简洁的树。

> 标准非对称树与CatBoost对称树的比较。在对称树中，同一层的所有节点（例如，第二层，以紫色表示）使用相同的划分条件（`特征 Y < 20?`）。

这种设计选择有两个主要优点：

1. **预测效率**：对称树的规则、可预测结构对于现代CPU和GPU来说效率极高。无需遍历一系列复杂的`if-then-else`分支，可以使用高度优化、矢量化 (quantization)的操作来计算预测。这通常使得CatBoost在推理 (inference)（预测阶段）时速度极快。
2. **内置正则化 (regularization)**：严格的结构约束避免了树变得过于复杂并针对单个训练样本进行定制。通过迫使模型在每一层找到适用于大量数据段的简单而有力的规则，它自然地抵制过拟合 (overfitting)。这是一种非常有效的正则化形式，内嵌于模型的架构中。

总而言之，有序提升和对称树使CatBoost成为一个高效的算法。有序提升有助于确保模型学习通用模式而不是记住训练数据，而对称树为集成模型中的每个弱学习器提供了快速且可靠的支持。

## 参考资料

- [CatBoost: Unbiased Gradient Boosting with Categorical Features](https://doi.org/10.55986/neurips.2018.0cce69792070659637c38bc8f7c6530a) — Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin (2018)
  Journal: Advances in Neural Information Processing Systems 31 (NeurIPS); DOI: [10.55986/neurips.2018.0cce69792070659637c38bc8f7c6530a](https://doi.org/10.55986/neurips.2018.0cce69792070659637c38bc8f7c6530a)
  这篇基础论文介绍了 CatBoost 算法，详细阐述了其核心创新，包括通过有序提升来对抗目标泄漏，以及使用对称（遗忘）决策树来实现高效预测和正则化。
- [CatBoost Documentation](https://catboost.ai/docs/) — Yandex (2023)
  官方文档提供了全面的指南、实用示例以及对 CatBoost 功能（包括有序提升和对称树）的解释。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer
  一本经典的机器学习算法教科书，提供了决策树、提升和集成方法的基础理解，为 CatBoost 在正则化和模型复杂性方面的架构选择提供了背景。
