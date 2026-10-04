---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "advanced-gradient-boosting-lightgbm-catboost"
lesson: "xgboost-vs-lightgbm-vs-catboost"
sourceId: 7602
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-5-advanced-gradient-boosting-lightgbm-catboost/xgboost-vs-lightgbm-vs-catboost"
title: "性能比较：XGBoost、LightGBM 与 CatBoost"
description: "对三大梯度提升库的比较分析，阐述了它们的优缺点以及常见的性能特点。"
order: 5
plots: ["plots/7602-0.json"]
sourceHash: "7040db8d5e309e0fc41f66b6273d084b6ba3197610104308fe58ebfd90640777"
sourceCorrections: []
---

XGBoost、LightGBM和CatBoost是强大的梯度提升算法，各自提供独特功能。为项目选择合适的算法可能具有挑战性，因为每个库在不同方面都有各自的优势。一项比较分析有助于根据具体需求选择最合适的工具，侧重于训练速度、数据处理和整体表现。

### 训练速度与资源占用

训练速度常常是一个决定性因素，尤其是在处理大型数据集或需要快速迭代模型时。

- **LightGBM：** 这个库通常是最快的。它的速度优势来自两个主要的优化：基于梯度的单边采样（GOSS）和互斥特征捆绑（EFB）。更重要的是，LightGBM是*叶子生长*而非*层级生长*的。这意味着它会在损失潜在减少最大的地方扩展树，通常能更快收敛。然而，如果未使用`num_leaves`和`max_depth`等参数 (parameter)进行适当调控，这种方法有时可能导致在较小数据集上过拟合 (overfitting)。
- **XGBoost：** XGBoost采用层级（或深度）生长策略，逐层构建树。这比叶子生长更具系统性，也不易过拟合，但计算速度可能较慢。XGBoost的性能已经过高度优化，是一个稳固的基准选择，但在非常大、稀疏的数据集上，LightGBM的速度可能超过它。
- **CatBoost：** CatBoost构建对称（或无偏）树。在对称树中，所有相同深度的节点都使用相同的特征和分裂条件。这种结构使得预测速度极快，并能减少过拟合，但训练过程本身有时可能比LightGBM慢，尤其是在处理其复杂的类别特征时会产生额外开销。



![训练时间比较](plots/7602-0.json)



> 在较小的数据集上，每个库优化的额外开销可能导致性能相似。随着数据集大小的增加，LightGBM的优势变得明显，而CatBoost的内部预处理有时会增加其训练时间。

### 类别特征处理

类别数据的处理方式是这些库之间一个重要的区别。

- **CatBoost：** 这是CatBoost的独特之处。它实现了一种高级版的目标编码，称为*有序提升*，按时间顺序处理数据点以防止目标泄漏。它能自动处理类别特征，无需手动预处理，例如独热编码。您只需传递类别特征的列索引，CatBoost会处理其余部分。这种自动化大大简化了建模流程，并通常能提高准确性。
- **LightGBM：** LightGBM也能直接处理类别特征，但它使用一种不同且更简单的方法。它根据训练目标对类别进行分组，这种方法效率高，但可能不如CatBoost的方法复杂。它要求您将类别列转换为特定的数据类型（例如pandas `category`），并且在高基数特征上的表现不如CatBoost。
- **XGBoost：** XGBoost不内置支持类别特征。在训练之前，您必须手动预处理它们，通常对名义特征使用独热编码，对有序特征使用序数编码。虽然这让您拥有完全的掌控，但它增加了一个预处理步骤，并且如果您的特征包含许多类别，可能导致非常宽、稀疏的数据集。

### 准确性和模型表现

尽管这三个库都能取得顶尖的结果，但它们的开箱即用表现和调优要求有所不同。

- **XGBoost：** 它是一个性能持续出色的库，并且经常在机器学习 (machine learning)竞赛中获胜。其正则化 (regularization)选项和实现使其成为获得高准确度的可靠选择，尽管它通常需要仔细的超参数 (parameter) (hyperparameter)调优。
- **LightGBM：** 同样能够达到顶尖表现，并常与XGBoost不相上下。其叶子生长方式能发现更复杂的模式，但如前所述，需要仔细调整与复杂度相关的超参数（`num_leaves`、`min_child_samples`）以避免过拟合 (overfitting)。
- **CatBoost：** 通常在很少的超参数调优下就能提供出色的结果。其默认设置选择得当，并且其有序提升机制能有效防止过拟合，尤其是在包含有影响力的类别变量的数据集中。对于需要快速建立一个稳固基线的团队来说，CatBoost是一个很好的起点。

### 差异总结

下表简要总结了每个库的主要特点。

| 特点 | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- |
| **训练速度** | 快（基准） | 最快 | 快，但在较小数据集上可能较慢 |
| **类别数据** | 手动预处理 | 基于整数（内置） | 自动（有序提升） |
| **树生长方式** | 层级生长 | 叶子生长 | 对称（无偏） |
| **调优难度** | 中到高 | 中到高 | 低到中 |
| **主要优势** | 稳定性高，表现出色 | 大型数据集上的速度 | 出色的类别特征处理 |

### 如何选择：实用指南

选择合适的库取决于您项目的限制和数据特点。下图提供了一个简化的决策路径。

> 一个简化的决策流程，用于选择梯度提升库。

在实际应用中，最好的方法通常是凭经验的。如果时间和资源允许，可以尝试用这三个库训练基线模型。它们在您特定数据集上的性能差异将提供最明确的依据，说明哪种工具最适合您的问题。

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785) — Tianqi Chen and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16); Publisher: Association for Computing Machinery (ACM); Pages: 785–794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍XGBoost的开创性论文，详细阐述了其并行和分布式训练、缓存感知访问以及树提升算法。
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) — Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NeurIPS 2017); Publisher: Curran Associates, Inc.; Pages: 3146--3154; DOI: [10.5555/3295222.3295232](https://doi.org/10.5555/3295222.3295232)
  介绍了LightGBM，阐述了其梯度单边采样（GOSS）和互斥特征捆绑（EFB）技术，以实现更快的训练速度和叶子生长策略。
- [CatBoost: Gradient Boosting with Decision Trees](https://www.jmlr.org/papers/v21/20-053.html) — Anna Veronika Dorogush, Andrey Gulin, Gleb Gusev, Neliya Kazimov, and Liudmila Prokhorenkova (2020)
  Journal: Journal of Machine Learning Research; Volume: 21; Pages: 1-17
  描述了CatBoost算法，重点介绍了其用于处理分类特征的有序提升机制和对称（无偏）树构建。
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/) — XGBoost Contributors (2024)
  官方文档，提供了XGBoost的实用指南、API详情和教程。
