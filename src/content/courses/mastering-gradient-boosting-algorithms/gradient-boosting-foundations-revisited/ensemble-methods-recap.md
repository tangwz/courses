---
course: "mastering-gradient-boosting-algorithms"
chapter: "gradient-boosting-foundations-revisited"
lesson: "ensemble-methods-recap"
sourceId: 1912
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-1-gradient-boosting-foundations-revisited/ensemble-methods-recap"
title: "集成方法：回顾"
description: "对 Bagging、Boosting 和 Stacking 等集成技术进行简要回顾。"
order: 1
plots: []
sourceHash: "65dff905dbc6b4105f1b68497a1ee03aa704ad58a132e4a34f736d77b53f684e"
sourceCorrections: []
---

正如章前言指出，理解高级梯度提升算法需要对它演变而来的基本技术有扎实的掌握。集成方法是这些技术的核心，它是一类机器学习 (machine learning)技术，它结合多个独立模型的预测结果，以生成通常比任何单个模型都更准确、更稳定的最终预测。你可能熟悉其基本原理，但有针对性的回顾将有助于我们构建关于提升算法的讨论。

集成方法的核心思想直接明了：“群众的智慧”。通过聚合多个不同模型的“意见”（预测结果），我们通常可以减轻单个模型的弱点。集成方法主要目标是通过降低预测的*方差*（对训练数据微小变化的敏感性）或*偏差*（模型中简化假设导致的潜在误差），或有时两者都降低，从而提高预测性能。

三种主要策略在集成方法中占据主导地位：Bagging、Boosting 和 Stacking。

### Bagging (Bootstrap Aggregating)

Bagging 通过自助采样（有放回抽样）从原始训练数据中创建多个子集。每个子集上训练一个独立的基模型（通常是相同类型，如决策树）。由于每个模型处理的数据版本略有不同，它们会学习到略微不同的模式，从而产生多样性。最终预测结果通过对所有单个模型的预测进行平均（回归任务）或进行多数投票（分类任务）获得。

Bagging最知名的例子是**随机森林**算法，它通过在每棵决策树的每个分裂点随机选择特征子集来进一步增强多样性。Bagging的主要优势在于它能够**降低方差**，使整体模型更不容易过拟合 (overfitting)，尤其是在使用深层决策树等复杂基学习器时。模型并行训练，这使得它在多核系统上具有计算效率。

### Boosting

Boosting 采用一种根本不同的方法。它不是独立并行地训练模型，而是**顺序地**构建模型。序列中的每个新模型都侧重于纠正到目前为止由集成模型所犯的错误。

考虑一个直观的视角：

1. 在数据上训练一个简单的基模型。
2. 识别该模型表现不佳的实例（即，有较大误差或残差的实例）。
3. 训练一个*新*的基模型，使其更关注这些难处理的实例，从而有效地针对前一模型未解决的“难题”部分。
4. 将新模型与之前的模型结合（通常通过加权求和）。
5. 重复步骤2-4，直到达到指定迭代次数或性能不再提高。

早期如AdaBoost的提升算法通过显式增加后续模型中错误分类实例的权重 (weight)来实现这一点。本课程的重点——**梯度提升**，通过将后续模型拟合到*残差误差*（用于回归）或更一般地拟合到损失函数 (loss function)相对于当前集成模型预测的*负梯度*来完善这一思想。这将问题构架为函数空间中的优化，通过迭代地添加指向整体损失函数最速下降方向的函数（模型）。

最终预测是所有顺序训练的基模型的加权和。Boosting 通常擅长**降低偏差**，并且经常产生高准确度的模型。然而，由于模型是顺序构建并依赖于前一个模型，训练本质上是串行的，并且如果正则化 (regularization)不当，提升模型更容易过拟合 (overfitting)（我们将在第3章详细讨论此主题）。

提升模型的一般形式通常表示为加性模型：


$$
F_M(x) = F_0(x) + \sum_{m=1}^{M} \eta \cdot h_m(x)
$$


其中 $F_0(x)$ 是初始猜测（通常是目标变量的平均值），$h_m(x)$ 是在步骤 $m$ 添加的基学习器（例如，决策树），$M$ 是提升的总轮数，以及 $\eta$ 是学习率（或收缩因子），它控制每个新树的贡献。

> 高层次对比，说明了 Bagging 中在自助样本上进行并行训练，与 Boosting 所特有的顺序、纠错过程的对比。

### Stacking (Stacked Generalization)

Stacking 是一种更复杂的集成技术。它涉及在相同数据上训练几种不同类型的基模型（例如，随机森林、支持向量 (vector)机、k-近邻模型）。然后，使用这些基模型的预测作为输入特征来训练一个*元模型*（也称为混合器或第一层模型）。元模型的目标是学习如何以最佳方式组合来自不同基学习器的预测。Stacking 有时可以比单独使用 Bagging 或 Boosting 获得更好的性能，但它需要仔细设置（尤其是在交叉验证方面，以便为元模型的训练集生成基模型预测），并且计算成本可能很高。

### 侧重于提升算法

尽管 Bagging 和 Stacking 是强大的技术，但本课程集中于**梯度提升**。其顺序、自适应的特性，结合在现代实现（如 XGBoost、LightGBM 和 CatBoost）中发现的效率和正则化 (regularization)创新，使其成为处理结构化（表格）数据问题最有效和最广泛使用的技术之一。理解并行（Bagging）与顺序（Boosting）集成构建之间的这种基本对比，对我们后续的学习非常重要。我们现在将更仔细地查看这些框架中常用的基学习器：决策树。

## 参考资料

- [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324) — Leo Breiman (2001)
  Journal: Machine Learning; Publisher: Springer; Volume: 45; Pages: 5-32; DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A%3A1010933404324)
  介绍Random Forest算法的原始文章，这是bagging的一个知名例子。
- [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: The Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  介绍Gradient Boosting Machine的原始论文，阐述了其框架和应用。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  涵盖统计学习方法的教科书，包含关于集成技术（如bagging、boosting和stacking）的详细章节。
