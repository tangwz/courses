---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "ensemble-learning-and-boosting-foundations"
lesson: "bias-variance-in-ensembles"
sourceId: 7577
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-1-ensemble-learning-and-boosting-foundations/bias-variance-in-ensembles"
title: "集成学习中的偏差-方差权衡"
description: "分析装袋法和提升法等集成方法如何处理偏差-方差权衡，以提升模型泛化能力并减少过度拟合。"
order: 6
plots: ["plots/7577-0.json"]
sourceHash: "8c7df880a52adf6c3ba7bfa1aa78c683875077ae33ebc96d28cc8f621dc8c465"
sourceCorrections: []
---

机器学习 (machine learning)中的一个主要问题是如何处理好偏差和方差之间的权衡。一个高偏差模型会对数据做出较强的假定，无法学到数据中固有的规律（欠拟合 (underfitting)）。一个高方差模型对训练数据过度敏感，捕捉到随机噪声，导致在新数据上表现不佳（过拟合 (overfitting)）。集成方法提供有效的对策来应对这种权衡问题，但装袋法和提升法采用显著不同的方式。

### 装袋法：通过平均减少方差

装袋法（Bagging），是自助聚合（Bootstrap Aggregating）的简称，主要是一种减少方差的技术。这种方法最适合不稳定且方差较大的模型，例如完全生长的决策树。这些模型被认为是“强”但不够稳定的学习器，因为它们倾向于具有低偏差，但对训练数据过度拟合。

此过程包含两个主要步骤：

1. **自助采样：** 从原始训练数据中通过有放回抽样创建多个随机子集。
2. **聚合：** 在每个子集上训练一个独立的基模型。最终预测是所有各个模型的平均值（用于回归）或多数投票（用于分类）的。

通过在数据的不同子集上训练模型，我们得到一个多样化的预测器集合。尽管每个单独的模型可能过度拟合并产生噪声预测，但这些误差通常不相关。当我们平均这些预测时，噪声趋于抵消，从而得到一个更平滑、更稳定的预测边界。最终模型的偏差大致与单个基模型的偏差相同，但方差大幅降低。



![装袋法如何减少模型方差](plots/7577-0.json)



> 单个高方差模型会紧密拟合它们所见的特定训练点。对它们的预测进行平均会得到一个装袋模型，该模型更加平滑并更接近真实的底层函数。

### 提升法：通过迭代修正减少偏差

提升法基于完全不同的原理。它主要是一种减少偏差的技术。该方法从具有高偏差的简单基模型开始，例如浅层决策树（通常只是单次拆分的“树桩”）。这些模型被认为是“弱学习器”，因为它们单独表现仅比随机猜测稍好。

提升法顺序地构建集成模型。每个新模型都经过训练，以修正之前模型组合所产生的误差。例如，在AdaBoost中，被早期模型错误分类的数据点在后续模型的训练中被赋予更高的权重 (weight)。这促使算法关注它难以处理的样本。

通过逐个添加模型，每个模型逐步减少剩余误差，集成模型逐渐成为一个强学习器。最终模型的偏差大幅低于其任何弱组成部分。然而，这种积极追求最小化训练误差的做法伴随风险。如果序列中添加的模型过多，集成模型可能会开始对训练数据过度拟合，从而增加其方差。这就是为什么控制模型数量和学习率的参数 (parameter)对于提升算法中的正则化 (regularization)作用很大的原因。

### 集成策略概览

装袋法和提升法之间的选择通常取决于您需要解决的误差类型。如果您的基模型过于复杂且过度拟合，装袋法是很好的选择。如果您的基模型过于简单且欠拟合 (underfitting)，提升法是更好的方案。

> 装袋法和提升法如何处理偏差-方差权衡的比较。

有了这个基本认识，您已为理解梯度提升机如何扩展提升法的核心思路做好了准备。它没有像AdaBoost那样使用简单的加权方案，而是采用一种更通用且高效的方法，基于梯度来修正误差，让我们能精细地控制模型表现。

## 参考资料

- [Bagging Predictors](https://link.springer.com/article/10.1007/BF00058655) — Leo Breiman (1996)
  Journal: Machine Learning; Publisher: Springer; Volume: 24; Pages: 123-140; DOI: [10.1007/BF00058655](https://doi.org/10.1007/BF00058655)
  介绍用于减少预测方差的 bagging 集成方法的开创性论文。
- [A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](https://doi.org/10.1006/jcss.1997.1504) — Yoav Freund, Robert E. Schapire (1997)
  Journal: Journal of Computer and System Sciences; Publisher: Academic Press; Volume: 55; Pages: 119-139; DOI: [10.1006/jcss.1997.1504](https://doi.org/10.1006/jcss.1997.1504)
  介绍了 AdaBoost 算法，这是一种顺序组合弱学习器的基础性提升方法。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://www.springer.com/gp/book/9780387848570) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本关于统计学习的综合性教材，包含对偏差-方差权衡、bagging、boosting 和梯度提升的详细解释。第二版。
- [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  介绍了广义梯度提升框架，将提升方法扩展到任意可微分的损失函数。
