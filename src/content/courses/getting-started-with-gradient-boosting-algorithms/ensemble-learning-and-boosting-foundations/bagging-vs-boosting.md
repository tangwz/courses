---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "ensemble-learning-and-boosting-foundations"
lesson: "bagging-vs-boosting"
sourceId: 7573
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-1-ensemble-learning-and-boosting-foundations/bagging-vs-boosting"
title: "Bagging 与 Boosting"
description: "比较和对比两种常见的集成方法：Bagging（自助聚合）和 Boosting。了解它们在模型构建上的不同方式。"
order: 2
plots: []
sourceHash: "5c393aecf845ad4896c0bbbe0fdfb0fef3a5329bb0cbda6500ed507e8fe9d1bb"
sourceCorrections: []
---

集成方法结合多个机器学习 (machine learning)模型，以生成一个单一的、表现更好的预测模型。虽然有许多变体，但多数技术属于两种基本策略之一：并行独立地构建模型，或者顺序构建模型，其中每个模型都从前一个模型中学习。Bagging 和 Boosting 是这两种不同方式的典型例子。

### Bagging：独立模型的平均

Bagging，即 **Bootstrap Aggregating**（自助聚合），是一种集成技术，旨在降低预测模型的方差。当与容易过拟合 (overfitting)的基础模型（如完全成长的决策树）一起使用时，它特别有效。该过程通过数据抽样引入随机性，独立训练多个模型，然后结合它们的输出。

该过程可分为两个主要步骤：

1. **自助采样 (Bootstrap Sampling)**：从大小为 $N$ 的原始训练数据集中，创建多个新的数据集，它们的大小也为 $N$。这些新数据集被称为自助样本，通过有放回抽样生成。这意味着原始数据集中的任何给定数据点可能会在样本中出现多次，而其他数据点可能根本不出现。平均而言，一个自助样本大约包含原始数据点的 63%。
2. **聚合 (Aggregation)**：在每个自助样本上独立训练一个基础模型。由于每个模型看到的数据子集略有不同，因此每个模型都学习到略微不同的决策边界。对于回归任务，所有模型的预测结果取平均值以生成最终结果。对于分类任务，最终预测通过模型间的多数投票决定。

通过平均这些不相关模型的输出，个体模型的误差和不稳定性会相互抵消。这使得最终模型的方差更低，泛化性能优于任何单独在原始数据上训练的模型。基于此思想的著名算法是随机森林（Random Forest），它通过在树的每个分裂点也对特征进行抽样，增加了另一层随机性。

> Bagging 过程并行训练多个模型，每个模型都在数据的不同随机样本上。最终预测是它们各自输出的聚合。

### Boosting：顺序从错误中学习

与 Bagging 的并行方法不同，Boosting 顺序构建模型集成。主要思想是从一系列“弱”学习器中构建一个“强”学习器，其中每个新的学习器都旨在纠正其前一个模型所犯的错误。弱学习器是性能仅略优于随机猜测的模型，简单的决策桩（一层决策树）是一个常见选择。

Boosting 过程是迭代的：

1. **初始模型**：在整个数据集上训练一个简单的基础模型（弱学习器）。
2. **错误分析**：将模型的预测与实际结果进行比较。识别出预测不正确的实例。
3. **顺序训练**：训练下一个弱学习器，但它会更多地关注前一个模型错误分类的实例。不同的 Boosting 算法以不同方式实现此目的。例如，AdaBoost 明确增加错误分类数据点的权重 (weight)，促使下一个模型更关注它们。梯度提升（我们将在后续详细讲解）则在前一个模型的残差错误上训练下一个模型。
4. **加权组合**：此过程重复特定次数的迭代。最终预测是所有弱学习器预测结果的加权和。表现更好的模型通常在最终组合中被赋予更高的权重。

因为 Boosting 专注于难以预测的样本，它在降低模型整体偏差方面非常有效。它通过组合许多简单模型，逐步构建一个复杂的决策边界，将一组高偏差的弱学习器转变为一个单一的低偏差强学习器。

> Boosting 过程顺序训练模型。每个模型都旨在纠正前一个模型的错误，并且它们的预测结果以加权和的方式组合。

### 差异总结

Bagging 和 Boosting 都是强大的集成技术，但它们的基本理念和主要目标却大不相同。

| 特点 | Bagging | Boosting |
| --- | --- | --- |
| **模型构建** | 并行且独立。 | 顺序且依赖。 |
| **主要目标** | 降低模型**方差**。 | 降低模型**偏差**。 |
| **基础学习器** | 复杂模型（低偏差，高方差）。 | 简单模型（高偏差，低方差）。 |
| **数据侧重** | 每个模型在随机数据样本上训练。 | 每个模型侧重于前一个模型的错误。 |
| **最终组合** | 简单平均或多数投票。 | 加权和或投票。 |
| **过拟合 (overfitting)** | 通常能防止过拟合。 | 若不仔细调整参数 (parameter)，可能过拟合。 |

了解这两种策略的区别为本课程的后续内容打下了必要基础。Bagging 构建一组多样化的独立决策者，而 Boosting 则构建一个高度专业化的团队，每个成员都在前一个成员的工作基础上进行改进。我们现在将完全把重点转向 Boosting，从它在 AdaBoost 算法中的最早形式化开始。

## 参考资料

- [Bagging Predictors](https://doi.org/10.1007/BF00058655) — Leo Breiman (1996)
  Journal: Machine Learning; Volume: 24; Pages: 123-140; DOI: [10.1007/BF00058655](https://doi.org/10.1007/BF00058655)
  介绍用于减少预测模型方差的装袋（bagging）技术的原始论文。
- [A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](https://doi.org/10.1006/jcss.1997.1504) — Yoav Freund and Robert E. Schapire (1997)
  Journal: Journal of Computer and System Sciences; Publisher: Elsevier; Volume: 55; Pages: 119-139; DOI: [10.1006/jcss.1997.1504](https://doi.org/10.1006/jcss.1997.1504)
  关于AdaBoost算法的奠基性论文，这是一种重要的早期提升（boosting）方法。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  涵盖统计学习理论基础的综合性教科书，包括对装袋、提升和随机森林等集成方法的详细章节。第二版。
- [Random Forests](https://doi.org/10.1023/A:1010933404324) — Leo Breiman (2001)
  Journal: Machine Learning; Volume: 45; Pages: 5-32; DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A%3A1010933404324)
  介绍随机森林算法的原始论文，该算法通过特征随机化扩展了装袋技术。
