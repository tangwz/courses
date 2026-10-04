---
course: "basics-model-evaluation-metrics"
chapter: "metrics-for-regression"
lesson: "r-squared-metric"
sourceId: 4008
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-3-metrics-for-regression/r-squared-metric"
title: "决定系数 (R平方)"
description: "定义R平方（R²）作为衡量预测值与实际值拟合程度的指标。"
order: 7
plots: ["plots/4008-0.json", "plots/4008-1.json"]
sourceHash: "d58993100c87a7dd2b40de2a279df842f0d7b24e025b6a20ded0bc87d5fd1583"
sourceCorrections: []
---

决定系数，通常称为R平方，有助于评估回归模型的性能。虽然像平均绝对误差（MAE）、均方误差（MSE）和均方根误差（RMSE）等其他指标量化 (quantization)了预测误差的平均大小，但R平方提供了一个不同的视角，它显示了模型在多大程度上解释了数据固有的变异性。这种区别具有实际意义，因为仅仅凭借绝对误差大小进行评估可能会产生误导；例如，RMSE为10对于预测数百万范围的房价可能非常出色，但对于预测0到40度之间的气温可能非常差。

决定系数，通常称为R平方（$R^2$），在此发挥作用。R平方并非仅仅关注误差大小，而是衡量了目标变量（您试图预测的值）的变异性有多少可以由您的模型解释。它本质上是将您的模型性能与一个非常简单的基准模型进行比较。

### 基准：预测平均值

想象一下预测连续值的最简单的“模型”：总是预测数据集中所有实际目标值的平均值。这个基准模型不使用任何输入特征；它只是每次都做出相同的预测。例如，如果您正在预测房价，并且数据集中平均价格是300,000美元，那么无论房屋的大小、位置或状况如何，这个基准模型都会预测每套房屋的价格为300,000美元。

R平方告诉您，您的实际回归模型与这个简单的平均值预测基准相比，性能好多少。

### 计算R平方

要了解R平方是如何计算的，我们需要两个组成部分：

1. **总平方和 (SST)：** 这衡量了实际目标值（$y_i$）的总方差。它的计算方法是，将每个实际值（$y_i$）与实际值的总体平均值（$\bar{y}$）之间的平方差相加。它代表了如果您仅使用平均值作为预测，数据中固有的变异性。

   
   $$
   SST = \sum_{i=1}^{n} (y_i - \bar{y})^2
   $$
   

   这里，$n$ 是数据点的数量，$y_i$ 是第 $i$ 个数据点的实际值，而 $\bar{y}$ 是所有实际值的平均值。
2. **残差平方和 (SSR) 或 误差平方和 (SSE)：** 这衡量了您的模型*无法*解释的方差。它的计算方法是，将每个实际值（$y_i$）与您的模型生成的对应预测值（$\hat{y}_i$）之间的平方差相加。这与计算MSE时使用的误差平方和相同，只是没有除以 $n$。

   
   $$
   SSR = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   $$
   

   这里，$\hat{y}_i$ 是您的模型为第 $i$ 个数据点预测的值。

现在，R平方公式结合了这两个：


$$
R^2 = 1 - \frac{SSR}{SST}
$$


思考一下这个公式：

- $\frac{SSR}{SST}$ 代表了您的模型*未能*解释的总方差中的*部分*（残差误差相对于总方差的比例）。
- 通过用1减去这个部分，您就得到了您的模型*能够*解释的总方差中的*部分*。

### R平方的解读

R平方值通常在0到1之间，尽管有时也可能为负（更多内容请参阅“R平方的局限性”部分）。它通常以百分比表示。

"\* **$R^2 = 1$（或100%）：** 这表示您的模型解释了目标变量的所有变异性。模型的预测值（$\hat{y}_i$）完美匹配实际值（$y_i$），因此$SSR = 0$。这在实际情况中通常好得不真实，可能表示过拟合 (overfitting)（即模型对训练数据学习得太好，包括其噪声）。"

- **$R^2 = 0$（或0%）：** 这表示您的模型没有解释任何变异性。其性能不比简单预测平均值（$\bar{y}$）的基准模型好。在这种情况下，$SSR = SST$。
- **$0 < R^2 < 1$：** 这是最常见的范围。R平方为0.75表示您的模型基于输入特征解释了目标变量75%的方差。剩余25%的方差未能被模型捕捉。

R平方值是否“好”，很大程度上取决于问题的背景。在某些领域，如物理实验，您可能期望非常高的R平方值（>0.95）。而在其他领域，如社会科学或预测股票价格，即使解释一小部分方差（例如R平方为0.1或0.2）也可能被认为是重要的，因为底层过程非常复杂或有噪声。

### R平方的可视化

考虑两种简单的回归情况：



![高R平方示例（点靠近线）](plots/4008-0.json)



> 在此图中，数据点紧密聚集在回归线（红色）周围。模型的预测值接近实际值，从而导致SSR相对于SST较小，因此R平方值较高（例如，R² ≈ 0.99）。



![低R平方示例（点分散）](plots/4008-1.json)



> 在此图中，数据点在回归线周围更加分散。虽然这条线显示出一般趋势，但模型的预测平均误差更大。SSR占SST的比例更大，从而导致R平方值较低（例如，R² ≈ 0.3）。

R平方补充了MAE、MSE和RMSE等指标。误差指标告诉您预测误差在其目标变量原始单位中的典型*大小*，而R平方则为您提供了一个无量纲度量（比率或百分比），说明您的模型捕获了数据*方差*的多少。它有助于回答以下问题：“与仅仅使用平均值相比，我的模型与数据的拟合程度如何？”

## 参考资料

- [Applied Linear Statistical Models](https://www.mheducation.com/highered/product/applied-linear-statistical-models-kutner-nachtsheim/M9780073108742.html) — Michael H. Kutner, Christopher J. Nachtsheim, John Neter, William Li (2005)
  Publisher: McGraw-Hill Irwin
  一本全面的教科书，为线性回归提供了深刻的统计基础，包括对R方、其组成部分及其在模型评估中作用的详细论述。
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani (2021)
  Publisher: Springer
  这本广泛使用的教科书提供了统计学习方法的易懂介绍。它涵盖了R方作为评估机器学习背景下回归模型性能的基本指标。
- [sklearn.metrics.r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) — scikit-learn developers (2024)
  scikit-learn（一个流行的Python机器学习库）中R方实现的官方文档。它解释了该指标的计算和实际应用。
