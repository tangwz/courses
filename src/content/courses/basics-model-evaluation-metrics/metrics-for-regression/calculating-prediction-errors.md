---
course: "basics-model-evaluation-metrics"
chapter: "metrics-for-regression"
lesson: "calculating-prediction-errors"
sourceId: 3991
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-3-metrics-for-regression/calculating-prediction-errors"
title: "计算预测误差"
description: "了解如何计算预测值与实际值之间的差异。"
order: 2
plots: ["plots/3991-0.json"]
sourceHash: "cbd32433f9eae45069085d28ff576fa1c272d7bab41f9fa6fb63fc72971645b2"
sourceCorrections: []
---

让我们首先考虑回归模型进行预测时会发生什么。与预测类别（如“垃圾邮件”或“非垃圾邮件”）的分类模型不同，回归模型预测连续的数值。设想您构建了一个模型来预测明天的气温，或者根据特征预测二手车的价格。模型会给出一个具体的数值作为其预测。

我们需要回答的基本问题是：这些数值预测的准确性如何？为此，我们首先需要量化 (quantization)每个单独预测的*误差*。

### 误差（或残差）的定义

对于您用来测试模型的每个数据点，您都有两个值：

1. **实际值：** 这是真实观测到的值。在我们的示例中，它是实际发生的气温或汽车的实际售价。我们通常将第 $i$ 个数据点的实际值表示为 $y_i$。
2. **预测值：** 这是您的回归模型为同一数据点生成的数值。我们通常将第 $i$ 个数据点的预测值表示为 $\hat{y}_i$（读作“y-hat”）。

**预测误差**，通常称为**残差**，简单来说就是单个观测的实际值与预测值之间的差。

第 $i$ 个数据点的误差 $e_i$ 的数学定义是：

$e_i = y_i - \hat{y}_i$

这个公式告诉我们预测与实际值之间的偏差有多大。

### 示例：预测每日气温

假设我们正在测试一个预测每日最高气温（摄氏度）的模型。我们有四天实际的气温，也有模型对这些天所做的预测：

- **第 1 天：** 实际气温 ($y_1$) = 15°C，预测气温 ($\hat{y}_1$) = 16°C
- **第 2 天：** 实际气温 ($y_2$) = 20°C，预测气温 ($\hat{y}_2$) = 19°C
- **第 3 天：** 实际气温 ($y_3$) = 25°C，预测气温 ($\hat{y}_3$) = 23°C
- **第 4 天：** 实际气温 ($y_4$) = 18°C，预测气温 ($\hat{y}_4$) = 18°C

现在，让我们使用 $e_i = y_i - \hat{y}_i$ 计算每天的预测误差：

- **第 1 天的误差：** $e_1 = 15 - 16 = -1$°C
- **第 2 天的误差：** $e_2 = 20 - 19 = +1$°C
- **第 3 天的误差：** $e_3 = 25 - 23 = +2$°C
- **第 4 天的误差：** $e_4 = 18 - 18 = 0$°C

### 误差的解读

请注意误差的符号：

- **负误差 ($e_1 = -1$):** 这表示预测值（16°C）*高于*实际值（15°C）。模型高估了气温。
- **正误差 ($e_2 = +1$, $e_3 = +2$):** 这表示预测值（19°C，23°C）*低于*实际值（20°C，25°C）。模型低估了气温。
- **零误差 ($e_4 = 0$):** 这表示预测值（18°C）与实际值（18°C）完全相同。模型对这一天的预测是完美的。

计算这些单个误差是基本的第一步。每个误差都告诉我们某个特定预测不准确的程度和方向。



![实际气温与预测气温对比](plots/3991-0.json)



> 散点图显示了实际气温与预测气温的对比。虚线代表完美预测 ($y = \hat{y}$)。箭头表示三个数据点的预测误差（垂直距离）。

尽管查看单个误差能提供信息，特别是对于理解模型的具体不足之处，但它不能给我们一个单一的、衡量模型在所有预测上表现的整体指标。如果预测有成千上万甚至数百万个，我们需要一种方法将这些单个误差汇总成一个或多个指标。

这些单个误差 $e_1, e_2, e_3, ...$ 是我们将用来计算诸如平均绝对误差 (MAE)、均方误差 (MSE) 和均方根误差 (RMSE) 等汇总回归指标的基本组成部分，我们接下来会介绍这些内容。

## 参考资料

- [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2013)
  Publisher: Springer; Pages: Chapter 3
  本书以入门级介绍统计学习方法，包括线性回归中残差的定义和实际应用。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 3
  一本全面性的教材，提供统计学习的理论处理，详细阐述线性回归模型中的残差。
- [Introductory Econometrics: A Modern Approach](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge/9781337558860/) — Jeffrey M. Wooldridge (2019)
  Publisher: Cengage Learning; Pages: Chapter 2
  这本广泛使用的计量经济学教科书在线性回归背景下定义了预测误差（残差），侧重于它们在模型估计和分析中的作用。
