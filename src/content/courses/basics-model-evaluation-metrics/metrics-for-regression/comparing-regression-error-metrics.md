---
course: "basics-model-evaluation-metrics"
chapter: "metrics-for-regression"
lesson: "comparing-regression-error-metrics"
sourceId: 4004
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-3-metrics-for-regression/comparing-regression-error-metrics"
title: "比较平均绝对误差 (MAE)、均方误差 (MSE) 和均方根误差 (RMSE)"
description: "讨论平均绝对误差 (MAE)、均方误差 (MSE) 和均方根误差 (RMSE) 之间的区别以及何时偏好某个指标。"
order: 6
plots: ["plots/4004-0.json"]
sourceHash: "42ca782efb808b7e04386705d52306f146ab3f188e8a1b2ed02b2f49ea70af0b"
sourceCorrections: []
---

平均绝对误差 (MAE)、均方误差 (MSE) 和均方根误差 (RMSE) 是评估回归模型的常见指标。这三种指标都能衡量回归模型的平均误差，但它们的方式不同，且各自具有不同的特性。理解这些差异对于为您的具体问题选择合适的指标以及正确解释模型表现非常重要。

让我们并排比较这三种指标。

### 计量单位

其中一个最实用的区别在于结果误差值的单位：

- **MAE：** 误差以目标变量的**原始单位**衡量。如果您预测的房价单位是美元，那么MAE也将以美元为单位。这使其非常容易理解。MAE为10,000美元意味着预测平均偏差10,000美元。
- **MSE：** 误差以目标变量的**平方单位**衡量。如果您预测的价格单位是美元，那么MSE将以美元的平方（$dollars^2$）为单位。这在某些情况下数学上很方便，但它不容易直接联系到目标变量的尺度进行解释。100,000,000美元的平方在实际意义上到底意味着什么？这更难理解。
- **RMSE：** 通过对MSE取平方根，RMSE将误差衡量值恢复到目标变量的**原始单位**。与MAE类似，如果您的目标变量是美元，RMSE也以美元为单位。这使得它比MSE更容易解释，代表着误差的一种典型量级。

**可解释性：** MAE和RMSE通常比MSE更容易解释，因为它们的单位与目标变量一致。

### 对误差（特别是大误差）的敏感度

这便是MAE与MSE/RMSE之间的核心区别所在：它们如何处理不同大小的误差。

- **MAE：** 计算*绝对*误差的平均值：

  
  $$
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  $$
  

  因为它使用绝对值，MAE线性处理每个误差。一个偏差10的预测对总误差的贡献恰好是一个偏差5的预测的两倍。它不会额外加权更大的误差。这意味着MAE对*离群值*（严重不正确的预测）不那么敏感。
- **MSE：** 计算*平方*误差的平均值：

  
  $$
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$
  

  通过对误差项 $(y_i - \hat{y}_i)$ 进行平方，MSE对大误差施加的惩罚远重于小误差。一个误差为10对总和的贡献是 $10^2 = 100$，而一个误差为5只贡献 $5^2 = 25$。误差为10的贡献是四倍，而不仅仅是两倍。这使得MSE对离群值非常敏感。少量具有大误差的预测可以大幅提高MSE分数。
- **RMSE：** 作为MSE的平方根，RMSE与MSE一样对大误差敏感，尽管最终值被恢复到原始单位。

  
  $$
  \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
  $$
  

  与MSE类似，RMSE会受到离群值的明显影响，因为平方运算发生在求平均和取平方根*之前*。

**敏感度示例：**

设想一个模型的两组预测误差：

1. 正常误差：`[2, -1, 3, -2, 1]`（误差相对较小）
2. 包含离群值的误差：`[2, -1, 3, -15, 1]`（其中一个误差大得多）

让我们计算这两组的MAE和RMSE（我们跳过MSE，因为它单位不同，不便于直接比较）：

- **第1组（正常误差）：**

  - MAE = `(|2|+|-1|+|3|+|-2|+|1|) / 5 = (2+1+3+2+1) / 5 = 9 / 5 = 1.8`
  - MSE = `(2^2+(-1)^2+3^2+(-2)^2+1^2) / 5 = (4+1+9+4+1) / 5 = 19 / 5 = 3.8`
  - RMSE = `sqrt(3.8) ≈ 1.95`
- **第2组（包含离群值）：**

  - MAE = `(|2|+|-1|+|3|+|-15|+|1|) / 5 = (2+1+3+15+1) / 5 = 22 / 5 = 4.4`
  - RMSE = `sqrt((2^2+(-1)^2+3^2+(-15)^2+1^2) / 5) = sqrt((4+1+9+225+1) / 5) = sqrt(240 / 5) = sqrt(48) ≈ 6.93`

请注意单个大误差 (-15) 如何影响了这些指标：

- MAE从1.8增加到4.4（增加了2.6）。
- RMSE从1.95增加到6.93（增加了近5.0）。

与MAE相比，RMSE被这个单个离群值大幅拉高。



![离群值对MAE与RMSE的影响](plots/4004-0.json)



> 为两组误差计算的MAE和RMSE：一组是典型误差，另一组包含一个单个大离群值。当存在离群值时，RMSE表现出明显更大的相对增长，突出了其对大误差的敏感性。

### 您应该选择哪个指标？

MAE、MSE和RMSE的选择取决于您的具体目标以及您希望如何处理误差：

- **在以下情况下选择MAE：**

  - 您希望指标易于直接以原始单位解释。
  - 您不希望离群值主导该指标。您认为偏差10的错误比偏差5的错误糟糕两倍。
  - 您希望一个直接衡量平均误差大小的方法。
- **在以下情况下选择RMSE（或MSE）：**

  - 您的应用中特别不希望出现大误差，并且您希望指标能强烈反映这一点。您认为偏差10的错误比偏差5的错误糟糕得多。
  - 您希望指标对离群值敏感，可能凸显出偶尔产生非常糟糕预测的模型。
  - RMSE在报告和解释时通常比MSE更受欢迎，因为它以原始单位表示。MSE由于其数学属性（例如易于微分），可能在学习算法内部使用得更频繁。
- **同时考虑两者：** 通常查看多个指标很有用。如果您的RMSE明显高于MAE，这可能表明存在抬高RMSE值的大误差（离群值）。检查这两个指标可以帮助您更全面地了解模型的误差分布。

并非所有回归问题都存在单一的“最佳”误差指标。理解它们的特性有助于您根据具体需求选择和解释与评估模型表现最相关的指标。

## 参考资料

- [An Introduction to Statistical Learning: with Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer; DOI: [10.1007/978-1-0716-2963-2](https://doi.org/10.1007/978-1-0716-2963-2)
  一本常用的统计学习教科书，涵盖回归基础概念及MAE、MSE、RMSE等评估指标。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本全面的参考书，涵盖统计学习的理论方面，包括回归损失函数和指标的数学基础和特性。
- [scikit-learn User Guide: Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) — scikit-learn developers (2024)
  提供MAE、MSE、RMSE等常用回归指标的实际定义、实现细节和比较。
