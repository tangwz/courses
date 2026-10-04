---
course: "rnns-and-sequence-modeling"
chapter: "evaluating-tuning-sequence-models"
lesson: "metrics-sequence-prediction"
sourceId: 2646
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-10-evaluating-tuning-sequence-models/metrics-sequence-prediction"
title: "序列预测的评估指标"
description: "讨论适用于类似回归的序列任务（MSE、MAE）的指标，例如时间序列预测。"
order: 2
plots: ["plots/2646-0.json"]
sourceHash: "2edb3864badac5f7d1e794d6de2f94267efd398e4d0a45ec3f6da024cb04a37e"
sourceCorrections: []
---

当您的序列模型目标是预测未来数值时，例如预测股票价格、预测能源消耗或估计随时间变化的传感器读数，您正在处理一个序列预测任务。与预测离散类别的分类任务不同，此处的目标通常是预测连续值。评估这些模型需要能衡量预测序列与实际序列之间差异的指标。

常用回归指标通常为此目的而调整使用。它们衡量预测时间范围内所有时间步的平均误差。让我们看看最常用的几个。

### 平均绝对误差 (MAE)

平均绝对误差衡量预测值与实际值之间误差的平均大小，不考虑其方向。它是整个序列（或序列集）上绝对差异的平均值。

对于具有 $T$ 个时间步的单个序列，其中 $y_t$ 是时间步 $t$ 的实际值，$\hat{y}_t$ 是预测值，MAE 计算如下：


$$
MAE = \frac{1}{T} \sum_{t=1}^{T} |y_t - \hat{y}_t|
$$


**解释：**
MAE 以您数据的原始单位给出平均误差。如果您预测摄氏温度，MAE 会告诉您预测值与实际读数之间的平均绝对温差。

**特性：**

- **直观：** 易于理解和说明。误差可以直接以目标变量的单位解释。
- **对异常值不敏感：** 由于它使用绝对差值，因此它不会像平方误差指标那样严厉惩罚较大的个体误差（异常值）。如果偶尔出现较大误差是预期或可接受的，这会很有用。

### 均方误差 (MSE)

均方误差衡量误差平方的平均值。由于平方运算，它对较大误差施加更高的惩罚。

MSE 的公式为：


$$
MSE = \frac{1}{T} \sum_{t=1}^{T} (y_t - \hat{y}_t)^2
$$


**解释：**
MSE 以原始单位的平方衡量（例如，摄氏度平方），这使得它与 MAE 或 RMSE 相比，直接解释起来不那么直观。但它被广泛使用，部分原因是平方项使其在数学上方便用于优化算法（如梯度下降 (gradient descent)），因为它会产生平滑、可微分的损失函数 (loss function)。

**特性：**

- **惩罚大误差：** 对误差项进行平方意味着较大的误差对总 MSE 的贡献不成比例地更多。当与实际值的较大偏差特别不理想时，这使其适用。
- **数学上便捷：** 该函数是连续可微分的，简化了其在基于梯度的优化中的使用。
- **对异常值敏感：** 平方效应也使得 MSE 比 MAE 对异常值更敏感。单个大误差可以明显地夸大 MSE。

### 均方根误差 (RMSE)

均方根误差就是 MSE 的平方根。求平方根将误差指标带回目标变量的原始单位，使其比 MSE 更具可解释性。

RMSE 的公式为：


$$
RMSE = \sqrt{MSE} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (y_t - \hat{y}_t)^2}
$$


**解释：**
与 MAE 类似，RMSE 以与目标变量相同的单位表示（例如，摄氏度）。它代表了误差的某种“典型”大小。

**特性：**

- **可解释的单位：** 具有 MAE 的可解释性优点。
- **惩罚大误差（如 MSE）：** 保留了比 MAE 更严厉惩罚大误差的特点，因为它源自平方误差。
- **常用：** 报告性能时，通常比 MSE 更受青睐，因为其单位更直观，同时仍反映了 MSE 固有的对大误差的敏感性。

这是一个比较实际值与两个模型预测值的简单可视化。



![实际值 vs. 预测时间序列](plots/2646-0.json)



> 实际时间序列数据与两个模型预测值的比较。模型 A 通常紧密跟踪实际值，从而得到较低的 MAE 和 RMSE。模型 B 则表现出较大的偏差，导致误差指标较高。

### 平均绝对百分比误差 (MAPE)

有时，您可能对相对于实际值的误差感兴趣。平均绝对百分比误差将平均误差表示为实际值的百分比。


$$
MAPE = \frac{100\%}{T} \sum_{t=1}^{T} \left| \frac{y_t - \hat{y}_t}{y_t} \right|
$$


**解释：**
MAPE 给出平均误差百分比的感觉。10% 的 MAPE 意味着预测值平均与实际值相差约 10%。

**特性：**

- **相对误差：** 当数据规模差异很大或您需要一个与规模无关的误差衡量标准时有用。
- **潜在问题：**
  - 如果任何实际值 $y_t$ 为零，它将是未定义或无限大。
  - 如果实际值非常接近零，它可能产生倾斜的结果。
  - 与相同绝对量但低于实际值的预测相比，它对高于实际值的预测施加更重的惩罚（不对称）。

### 选择合适的指标

这些指标的选择取决于您预测问题的具体目标和特点：

- 使用 **MAE**，如果您想要一个易于解释的原始单位误差指标，并且希望线性处理所有误差，而不对异常值赋予额外权重 (weight)。
- 主要在训练期间使用 **MSE** 作为损失函数 (loss function)，因为它具有数学特性。请注意它对异常值的敏感性及其单位（平方）。
- 使用 **RMSE**，如果您想要一个原始单位的误差指标（如 MAE），但仍想更显著地惩罚较大误差（如 MSE）。它通常是报告性能的一个良好平衡点。
- 当相对误差比绝对误差更重要，并且您确信实际值不会为零或非常接近零时，使用 **MAPE**。

在实践中，您通常会监测其中的多个指标。例如，您可以在训练期间使用 MSE 作为损失函数来优化模型，但报告 RMSE 和 MAE，以提供模型性能的更完整视图。在处理序列时，这些指标通常在评估序列中的所有时间步上取平均值，并可能在评估数据集中的所有序列上取平均值。

## 参考资料

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman, George Athanasopoulos (2021)
  Publisher: OTexts
  一本全面的在线教科书，详细介绍了各种预测方法和时间序列预测特有的评估指标。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  深度学习的核心教材，提供了许多序列预测模型所依据的损失函数和评估指标的基础知识。
- [Hands-On Machine Learning with Scikit-Learn, Keras, & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，阐述了机器学习项目中各种回归指标的应用和解释。
- [A survey of forecast error measures](https://www.researchgate.net/publication/271257404_A_survey_of_forecast_error_measures) — M. Shcherbakov, A. Breitenstein, and J. Reif (2013)
  Journal: Proceedings of the 32nd International Symposium on Forecasting
  这篇论文系统地调查了预测误差指标，讨论了它们的性质以及对不同预测场景的适用性。
