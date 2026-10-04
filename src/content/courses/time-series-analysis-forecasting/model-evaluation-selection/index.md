---
course: "time-series-analysis-forecasting"
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-6-model-evaluation-selection"
sourceId: 727
chapter: "model-evaluation-selection"
title: "模型评估与选择"
order: 6
description: "学习如何使用各种指标评估时间序列预测模型的表现，并比较不同的模型。"
hasQuiz: true
---

在之前的章节中构建了ARIMA和SARIMA等预测模型后，下一步是评估它们的表现。建立模型只是流程的一部分；我们需要客观的方法来确定它预测未来值的准确程度，以及它与替代方案相比如何。

在本章中，您将学习针对时间序列数据的有效模型评估方法。我们将从如何正确地将数据分成训练集和测试集开始，同时遵守时间顺序以避免“预见未来”。然后，您将学习计算和解释常见的预测准确性指标，包括：

*   平均绝对误差 ($MAE$)
*   均方误差 ($MSE$)
*   均方根误差 ($RMSE$)
*   平均绝对百分比误差 ($MAPE$)

例如，$MAE$ 表示预测值与实际值之间的平均绝对差：
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |Actual_i - Forecast_i|$$

我们还将考察像赤池信息准则 ($AIC$) 和贝叶斯信息准则 ($BIC$) 这样的信息准则，它们通过平衡模型拟合度和复杂度来辅助模型选择。在本章结束时，您将能够应用这些指标和准则来比较不同的预测模型，并可视化它们与实际数据的表现。
