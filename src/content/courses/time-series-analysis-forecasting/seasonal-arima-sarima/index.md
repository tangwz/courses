---
course: "time-series-analysis-forecasting"
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima"
sourceId: 725
chapter: "seasonal-arima-sarima"
title: "使用SARIMA处理季节性"
order: 5
description: "使用SARIMA扩展ARIMA模型，以处理季节性模式。学习季节性数据的参数选择、模型拟合和预测方法。"
hasQuiz: true
---

尽管ARIMA模型对许多时间序列都很有效，但在处理具有强烈季节性模式（即在固定周期内重复出现的可预测周期，如月度或季度变化）的数据时，它们常常显得不足。

本章将介绍季节性ARIMA（SARIMA），它是专门为具有季节性组成部分的时间序列数据建模和预测而设计的扩展。我们将介绍SARIMA如何将季节性项与您在ARIMA中学到的非季节性部分相结合。

您将学习如何：

*   理解SARIMA模型的结构，其表示形式为 $ SARIMA(p, d, q)(P, D, Q)_m $，其中 $ (P, D, Q) $ 代表季节性自回归、差分和移动平均阶数，$ m $ 是季节性周期。
*   使用自相关函数（ACF）图和偏自相关函数（PACF）图来识别潜在的季节性阶数。
*   为非季节性 $ (p, d, q) $ 和季节性 $ (P, D, Q)_m $ 参数选择恰当的数值。
*   使用Python的`statsmodels`库将SARIMA模型拟合到数据。
*   通过检验残差来诊断已拟合的模型。
*   生成明确考虑季节性影响的预测。

在本章结束时，您将能够应用SARIMA模型，有效地分析和预测受季节性影响的时间序列。
