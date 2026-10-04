---
course: "time-series-analysis-forecasting"
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting"
sourceId: 722
chapter: "arima-models-forecasting"
title: "ARIMA 模型用于预测"
order: 4
description: "学习如何使用 Python 构建、拟合和评估自回归整合移动平均 (ARIMA) 模型进行时间序列预测。"
hasQuiz: true
---

在前几章中，我们已经确定了如何分解时间序列、检验平稳性，并借助 ACF 和 PACF 图识别可能的模型结构，现在我们将侧重于构建统计预测模型。本章介绍自回归整合移动平均 (ARIMA) 模型系列，这是一类广泛应用于分析和预测平稳时间序列数据的模型。

你将了解其核心组成部分：
* **自回归 (AR) 模型：** 过去的数值如何影响当前数值。
* **移动平均 (MA) 模型：** 过去的预测误差如何影响当前数值。
* **ARMA 模型：** 结合 AR 和 MA 分量用于平稳序列。
* **ARIMA 模型：** 引入差分（“整合”部分）以处理非平稳序列，再应用 ARMA 建模。

我们将讲解如何确定 ARIMA 模型阶数（表示为 ($p, d, q$)），依据 ACF/PACF 分析所得。你将学习如何使用 Python 的 `statsmodels` 库将这些模型拟合到数据，通过检查残差诊断模型的拟合程度，并为未来时间点生成预测。本章最后将通过动手实践，构建和评估一个完整的 ARIMA 预测工作流程。
