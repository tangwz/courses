---
course: "time-series-analysis-forecasting"
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-2-decomposition-stationarity"
sourceId: 713
chapter: "decomposition-stationarity"
title: "时间序列分解与平稳性"
order: 2
description: "学习时间序列分解以及平稳性的重要性。运用差分法和统计检验（如ADF）等方法。"
hasQuiz: true
---

在学习了如何加载、处理和可视化时间序列数据之后，我们现在将重点放在分析其内在结构并为建模做准备上。许多标准的时间序列模型都基于一个假设：序列的统计特性，例如其均值和方差，随时间保持不变。这种特性被称为**平稳性**。

通常，原始时间序列数据会表现出趋势或季节性模式，这意味着它*不平稳*。本章将介绍**时间序列分解**，这是一种将序列分离为趋势、季节性以及随机噪声等组成部分的方法（对于加法模型是$y_t = T_t + S_t + R_t$，对于乘法模型是$y_t = T_t \times S_t \times R_t$）。理解这些组成部分有助于获得见解并辅助模型选择。

我们将学习：
*   Python中分解时间序列数据的方法。
*   平稳性的定义及其重要性。
*   如何直观评估平稳性。
*   针对平稳性的正式统计检验，例如增广迪基-富勒（ADF）检验。
*   将不平稳数据转换为适用于建模的平稳数据的方法，主要是**差分法**。

在本章结束时，您将能够分解时间序列，并判断它们是否符合许多预测模型所需的平稳性假设。
