---
course: "time-series-analysis-forecasting"
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-1-intro-time-series-data"
sourceId: 710
chapter: "intro-time-series-data"
title: "时间序列数据介绍"
order: 1
description: "了解时间序列数据的基本原理、其组成（趋势、季节性、噪声）以及Python中的预处理步骤。"
hasQuiz: true
---

时间序列数据，即按时间顺序排列的观测值序列，在许多地方都经常出现，从金融和经济到天气模式和传感器读数。与标准截面数据不同，时间排序是其主要特点，这意味着观测值 $y_t$ 通常依赖于先前的数值，例如 $y_{t-1}$。

本章将介绍处理此类数据所需的基本知识。我们将涵盖以下内容：

*   识别时间序列中常见的独有特性以及趋势和季节性等成分。
*   使用Python中的Pandas库来高效地加载、处理和操作时间索引数据。
*   时间平移、计算滞后值和应用滚动窗口操作等方法。
*   可视化时间序列数据的基本方法，以便初步了解其结构。

在本章结束时，您将具备处理和准备时间序列数据集以进行分析的基本工具。
