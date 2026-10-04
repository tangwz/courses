---
course: "probability-statistics-essentials-ml"
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-6-introduction-regression-analysis"
sourceId: 459
chapter: "introduction-regression-analysis"
title: "回归分析简介"
order: 6
description: "了解简单和多元线性回归的基础，使用最小二乘法进行模型拟合，以及模型评估指标。"
hasQuiz: true
---

在统计推断和数据描述的基础上，本章介绍使用回归分析对变量间关系进行建模的方法。我们将从最基本的简单线性回归技术开始。

您将学习如何从数学上定义简单线性回归模型，它通常表示为 $y = \beta_0 + \beta_1 x + \epsilon$。我们将讲解最小二乘法，用于估计模型参数（$\beta_0$ 和 $\beta_1$），以找到穿过数据的最佳拟合线。主要内容包括从实际角度解释这些估计系数的含义，并使用R方 ($R^2$) 和均方误差 (MSE) 等常用指标评估模型性能。

此外，我们将讨论线性回归所依据的、对有效推断所必需的基本假设。本章概述了如何将这些内容扩展到多元线性回归，即使用多个预测变量的情况。最后，实际示例将演示如何使用Statsmodels和Scikit-learn等Python库实现、拟合和评估回归模型。
