# 第 6 章：回归分析简介

来源：[原章节](https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-6-introduction-regression-analysis)

[返回课程目录](../README.md)

在统计推断和数据描述的基础上，本章介绍使用回归分析对变量间关系进行建模的方法。我们将从最基本的简单线性回归技术开始。

您将学习如何从数学上定义简单线性回归模型，它通常表示为 $y = \beta_0 + \beta_1 x + \epsilon$。我们将讲解最小二乘法，用于估计模型参数（$\beta_0$ 和 $\beta_1$），以找到穿过数据的最佳拟合线。主要内容包括从实际角度解释这些估计系数的含义，并使用R方 ($R^2$) 和均方误差 (MSE) 等常用指标评估模型性能。

此外，我们将讨论线性回归所依据的、对有效推断所必需的基本假设。本章概述了如何将这些内容扩展到多元线性回归，即使用多个预测变量的情况。最后，实际示例将演示如何使用Statsmodels和Scikit-learn等Python库实现、拟合和评估回归模型。

## 小节

- 1. [简单线性回归模型](01-%E7%AE%80%E5%8D%95%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B.md)
- 2. [最小二乘估计法](02-%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E4%BC%B0%E8%AE%A1%E6%B3%95.md)
- 3. [解读回归系数](03-%E8%A7%A3%E8%AF%BB%E5%9B%9E%E5%BD%92%E7%B3%BB%E6%95%B0.md)
- 4. [模型评估指标 (R平方, 均方误差)](04-%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87%20%28R%E5%B9%B3%E6%96%B9%2C%20%E5%9D%87%E6%96%B9%E8%AF%AF%E5%B7%AE%29.md)
- 5. [线性回归的假设](05-%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E7%9A%84%E5%81%87%E8%AE%BE.md)
- 6. [多元线性回归概述](06-%E5%A4%9A%E5%85%83%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%A6%82%E8%BF%B0.md)
- 7. [使用Python构建回归模型](07-%E4%BD%BF%E7%94%A8Python%E6%9E%84%E5%BB%BA%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B.md)
- 8. [动手实践：拟合与评估线性模型](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%8B%9F%E5%90%88%E4%B8%8E%E8%AF%84%E4%BC%B0%E7%BA%BF%E6%80%A7%E6%A8%A1%E5%9E%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-6-introduction-regression-analysis/quiz)
