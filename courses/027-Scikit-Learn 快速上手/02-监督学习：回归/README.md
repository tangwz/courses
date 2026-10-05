# 第 2 章：监督学习：回归

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-2-supervised-learning-regression)

[返回课程目录](../README.md)

本章介绍回归任务中的监督学习。回归旨在预测连续数值，例如预测房价或气温，基于相关特征。

你将从线性回归的基本知识开始，它是这类问题的基本算法。我们将涵盖如何：

*   使用 Scikit-learn 统一的 API 实现线性回归模型，特别是 `LinearRegression` 估计器。
*   解释模型学习到的系数，以了解特征的重要性。
*   使用标准回归指标评估模型性能，例如平均绝对误差 (MAE)、均方误差 (MSE) 和 $R^2$ 分数。
*   在 Scikit-learn 中高效地计算这些指标。

在本章结束时，你将使用一个实际数据集，构建并评估一个完整的回归模型，应用所学知识。

## 小节

- 1. [回归问题简介](01-%E5%9B%9E%E5%BD%92%E9%97%AE%E9%A2%98%E7%AE%80%E4%BB%8B.md)
- 2. [线性回归基本原理](02-%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86.md)
- 3. [使用Scikit-learn实现线性回归](03-%E4%BD%BF%E7%94%A8Scikit-learn%E5%AE%9E%E7%8E%B0%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92.md)
- 4. [解释模型系数](04-%E8%A7%A3%E9%87%8A%E6%A8%A1%E5%9E%8B%E7%B3%BB%E6%95%B0.md)
- 5. [回归评估指标](05-%E5%9B%9E%E5%BD%92%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.md)
- 6. [Scikit-learn 中的指标计算](06-Scikit-learn%20%E4%B8%AD%E7%9A%84%E6%8C%87%E6%A0%87%E8%AE%A1%E7%AE%97.md)
- 7. [动手实践：构建回归模型](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-2-supervised-learning-regression/quiz)
