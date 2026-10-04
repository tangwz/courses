---
course: "getting-started-with-gradient-boosting-algorithms"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-4-advanced-gradient-boosting-xgboost"
sourceId: 1313
chapter: "advanced-gradient-boosting-xgboost"
title: "梯度提升进阶：XGBoost"
order: 4
description: "XGBoost 库使用指南。了解其性能优势、正则化特性，以及如何使用 XGBoost API 构建模型。"
hasQuiz: false
---

在 Scikit-Learn 中实践了梯度提升机之后，我们现在转向 XGBoost，它是“极端梯度提升”的缩写。该库是一个优化过的、分布式的梯度提升框架，旨在实现高效率和准确性。它在机器学习竞赛中持续取得优异表现，并在生产环境中被广泛应用，使其成为任何实践者不可或缺的工具。

本章主要介绍有助于 XGBoost 性能的特有属性。我们将考察它在架构方面对标准 GBM 的改进，其中包含更规范化的正则化方法。XGBoost 的目标函数明确加入了对模型复杂度的惩罚，通常表示为：

$$Obj(\Theta) = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$$

其中，$l$ 代表损失函数，$\Omega$ 是惩罚树 $f_k$ 复杂程度的正则化项。我们还将讲解其内置的处理机制，用于处理缺失值，这可以简化数据预处理过程。本章最后通过实践介绍 XGBoost Python API，包括安装、特有的 `DMatrix` 数据结构、模型训练和预测。
