---
course: "getting-started-with-gradient-boosting-algorithms"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-3-gradient-boosting-with-scikit-learn"
sourceId: 1312
chapter: "gradient-boosting-with-scikit-learn"
title: "使用 Scikit-Learn 实现梯度提升"
order: 3
description: "学习使用 Python 的 Scikit-Learn 库实现用于分类和回归的梯度提升模型。内容包括模型解释方法。"
hasQuiz: false
---

在掌握了梯度提升机 (GBM) 的工作原理后，我们现在转向其实际应用。本章主要介绍如何使用 Scikit-Learn 实现这些模型。Scikit-Learn 是 Python 机器学习技术栈中的一个标准库。其统一的 API 为您构建第一个梯度提升模型提供了直接途径。

您将使用 Scikit-Learn 的两种主要实现：用于分类任务的 `GradientBoostingClassifier` 和用于回归问题的 `GradientBoostingRegressor`。我们将介绍将这些模型拟合到数据、生成预测结果以及理解决定模型行为的主要参数的标准流程，例如提升阶段的数量 ($n\_estimators$) 和学习率。

构建模型只是过程的一部分；模型解释也同等重要。我们还将说明如何理解训练好的模型。您将学习提取特征重要性分数，以识别哪些变量对预测结果影响最大，并使用偏依赖图来呈现单个特征与模型输出之间的关系。
