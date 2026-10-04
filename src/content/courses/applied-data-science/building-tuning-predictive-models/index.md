---
course: "applied-data-science"
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-3-building-tuning-predictive-models"
sourceId: 518
chapter: "building-tuning-predictive-models"
title: "构建与调整预测模型"
order: 3
description: "使用Python实现、评估并优化常见的监督机器学习模型。"
hasQuiz: true
---

准备数据并构建有意义的特征后，重心将转向建立能够学习规律并进行预测的模型。本章着重讲解监督学习，即模型从带有标签的数据中学习。

我们将首先对常见的回归和分类算法进行简要回顾，随后使用Python的scikit-learn库进行实际操作。您将应用线性模型（如线性回归和逻辑回归）、基于树的方法（如决策树和随机森林），以及集成技术（包括梯度提升机XGBoost和LightGBM）。

构建有效模型的一个重要环节是正确评估并优化其设置。我们将介绍适用于不同任务的评估指标，不局限于基本准确率，还会涵盖精确度、召回率、$F_1$ 分数和ROC AUC。您将学习实施可靠的验证策略，例如$k$折交叉验证。最后，我们将讨论超参数调优，运用网格搜索和随机搜索等系统方法来寻找最佳模型配置。

学完本章，您将能够训练、严格评估和调整多种标准的监督学习模型，以应对预测任务。
