---
course: "getting-started-with-gradient-boosting-algorithms"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-6-hyperparameter-tuning-and-optimization"
sourceId: 1315
chapter: "hyperparameter-tuning-and-optimization"
title: "超参数调整与模型优化"
order: 6
description: "学习优化梯度提升模型。本章介绍超参数调整，例如学习率、树深度和子采样，并使用网格搜索进行。"
hasQuiz: false
---

您已使用 Scikit-Learn、XGBoost 及其他高级库构建了梯度提升模型。尽管这些库的默认设置能提供一个不错的基准，但要针对特定问题获得出色表现，需要一个系统性的调整与优化过程。此过程称为超参数调整。

本章提供了一份有条理的指南，帮助您优化梯度提升模型。我们将从确定影响最大的超参数入手，这些超参数控制着模型的行为，例如提升阶段数 ($M$)、学习率 ($\eta$)，以及控制单个决策树复杂度的参数。

您将学习这些设置如何影响偏差-方差权衡，以及如何调整它们以防止过拟合。我们将介绍正则化技术，包括行和列子采样。最后，我们将实践系统化的搜索策略，包括网格搜索 (Grid Search) 和随机搜索 (Randomized Search)，以自动化为您的数据集寻找有效超参数组合的过程。本章最后有一个实践练习，在其中您将应用这些调整技术来提升模型的预测准确性。
