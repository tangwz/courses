---
course: "basics-model-evaluation-metrics"
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-4-preparing-data-evaluation"
sourceId: 803
chapter: "preparing-data-evaluation"
title: "为评估准备数据"
order: 4
description: "了解将数据划分为训练集和测试集的重要性，以进行无偏的模型评估。"
hasQuiz: true
---

你已经学习了如何计算分类和回归模型的各种指标。但要正确应用这些指标，需要一种特定的数据处理方法。在模型训练所用的*相同*数据上进行评估，会得到对其在新数据（未见过的数据）上实际表现的不可靠且往往过于乐观的估计。

本章介绍数据划分的基本做法。我们将介绍：

*   为什么在未见过的数据上进行评估对于衡量泛化能力很重要。
*   训练集和测试集的理念与作用。
*   执行训练集-测试集划分的标准步骤。
*   常见的比例以及随机性在划分中的作用。
*   单次划分的局限性以及对交叉验证理念的简要介绍。

在本章结束时，你将懂得如何正确准备数据，以获得对机器学习模型更可靠的评估结果。
