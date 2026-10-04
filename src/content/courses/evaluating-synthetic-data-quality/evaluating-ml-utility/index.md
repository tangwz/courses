---
course: "evaluating-synthetic-data-quality"
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-3-evaluating-ml-utility"
sourceId: 1032
chapter: "evaluating-ml-utility"
title: "评估机器学习效用"
order: 3
description: "评估合成数据在训练机器学习模型时作为真实数据替代品的表现。"
hasQuiz: false
---

在评估了真实数据集与合成数据集之间的统计相似性后，我们将转向一个重要的实际问题：合成数据在训练实际机器学习模型方面有多大用处？本章侧重于量化这种*机器学习效用*。

你将学习标准的评估方法，主要是训练-合成-测试-真实 (TSTR) 方法。在TSTR中，模型仅使用合成数据进行训练，然后在一个留存的真实数据集上进行评估。我们还将考察补充性的训练-真实-测试-合成 (TRTS) 方法。

主要涵盖的技术有：

*   比较真实数据训练的模型与合成数据训练的模型之间的标准性能指标（如准确率、F1分数或AUC）。
*   分析合成数据训练的模型是否能学到与真实数据训练的模型相似的特征重要性模式。
*   理解使用合成数据可能如何影响超参数调优的结果。

目标是提供客观的衡量标准，以确定合成数据是否能有效地替代真实数据在你的下游机器学习应用中。这包括动手实践这些评估流程。
