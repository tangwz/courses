---
course: "monitoring-managing-ml-models-production"
sourceUrl: "https://apxml.com/zh/courses/monitoring-managing-ml-models-production/chapter-3-granular-performance-monitoring"
sourceId: 702
chapter: "granular-performance-monitoring"
title: "精细性能监控与诊断"
order: 3
description: "细致监控具体的机器学习性能指标，分析数据分段，并诊断问题。"
hasQuiz: false
---

发现像数据漂移这样的运行问题仅是考量的一部分。理解其对模型效果的准确影响，并找出性能变化*原因*，则需要更细致的视角。本章重点讲解精细性能监控和诊断的方法。

您将学习选择适用于特定任务的指标，不仅仅是简单的准确率 ($A$)，还包括精确率 ($P$) 和召回率 ($R$) 等指标。我们将讨论评估性能的方法，不仅是整体表现，还要针对数据的特定部分或切片进行评估，以找出隐藏的问题。此外，我们还会研究监控模型公平性的方法，分析异常值的影响，在性能下降时进行系统性根本原因分析，并将模型可解释性方法（如 SHAP 或 LIME）作为生产环境中的诊断工具。
