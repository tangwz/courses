---
course: "data-versioning-experiment-tracking"
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-3-tracking-experiments-mlflow"
sourceId: 811
chapter: "tracking-experiments-mlflow"
title: "使用 MLflow 追踪实验"
order: 3
description: "学习如何使用 MLflow Tracking 系统地追踪机器学习实验。"
hasQuiz: false
---

前一章主要介绍如何使用 DVC 管理数据集，而开发机器学习模型则会遇到另一类管理挑战。模型训练通常需要多次迭代，期间会调整超参数、代码或进行特征工程。如果没有条理分明的方法，就很难回溯是哪些设置产生了特定的结果，也难以准确复现过去的实验。

本章将介绍 MLflow Tracking，它是一种系统记录模型开发工作的方案。我们将讲解如何：
*   设置 MLflow 环境。
*   修改训练脚本以记录重要信息：参数、指标以及模型或图表等工件。
*   将相关的训练尝试归类到实验中。
*   使用 MLflow 界面查看、搜索和比较不同运行的结果。

采纳这些做法能为您的开发过程提供清晰的记录，有助于提高模型的复现性及迭代效率。
