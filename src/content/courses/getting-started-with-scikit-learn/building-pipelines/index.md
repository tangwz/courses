---
course: "getting-started-with-scikit-learn"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-6-building-pipelines"
sourceId: 522
chapter: "building-pipelines"
title: "构建管道"
order: 6
description: "学习如何使用 Scikit-learn 管道将预处理步骤和估计器链接起来，从而简化机器学习工作流程。"
hasQuiz: true
---

在之前的章节中，我们将数据预处理和模型训练作为独立的步骤进行处理。顺序执行这些操作，尤其是在交叉验证循环中，可能会很麻烦，并且存在无意中将测试折叠中的信息泄露到训练过程中的风险。

本章介绍 Scikit-learn 的 `Pipeline` 对象，这是一个旨在将多个处理步骤（例如缩放器、编码器和缺失值填充器）与一个最终估计器（例如分类器或回归器）链接起来的工具。您将学习如何构建这些管道，以创建一个代表您整个建模流程的单一对象。我们将介绍如何使用 `GridSearchCV` 将管道与交叉验证和超参数调优有效地结合起来，确保预处理在每个折叠中正确应用。最后，我们将讨论如何使用 `ColumnTransformer` 来构建更复杂的管道，这些管道可以对数据集中的不同列子集应用不同的转换。
