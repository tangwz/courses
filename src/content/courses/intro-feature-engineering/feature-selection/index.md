---
course: "intro-feature-engineering"
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-6-feature-selection"
sourceId: 455
chapter: "feature-selection"
title: "特征选择"
order: 6
description: "学习选择最相关特征的方法，以减少维度并提升模型的泛化能力。"
hasQuiz: true
---

在之前的步骤中生成了可能许多特征后，接下来的任务是确定哪些对模型构建最有价值。包含不相关或冗余的特征会增加计算成本，使模型难以理解，并可能导致过拟合。特征选择旨在从原始数据集中识别并保留那些信息量最大的特征。

本章将介绍系统性减少特征数量的方法，同时保持甚至提高模型性能。您将了解到：

*   **过滤法：** 这些方法根据特征的统计特性进行评估，例如方差（使用`VarianceThreshold`）或与目标变量的关系（使用ANOVA F值或$χ^2$等检验），不涉及特定的机器学习模型。
*   **封装法：** 这些方法使用预测模型对特征子集进行评分。我们将介绍递归特征消除（RFE）等方法，该方法会迭代地移除贡献最小的特征。
*   **嵌入法：** 这些方法将特征选择直接融入模型训练过程中。例如L1正则化（Lasso），它可以将不重要特征的系数收缩为零，以及使用从随机森林等树形算法获得的特征重要性得分。

学完本章，您将能够运用Scikit-learn等库中的各种特征选择策略，构建更高效、更实用的机器学习模型。
