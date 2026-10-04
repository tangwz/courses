---
course: "intro-feature-engineering"
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-4-feature-scaling-transformation"
sourceId: 446
chapter: "feature-scaling-transformation"
title: "特征缩放与变换"
order: 4
description: "学习缩放数值特征的方法，以及应用变换来改变数据分布。"
hasQuiz: true
---

既然我们有了处理类别数据的方法，我们接下来看数值特征。原始数值特征的范围和分布会直接影响对特征尺度敏感的算法的有效性，例如基于距离的方法或使用梯度下降的方法。本章主要介绍通过调整数值数据的尺度和分布来准备数据进行模型训练的技术。

我们将介绍常见的缩放方法，包括标准化（$Z$-score scaling），它使数据具有零均值和单位方差（$Z = \frac{x - \mu}{\sigma}$），以及归一化（Min-Max scaling），它将数值限制在 [0, 1] 等特定区间。我们也将介绍适用于含有异常值数据的鲁棒缩放（Robust Scaling）。此外，你还将学习对数、Box-Cox 和 Yeo-Johnson 等变换方法，用来改变偏斜分布，并使数据更适合某些模型假设。本章将指导你如何有效地选择和应用这些技术，使用 Scikit-learn。
