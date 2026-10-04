---
course: "intro-feature-engineering"
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-5-feature-creation"
sourceId: 451
chapter: "feature-creation"
title: "特征构建"
order: 5
description: "学习如何从现有特征中构建新特征，以期提升模型性能。"
hasQuiz: true
---

在前几章中，我们已经对现有数据特征进行了准备和转换，现在我们将重点转向构造全新的特征。构建有信息量的特征对于提升模型性能非常重要，因为新特征可以捕捉到复杂的关系或业务背景下的独特理解，而这些仅凭原始变量可能无法显现。

本章介绍从数据中构造有用特征的方法：

*   **交互特征：** 学习如何组合两个或更多特征以捕获它们共同产生的影响，例如，通过计算它们的乘积 $f_{interaction} = f_1 \times f_2$。
*   **多项式特征：** 从数值特征中产生高阶项（如 $f_{squared} = f_1^2$），以帮助模型捕获非线性模式。
*   **日期/时间特征提取：** 了解将时间戳数据分解为有用组成部分的方法，例如年、月、周几或小时。
*   **分箱：** 理解如何将连续数值数据转换为离散的类别箱。
*   **特定业务背景下的特征构建：** 简要讨论理解问题业务背景如何指导相关特征的构造。

我们将使用Python实现这些技术，主要借助Pandas进行数据处理，并使用Scikit-learn提供的专门特征工程工具。
