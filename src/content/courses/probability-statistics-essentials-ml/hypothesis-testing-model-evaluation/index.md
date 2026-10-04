---
course: "probability-statistics-essentials-ml"
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-5-hypothesis-testing-model-evaluation"
sourceId: 458
chapter: "hypothesis-testing-model-evaluation"
title: "模型评估的假设检验"
order: 5
description: "应用假设检验体系：建立假设，理解误差和 p 值，并进行 t 检验和卡方检验等。"
hasQuiz: true
---

在处理数据时，尤其是在机器学习中，我们经常需要根据样本做出判断或验证假定。新模型是否明显优于旧模型？某个特定特征是否有可测量的影响？假设检验提供了一个有结构的统计体系来回答此类问题。

本章涵盖了假设检验的基本知识。您将学习如何：
*   建立零假设 ($H_0$) 和备择假设 ($H_1$)。
*   理解第一类 ($\alpha$) 和第二类 ($\beta$) 错误之间的权衡。
*   解读 p 值以做出统计判断。
*   应用常用检验方法，包括用于比较均值的 t 检验和用于分类数据分析的卡方检验。
*   对方差分析 (ANOVA) 有一个概览，用于比较多个组。

我们还将演示如何使用 Python 的 SciPy 库高效地实现这些检验，提供实用的工具用于模型评估和数据分析。
