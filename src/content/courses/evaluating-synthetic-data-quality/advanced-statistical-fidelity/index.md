---
course: "evaluating-synthetic-data-quality"
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-2-advanced-statistical-fidelity"
sourceId: 1028
chapter: "advanced-statistical-fidelity"
title: "进阶统计保真度评估"
order: 2
description: "应用高级统计检验，严谨比较合成数据与真实数据的分布。"
hasQuiz: false
---

第一章介绍了评估合成数据的主要维度。统计保真度，衡量合成数据的统计特性与真实数据之间的贴近程度，是一个基本方面。虽然比较针对单个特征的基本统计量，例如均值 ($ \mu $) 或标准差 ($ \sigma $)，能提供一个起点，但它通常无法捕捉真实数据集中存在的复杂高维关联。简单地匹配每个特征 $X_i$ 的边际分布 $P(X_i)$ 并不能保证联合分布 $P(X_1, X_2, ..., P_n)$ 得到准确的体现。

本章侧重于用于更全面统计保真度评估的进阶方法。您将学习如何：
*   比较多变量分布，不限于单特征分析。
*   应用严谨的假设检验，这些检验专门为评估数据集间分布相似性而设计。
*   分析和比较相关和协方差结构，以确保变量间的关联性得以保持。
*   运用信息论度量来量化分布差异。
*   采用倾向分数方法来评估合成数据点与真实数据点之间的可区分性。

本章最后包含一个实践环节，您会在此环节使用 Python 库实现多个多变量统计检验。
