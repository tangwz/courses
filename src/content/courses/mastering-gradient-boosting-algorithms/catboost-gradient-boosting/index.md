---
course: "mastering-gradient-boosting-algorithms"
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-6-catboost-gradient-boosting"
sourceId: 538
chapter: "catboost-gradient-boosting"
title: "CatBoost：梯度提升决策树"
order: 6
description: "考察CatBoost处理类别特征的独特方法，包括有序提升和特征组合。"
hasQuiz: false
---

在考察了XGBoost和LightGBM之后，我们现在转向CatBoost。这是一个梯度提升库，专门为解决一个特定但普遍的难题而优化：即如何有效处理类别特征。传统的做法通常涉及预处理步骤，这些步骤可能不够理想，或导致诸如目标泄漏之类的问题。CatBoost将其处理类别数据的创新方案直接整合到算法中。

本章内容包括：

*   传统提升方法在处理类别变量时遇到的困难。
*   有序目标统计（Ordered TS）：CatBoost在编码类别特征时，如何最大限度地减少目标泄漏的方法。
*   有序提升（Ordered Boosting）：一种在训练过程中抵消预测漂移的技术。
*   自动特征组合：CatBoost如何生成类别特征之间的关联。
*   无偏树（Oblivious Trees）：CatBoost采用的对称决策树。
*   CatBoost API的使用：主要参数和具体的实现细节。

完成本章后，你将理解CatBoost的独特方法，并能够应用它们，尤其是在处理包含大量类别数据的问题时。
