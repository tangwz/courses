---
course: "adversarial-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-6-evaluating-model-robustness"
sourceId: 825
chapter: "evaluating-model-robustness"
title: "评估模型抗攻击能力"
order: 6
description: "学习使用指标、基准、自适应攻击和评估框架来评估模型安全性。"
hasQuiz: false
---

在考察了攻击机器学习模型的方法以及抵御这些攻击的策略之后，下一步是弄清楚这些防御措施的实际效果。仅仅部署防御机制是不够的；我们需要可靠的方法来衡量模型抵御潜在威胁的安全程度。本章主要介绍严格评估模型安全的方法和实践。

你将了解到用于量化安全性的标准指标，例如攻击下的准确性，或导致错误分类所需的最小扰动幅度，这通常用$L_p$范数表示，如$L_0$、$L_2$或$L_\infty$。我们将审视ART和CleverHans等辅助这些评估的常见基准测试工具和框架。评估的重要部分包括设计强大的自适应攻击，这些攻击是专门为测试防御极限而定制的，以避免弱评估带来的虚假安全感。我们还将讨论如何在考虑不同攻击者假设（威胁模型）的情况下进行评估，以及如何有效地解读所得出的安全评估结果。在本章结束时，你将有能力为机器学习模型建立并运行系统的安全评估。
