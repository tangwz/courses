---
course: "bayesian-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-4-gaussian-processes"
sourceId: 742
chapter: "gaussian-processes"
title: "高斯过程"
order: 4
description: "熟练掌握高斯过程，涵盖核函数选择、超参数调整、回归、分类和可扩展性。"
hasQuiz: false
---

之前我们主要讲解了近似后验分布的通用方法，例如MCMC和变分推断。现在我们将注意力转向高斯过程（GP），它是一种在贝叶斯非参数建模中效能显著的框架。与由固定数量参数定义的模型不同，GP通过直接在函数上设置先验，允许模型复杂度根据数据调整。

在本章中，您将学到：

*   高斯过程作为函数上的分布的基本原理。
*   协方差函数（或核函数）如何决定从高斯过程抽取的函数的性质（如平滑度）。
*   高斯过程回归的数学表述，使得能够进行预测并给出相应的不确定性估计，通常表示为预测均值 $$ \mathbb{E}[f_* | \mathbf{y}] $$ 和方差 $$ \text{Var}[f_* | \mathbf{y}] $$。
*   通过最大化边际似然来选择合适的核函数和优化超参数的方法。
*   将高斯过程应用于分类问题的方法。
*   近似方法，例如使用诱导点进行稀疏高斯过程，以便将高斯过程应用于精确计算变得不可行的大型数据集。

我们将学习这些理论知识，并通过实际编程练习进行应用。
