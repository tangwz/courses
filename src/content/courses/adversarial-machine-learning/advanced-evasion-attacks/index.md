---
course: "adversarial-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-2-advanced-evasion-attacks"
sourceId: 817
chapter: "advanced-evasion-attacks"
title: "进阶规避攻击"
order: 2
description: "学习高阶规避攻击，如PGD、C&W、基于分数的、基于决策的和可迁移攻击。"
hasQuiz: false
---

在介绍完对抗性机器学习的基本原理后，本章集中于*规避攻击*。它们是在模型推断阶段执行的攻击，通过向输入数据引入精心制作的扰动，旨在导致模型错误分类。目标通常是找到一个小的扰动$\delta$，使得输入$x$被修改为$x_{adv} = x + \delta$，导致模型$f$输出一个不正确的预测，$f(x_{adv}) \neq f(x)$，同时满足对扰动大小的限制，这些限制通常使用像$||\delta||_p \le \epsilon$这样的$L_p$范数来定义。

本章审视了生成这些对抗样本的几种高级方法。我们将分析从基本的基于梯度的攻击（如FGSM、BIM）到更有效的迭代方法（如投影梯度下降（PGD））的演变。您将学习基于优化的方法，以Carlini & Wagner (C&W)攻击为例，这些方法常能找到高效、低失真的扰动。我们还将涵盖在攻击者知识有限的情况下适用的技术，包括基于分数的攻击（使用模型置信分数）和基于决策的攻击（仅使用最终预测标签）。此外，我们将研究不同模型之间攻击的可迁移性以及攻击集成模型的具体策略。本章最后是一个实践部分，您将在其中实现一些这些进阶规避攻击技术。
