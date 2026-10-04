---
course: "mastering-gradient-boosting-algorithms"
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-3-regularization-gradient-boosting"
sourceId: 532
chapter: "regularization-gradient-boosting"
title: "梯度提升中的正则化"
order: 3
description: "学习专门应用于梯度提升框架内的进阶正则化技术，以应对过拟合。"
hasQuiz: false
---

梯度提升机（GBM）是高效的学习器，能够对数据中的复杂关联进行建模。然而，这种灵活性伴随一个缺点：GBM容易过拟合训练数据，捕捉到噪声，导致在新的、未见过的数据上泛化能力不足。本章专注于专门设计用于应对提升（boosting）框架内这种倾向的方法。

我们将考察适用于GBM的多种正则化策略。你将学习如何通过最大深度或叶节点最小样本数等约束来控制单个决策树（基本学习器）的复杂度，从而有助于防止过拟合。我们将重新审视收缩，也就是学习率($ \eta $)，分析其作为隐式正则器的作用。此外，我们将介绍子采样技术（行采样和特征采样），这些技术常被称为随机梯度提升，它们引入随机性以提高模型的稳定性。我们还将研究如何向目标函数添加显式惩罚项，例如$L_1$和$L_2$范数，可以直接对模型进行正则化，这种技术在XGBoost中突出体现。最后，我们将讨论诸如使用验证集进行提前停止等实用策略，以确定最佳提升迭代次数。

通过学习本章，你将对如何诊断梯度提升模型中可能出现的过拟合以及如何应用适当的正则化技术来构建在未见过数据上表现可靠的模型，获得实践性的理解。
