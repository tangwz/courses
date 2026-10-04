---
course: "autoencoders-representation-learning"
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-3-regularized-autoencoders"
sourceId: 646
chapter: "regularized-autoencoders"
title: "正则化自动编码器：构建稳固表示"
order: 3
description: "学习稀疏、去噪和收缩自动编码器等技术，以防止过拟合并获得更稳定的特征。"
hasQuiz: false
---

虽然标准自动编码器为学习压缩表示提供了一个根基，但它们有时会学到琐碎的解或过拟合训练数据，从而限制了它们在获取有意义数据结构方面的有效性。本章将介绍旨在克服这些问题并促进学习更稳定特征的正则化技术。

我们将考察几种主要方法。您将学习**稀疏自动编码器**，它们利用$L_1$正则化或Kullback–Leibler (KL)散度等惩罚，在潜在编码中强制实现稀疏性，从而有效地使网络关注最显著的信息。我们还将涵盖**去噪自动编码器 (DAEs)**，它们被训练从损坏的版本中重建干净数据，从而学习对输入噪声具有韧性的特征。此外，我们将讨论**收缩自动编码器 (CAEs)**，它们明确地惩罚所学特征对微小输入变化的敏感度。

通过理解和实施这些方法，您将能够构建泛化能力更好、并能为各种任务学习更有用表示的自动编码器模型。我们将比较这些技术并提供实际的实施指导，包括去噪自动编码器和稀疏自动编码器的动手练习。
