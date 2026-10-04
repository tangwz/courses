---
course: "introduction-to-neural-networks"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-6-improving-network-performance"
sourceId: 540
chapter: "improving-network-performance"
title: "提升网络性能及泛化能力"
order: 6
description: "学习应对过拟合并提升神经网络泛化能力的技术。"
hasQuiz: true
---

你已成功训练了一个基础神经网络，使其能够从训练数据中学习模式。然而，一个只对自己已见过的数据表现良好的模型作用不大。真正的目的是*泛化*：即网络在新数据、未见过的数据上的表现如何。网络常常对训练数据学习得*过于*好，这是一种被称为过拟合的现象，会导致泛化能力不足。

本章侧重于诊断和解决这一难题。我们将介绍实用技术，以提高网络在训练中未曾遇到的数据上的泛化能力和可靠表现。

具体来说，你将学到：

*   识别过拟合及其对应情况——欠拟合。
*   有效使用验证集来监控训练过程中的泛化情况。
*   应用$L1$、$L2$和Dropout等正则化方法来限制模型的复杂程度。
*   实施提前停止，以避免网络训练时间过长。
*   调整超参数以优化模型表现的策略。
