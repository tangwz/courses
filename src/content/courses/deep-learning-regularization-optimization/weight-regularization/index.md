---
course: "deep-learning-regularization-optimization"
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-2-weight-regularization"
sourceId: 927
chapter: "weight-regularization"
title: "权重正则化方法"
order: 2
description: "理解并实现 L1、L2（权重衰减）和弹性网络正则化，通过惩罚大权重来防止过拟合。"
hasQuiz: true
---

在确认过拟合是模型在训练数据上学习得过好，但对未见过的数据性能下降的普遍问题后，我们现在转向一种直接限制模型复杂度的方法：权重正则化。本章介绍通过根据网络权重的大小添加惩罚项来修改学习目标的方法。主要观点是，更简单的模型（通常表现为权重值较小）往往泛化能力更好。

你将学习到：

*   惩罚大权重的直观解释及其对模型复杂度的影响。
*   **L2 正则化（权重衰减）**：如何通过添加与权重平方大小（$L_2$ 范数）成比例的惩罚来促使权重变小，以及其数学表述。
*   **L1 正则化**：如何通过添加与权重绝对大小（$L_1$ 范数）成比例的惩罚项，以及它如何能够产生稀疏权重向量（特征选择）。
*   L1 与 L2 正则化在效果和适用情况上的比较。
*   **弹性网络正则化**：一种结合 L1 和 L2 惩罚项的方法。
*   如何在常见的深度学习框架中通过代码示例实际实现这些方法。

在本章结束时，你将学会如何应用这些方法来减少过拟合并提升神经网络的泛化能力。
