---
course: "introduction-to-deep-learning"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-6-regularization-performance-improvement"
sourceId: 945
chapter: "regularization-performance-improvement"
title: "正则化与性能提升"
order: 6
description: "了解过拟合、正则化（L1/L2，Dropout）、批量归一化和超参数调整入门。"
hasQuiz: true
---

在确定了构建和训练基本深度神经网络的过程后，我们现在关注实践中遇到的一个常见问题：过拟合。过拟合的模型会过度学习训练数据，包括其噪声和特定模式，导致在新数据上的表现不佳。

本章介绍旨在对抗过拟合的技术，并提高模型泛化能力的方法。我们将审视几种重要的正则化方法：

*   **L1 和 L2 正则化：** 了解如何根据权重的大小向损失函数添加惩罚项（$L_1 = \lambda \sum |w_i|$ 或 $L_2 = \lambda \sum w_i^2$）可以限制模型的复杂性。
*   **Dropout：** 学习在训练期间随机将一部分神经元激活设置为零，有助于阻止神经元之间形成复杂的相互依赖关系。
*   **提前停止：** 实施一种策略来监控验证表现，并在模型开始过拟合之前停止训练。
*   **批量归一化：** 研究如何在小批量中归一化激活，可以稳定并加速训练，通常会产生正则化效果。

最后，我们将介绍**超参数调整**的入门知识，涵盖网格搜索和随机搜索等基本策略，旨在帮助为学习率、网络大小或正则化强度等元素找到更好的配置。到本章结束时，您将掌握实用的工具，以构建更可靠、表现更好的深度学习模型。
