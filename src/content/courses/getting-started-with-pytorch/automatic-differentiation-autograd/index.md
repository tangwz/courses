---
course: "getting-started-with-pytorch"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-3-automatic-differentiation-autograd"
sourceId: 554
chapter: "automatic-differentiation-autograd"
title: "Autograd 自动微分"
order: 3
description: "了解 PyTorch Autograd 的工作方式。学习计算图、梯度计算以及如何控制梯度流动。"
hasQuiz: true
---

有效地训练神经网络需要调整模型参数以最小化损失函数，这通常使用梯度下降或其变体。这个过程的**主要部分**是计算损失函数相对于每个参数的梯度，数学上表示为损失 $L$ 和参数 $w$ 的 $\frac{\partial L}{\partial w}$。对于复杂模型，手动推导和实现这些梯度计算是不切实际的。

本章介绍 PyTorch 的自动微分引擎 Autograd，它旨在自动计算这些梯度。我们将**了解** PyTorch 如何在对张量执行操作时动态构建计算图。你将学习如何使用 `requires_grad=True` 标记张量以进行梯度计算，使用 `.backward()` 触发反向传播以计算梯度，以及查看存储在 `.grad` 属性中的结果梯度。我们还将涉及梯度累积，使用 `optimizer.zero_grad()` 清零梯度的**必要性**，以及如何使用 `torch.no_grad()` 等上下文临时禁用梯度计算，以提高推理或评估期间的效率。
