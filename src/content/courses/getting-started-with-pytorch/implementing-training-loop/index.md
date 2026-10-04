---
course: "getting-started-with-pytorch"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-6-implementing-training-loop"
sourceId: 561
chapter: "implementing-training-loop"
title: "实现训练循环"
order: 6
description: "实现完整的 PyTorch 训练循环：前向传播、损失计算、反向传播、优化器步骤和模型评估。"
hasQuiz: true
---

你已经学习了如何使用 `torch.nn` 定义模型，以及如何高效地使用 `Dataset` 和 `DataLoader` 准备数据。本章将侧重于整合这些组成部分，以实际训练你的神经网络。我们将构建被称为训练循环的标准迭代过程。

你将学习涉及的重要步骤：

*   设置模型、损失函数和优化器。
*   迭代 `DataLoader` 提供的数据批次。
*   执行**前向传播**：将输入数据送入模型以获得预测结果。
*   计算**损失**：使用所选标准来衡量模型预测与真实标签之间的差异，通常表示为 $L$。
*   执行**反向传播**：通过调用 `$loss.backward()` 计算损失相对于模型参数 ($\nabla_{\theta} L$) 的梯度。
*   更新**模型权重**：使用优化器根据计算出的梯度调整模型参数 ($\theta$)，通常通过 `$optimizer.step()`。
*   在下一次迭代前，使用 `$optimizer.zero_grad()` 将梯度归零。

此外，我们将介绍如何实现一个独立的循环来评估模型在验证集或测试集上的表现，以及在训练期间或之后保存和加载模型状态（检查点）的方法。到本章结束时，你将能够为你的 PyTorch 模型实现一个完整的训练和评估流程。
