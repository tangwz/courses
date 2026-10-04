---
course: "pytorch-for-tensorflow-developers"
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-4-pytorch-training-loops-for-keras-devs"
sourceId: 1046
chapter: "pytorch-training-loops-for-keras-devs"
title: "训练与评估：Keras 方法到 PyTorch 循环的对应"
order: 4
description: "学习编写自定义的 PyTorch 训练和评估循环，并与 TensorFlow Keras 的 `fit` 和 `compile` 方法进行比较。"
hasQuiz: true
---

在学习了如何通过与 Keras 对比来定义 PyTorch 中的模型架构之后，我们现在转向这些模型的训练和评估过程。如果您习惯使用 TensorFlow Keras，您可能已经使用过 `model.compile()` 来指定优化器和损失函数，然后使用 `model.fit()` 来管理训练迭代。PyTorch 采用一种更清晰的方法，您需要从基础组成部分构建训练循环。这种方式提供了更大的灵活性，并让您对整个训练过程有更清晰的认识。

本章将指导您在 PyTorch 中构建这些自定义的训练和评估循环。我们将考察损失函数（来自 `torch.nn` 或 `torch.nn.functional`）和优化算法（在 `torch.optim` 包中）是如何实现和使用的，并将它们与 TensorFlow Keras 的对应部分进行比较。您将学习标准的 PyTorch 训练流程：执行前向传播以获取预测，计算损失，通过 `loss.backward()` 计算梯度，以及使用 `optimizer.step()` 更新模型参数。此外，我们还将讨论如何组织模型评估，有效地追踪性能指标，并引入类似于 Keras Callbacks 的训练控制机制。
