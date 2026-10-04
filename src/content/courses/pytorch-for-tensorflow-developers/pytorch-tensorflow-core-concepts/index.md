---
course: "pytorch-for-tensorflow-developers"
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-1-pytorch-tensorflow-core-concepts"
sourceId: 1043
chapter: "pytorch-tensorflow-core-concepts"
title: "衔接 TensorFlow 与 PyTorch：核心要点"
order: 1
description: "比较 TensorFlow 与 PyTorch 的基础要点。学习面向 TensorFlow 开发者 的 PyTorch 张量、动态图和自动微分。"
hasQuiz: true
---

本章对 TensorFlow 和 PyTorch 进行基础对比，侧重它们的核心运作方式。如果您已熟悉 TensorFlow，本章将帮助您把现有知识对应到 PyTorch 环境中。

您将学会：
*   区分 TensorFlow 的静态计算图与 PyTorch 动态的“定义即运行”方式。
*   比较 `tf.Tensor` 和 `torch.Tensor`，包括它们的创建、属性和常用操作。
*   了解 PyTorch 的自动微分机制 `autograd` 与 TensorFlow 的 `tf.GradientTape` 之间的关系。
*   查看两个框架如何与 NumPy 集成以实现高效数据处理。
*   在 PyTorch 中管理跨不同设备（例如 CPU 和 GPU）的计算。

我们将涵盖主要异同点，为 PyTorch 的使用打好底子。本章最后将提供实践练习，以应用这些核心要点。
