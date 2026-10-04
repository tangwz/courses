---
course: "getting-started-with-tensorflow"
chapter: "tensorflow-fundamentals-setup"
lesson: "intro-tensorflow-2"
sourceId: 1672
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-1-tensorflow-fundamentals-setup/intro-tensorflow-2"
title: "TensorFlow 2.x 简介"
description: "了解 TensorFlow 2.x 在机器学习中的目的和架构。"
order: 1
plots: []
sourceHash: "ff25f12e13bdb81ad67c9af566eb18e7e299d502cbdf29867b3ab3babcb593b0"
sourceCorrections: []
---

TensorFlow 是一个由 Google Brain 团队开发的强大开源软件库，主要设计用于数值计算，尤其注重机器学习 (machine learning)和深度学习 (deep learning)应用。如果你想构建、训练和部署机器学习模型，特别是复杂的神经网络 (neural network)，TensorFlow 提供了一套全面的工具和资源来支持你的工作。

其核心是，TensorFlow 基于 **张量** 进行操作，张量是多维数组，将向量 (vector)（一维张量）和矩阵（二维张量）推广到更高维度。“TensorFlow” 这个名字本身就表明了这一点：计算涉及这些张量在操作图中流动。虽然图的思路在 TensorFlow 1.x 中非常明确，要求用户先定义整个计算图，然后在会话中执行它，但 TensorFlow 2.x 采用了更直接的方法。

## TensorFlow 2.x 的特点

TensorFlow 2.x 代表着一次重大重构，旨在与前身相比提高可用性和生产力。主要变化解决了常见痛点，并将最佳实践直接整合到库中：

1. **默认即时执行：** 这也许是最具影响力的变化。在 TensorFlow 2.x 中，操作在 Python 调用时立即执行。它们返回具体数值，而不是仅仅在需要单独执行的图中构建符号节点。这让 TensorFlow 用起来更像带有 NumPy 等库的标准 Python 编程。它极大简化了调试，并使得迭代和试验更直接。你可以打印张量值，使用标准 Python 控制流（`if`、`while`），并轻松查看中间结果。
2. **高层 API 集成 (Keras)：** Keras 是一个常用且用户友好的神经网络 (neural network)库，现在是 TensorFlow 的官方高层 API。`tf.keras` 提供直接的构建模块（层、模型、优化器、损失函数 (loss function)），显著减少了定义和训练模型所需的样板代码。我们将在后续章节中大量使用 Keras 提供的 Sequential 和 Functional API。这种紧密集成使得常见的深度学习 (deep learning)任务更容易上手。
3. **简化的 API 结构：** TensorFlow 2.x 精简了其 API 表面，移除了 TF 1.x 中许多冗余或令人困惑的函数（如 `tf.contrib`），并更合理地组织了其余功能。这使得代码更简洁、更易于维护。
4. **使用 `tf.function` 优化性能：** 尽管即时执行提高了可用性，但纯 Python 执行有时会比执行预编译计算图慢，尤其是对于复杂模型或在 GPU 或 TPU 等加速器上运行时。TensorFlow 2.x 提供了 `tf.function` 装饰器。将此装饰器应用于 Python 函数，TensorFlow 会自动将其编译为高度优化的计算图。这兼顾了两者的优点：开发和调试期间即时执行的便捷性，以及图执行在训练和部署时的性能优势。

## TensorFlow 的用途？

TensorFlow 用途广泛，尤其在深度学习 (deep learning)任务中表现出色：

- **图像识别与计算机视觉：** 构建卷积神经网络 (neural network)（CNN），用于图像分类、目标检测和图像分割等任务。
- **自然语言处理（NLP）：** 开发循环神经网络（RNN）、LSTM 和 Transformer 等模型，用于文本分类、机器翻译、情感分析和语言生成。
- **时间序列分析：** 根据历史数据预测未来值。
- **强化学习 (reinforcement learning)：** 训练智能体在环境中做出决策。
- **生成模型：** 使用生成对抗网络（GAN）或变分自编码器（VAE）等架构创建新的数据样本（例如图像、文本）。
- **科学计算：** 执行复杂的数值模拟和计算。

## 为什么选择 TensorFlow？

有几个因素促成了 TensorFlow 的受欢迎：

- **可扩展性：** 模型可以在单个 CPU、多个 GPU 或分布式机器集群上训练，包括 Google 张量处理单元（TPU）等专用硬件。
- **灵活性：** 尽管 Keras 提供了高层接口，TensorFlow 仍保持灵活性，可定义自定义层、损失函数 (loss function)、训练循环和复杂模型架构。
- **生产部署：** TensorFlow 提供在各种环境中部署模型的工具，包括服务器（TensorFlow Serving）、移动和嵌入 (embedding)式设备（TensorFlow Lite）以及网络浏览器（TensorFlow.js）。
- **生态系统与社区：** 它拥有庞大而活跃的社区、全面的文档、大量的教程、通过 TensorFlow Hub 可获得的预训练 (pre-training)模型，以及像 TensorBoard 这样的可视化工具（我们将在后面介绍）。

本章主要介绍如何正确设置环境。安装并验证后，你将准备好在下一章学习 TensorFlow 的基本组成部分，从张量的核心思路开始。

## 参考资料

- [TensorFlow Guide](https://www.tensorflow.org/guide) — TensorFlow Authors (2023)
  解释 TensorFlow 2.x 核心概念、即时执行、Keras 集成和 tf.function 的官方文档。
- [Keras: The Python Deep Learning API](https://keras.io) — Keras Team (2024)
  Keras 的官方文档，Keras 是集成到 TensorFlow 2.x 中的高级 API，涵盖其组件和使用方法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  机器学习和深度学习的实践指南，包含大量使用 TensorFlow 2.x 和 Keras 的示例。
