---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-tensorflow-core-concepts"
lesson: "tensorflow-to-pytorch-roadmap"
sourceId: 5736
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-1-pytorch-tensorflow-core-concepts/tensorflow-to-pytorch-roadmap"
title: "从 TensorFlow 到 PyTorch：开发者指南"
description: "了解 TensorFlow 开发人员学习 PyTorch 的理由和路径。"
order: 1
plots: []
sourceHash: "7af81b3e0693cd69cb178b12dab4f026ac5f96a7aad36254bdf926fe83bda4f2"
sourceCorrections: []
---

如果你有 TensorFlow 背景，那么你已经掌握了扎实的深度学习 (deep learning)基本原理。像张量、计算图、层、优化器以及整个模型训练周期等这些内容对你来说已经很熟悉。通过发挥你的现有知识，无需从零开始。一个主要目标是提供一个清晰的路径，帮你从 TensorFlow 的方法过渡到 PyTorch 的实践方式，有效地转化你的技能。

### 为什么选择 PyTorch？从迁移角度看

TensorFlow，特别是通过 Keras 的紧密结合以及默认采用即时执行，变得越来越易用且灵活。那么，为什么还要考虑将 PyTorch 加入你的工具集呢？

- **动态计算图：** PyTorch 的“定义时运行”方式，即计算图在操作执行时即时构建，提供了极高的灵活性。这对于具有动态架构的模型特别有利，例如自然语言处理 (NLP) 中常见的模型，它们的输入长度可能不同，或者图结构可能根据数据而改变。
- **Python 风格：** 许多开发人员发现 PyTorch 的 API 更符合传统的 Python 编程习惯。它与 Python 的生态系统（包括调试器和标准库）良好集成，让开发体验非常原生。
- **研究社区选用：** PyTorch 在研究社区中获得了广泛采纳，从而有大量前沿实现和预训练 (pre-training)模型可供使用。
- **明确控制：** 尽管 Keras 提供了像 `model.fit()` 这样的高级抽象，但 PyTorch 通常更鼓励明确的编程风格，特别是对于训练循环。这有助于你对工作原理有更透彻的了解，并在需要时提供更精细的控制。

本课程将通过持续与 TensorFlow 进行比较，帮助你理解这些方面。

### 你的学习路径：从 TensorFlow 到 PyTorch

可以将本课程视为一座桥梁。我们将从基本要素开始，逐步过渡到更复杂的应用，并且总是将所学与你在 TensorFlow 中可能完成类似任务的方式联系起来。

> 此图概述了学习进程，从你的 TensorFlow 经验开始，逐步掌握 PyTorch 的重要组成部分，以达到熟练程度。

以下是此路径涉及的概览：

1. **核心机制（本章）：** 我们将首先分析 TensorFlow 图执行（即使使用 `tf.function`）与 PyTorch 动态图之间的基本差异。你将看到 `tf.Tensor` 如何与 `torch.Tensor` 进行比较，以及 PyTorch 的 `autograd` 系统与 TensorFlow 的 `tf.GradientTape` 如何处理自动微分。
2. **构建模型：** 从 `tf.keras.Model` 和 `tf.keras.layers` 过渡到 PyTorch 的 `torch.nn.Module` 及其相关层。我们将了解如何定义模型架构，从简单的顺序堆叠到更复杂的自定义设计。
3. **数据处理：** 从 `tf.data` 管道转向 PyTorch 的 `torch.utils.data.Dataset` 和 `torch.utils.data.DataLoader`。你将学习使用 PyTorch 的工具创建高效的数据加载和预处理管道，包括用于图像数据的 `torchvision.transforms`。
4. **训练与评估：** 这是最显著的变化之一。尽管 TensorFlow 的 Keras API 提供了便捷的 `model.compile()` 和 `model.fit()` 方法，但 PyTorch 的开发通常涉及编写明确的训练和评估循环。我们将引导你构建这些循环，涵盖损失函数 (loss function)（`torch.nn`）、优化器（`torch.optim`）和指标计算。
5. **模型持久化：** 我们将介绍 PyTorch 中模型的保存和加载，并将 `state_dict` 与 TensorFlow 的 SavedModel 格式进行比较。我们还将涉及 TorchScript 用于模型序列化，并引入分布式训练和性能分析等高级主题。

### 预期收获

- **更“命令式”的风格：** PyTorch 代码通常感觉像标准的 Python 代码。操作在被调用时即执行，使用 Python 的原生工具进行调试更加直接。
- **明确的梯度控制：** 你会非常熟悉 `loss.backward()` 和 `optimizer.step()`。这种明确的控制，虽然需要更多的样板代码，但能让你清楚地理解反向传播 (backpropagation)过程。
- **模型定义的灵活性：** 通过继承 `nn.Module` 并实现 `forward` 方法来定义模型，为你提供了极大的灵活性，可以在模型执行中纳入任意 Python 代码和控制流。

本初始章节是你的起点。到本章结束时，你将牢固掌握 PyTorch 的基本构建块，以及它们与你从 TensorFlow 已有知识的关联。这将为后续章节奠定坚实底子，届时你将应用这些知识来构建、训练和管理 PyTorch 深度学习 (deep learning)模型。

## 参考资料

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Team (2024)
  Publisher: PyTorch Foundation
  PyTorch API和功能的官方综合指南，对理解核心概念和实际实现至关重要。
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf) — Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, Soumith Chintala (2019)
  Journal: Advances in Neural Information Processing Systems 32; Publisher: Curran Associates, Inc.; Volume: 32; Pages: 8024-8035; DOI: [10.48550/arXiv.1912.01703](https://doi.org/10.48550/arXiv.1912.01703)
  介绍了PyTorch的设计原则，包括其动态计算图和命令式编程模型。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  使用PyTorch构建深度学习模型的实用指南，涵盖核心功能和开发工作流程。
