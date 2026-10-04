---
course: "getting-started-with-pytorch"
chapter: "pytorch-fundamentals-setup"
lesson: "what-is-pytorch"
sourceId: 2080
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-1-pytorch-fundamentals-setup/what-is-pytorch"
title: "PyTorch 是什么？"
description: "An overview of PyTorch and its role in the deep learning ecosystem."
order: 1
plots: []
sourceHash: "6108d47cb3083abdf7617ae7d2dc65228da2dfc6d7e7d2f59801e18da9fe3999"
sourceCorrections: []
---

PyTorch 是一个主要由 Meta AI 开发的开源机器学习 (machine learning)库。它在研究和工业界都受到了广泛欢迎，用于构建和训练深度学习 (deep learning)模型。它是一个强大的工具集，专门为满足现代机器学习应用的需求而设计。

其核心功能在于，PyTorch 提供了两个使其非常有效的基本功能：

1. **张量计算：** 与 NumPy 数组类似，PyTorch 提供了称为张量的多维数组。然而，PyTorch 张量带有一个主要优势：它们可以方便地在图形处理器（GPU）或其他专用硬件加速器上处理。这种能力显著加快了训练大型神经网络 (neural network)所需的数值计算，与仅使用 CPU 执行相比，通常能将速度提高几个数量级。如果你使用过 NumPy，你会发现操作这些张量的 API 会让你感到熟悉，从而简化了转换过程。
2. **自动微分：** 训练神经网络涉及根据损失函数 (loss function)的梯度调整模型参数 (parameter)。手动计算这些梯度既复杂又容易出错，特别是对于深层架构。PyTorch 包含一个复杂的自动微分引擎，称为 `Autograd`。当对张量执行操作时，`Autograd` 会动态构建一个计算图。这个图会记录操作序列，允许 PyTorch 在需要时（通常通过调用 `.backward()`）使用链式法则自动计算梯度。与需要预先定义静态图的框架相比，这种动态特性在模型设计上提供了相当大的灵活性。

### 为何选用 PyTorch？

有几个因素促成了 PyTorch 的广泛使用：

- **Python 风格的接口：** PyTorch 与 Python 数据科学环境紧密结合。其 API 设计直观自然，对于 Python 开发者来说相对容易学习和使用。调试通常感觉就像调试标准的 Python 代码。
- **灵活性：** 动态计算图（由运行定义）意味着网络的结构可以在执行期间改变。这对某些类型的模型特别有用，如循环神经网络 (neural network) (RNN)，其中序列长度可能不同，或者在模型内部实现复杂的控制流。
- **丰富的生态体系：** PyTorch 得益于一套丰富的支持库和工具。像 `torchvision`、`torchaudio` 和 `torchtext` 这样的库针对特定应用领域（分别是计算机视觉、音频和自然语言处理）提供预构建的数据集、模型架构和数据转换。与 TensorBoard 等可视化工具的集成进一步提升了开发流程。
- **研究与生产：** 虽然最初因其灵活性而在研究社区中广受欢迎，PyTorch 已大幅成熟，现在包含了 TorchServe 和 TorchScript 等工具，使其成为将模型部署到生产环境的可行选择。

本章将带你开始了解 PyTorch，通过关注第一个基本功能：张量。你将学习如何创建它们、操作它们、执行基本运算，并理解它们与 NumPy 数组的关系。掌握这些基础元素是第一步，以使用 PyTorch 的强大功能构建和训练复杂的深度学习 (deep learning)模型。

## 参考资料

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Team (2024)
  Publisher: PyTorch Foundation
  这是PyTorch的主要且全面的资源，涵盖其张量和自动求导等核心功能以及API细节。
- [Dive into Deep Learning](https://d2l.ai/) — Aston Zhang, Zachary C. Lipton, Mu Li, and Alex Smola (2023)
  Publisher: Cambridge University Press
  这本互动教科书介绍深度学习概念，并使用PyTorch实现，为张量、神经网络和训练提供实践示例。
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) — Fei-Fei Li, Ehsan Adeli, Justin Johnson, Zane Durante (2025)
  Publisher: Stanford University
  本课程提供对神经网络的详细理解，常使用PyTorch进行实践练习，涵盖张量和反向传播等基本概念。
