---
course: "getting-started-with-pytorch"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-7-introduction-common-architectures"
sourceId: 563
chapter: "introduction-common-architectures"
title: "常用模型结构介绍"
order: 7
description: "开始在 PyTorch 中构建常用深度学习模型结构，比如卷积神经网络（CNNs）和循环神经网络（RNNs）。"
hasQuiz: true
---

您已经掌握了 PyTorch 的核心构成部分，比如张量（Tensors）、使用 Autograd 的自动求导、通过 `torch.nn` 定义模型，以及实现数据加载和训练步骤。本章将在之前所学知识之上，讲解如何构建特定且应用广泛的神经网络模型。

我们将着重介绍两种重要的模型类别：

1.  **卷积神经网络（CNNs）：** 您将了解卷积和池化的核心思想，学习为何 CNN 对网格状数据（特别是图像）表现出色，并使用 `nn.Conv2d` 和 `nn.MaxPool2d` 等层实现一个简单的 CNN 模型。我们还将说明如何处理这些层的输入和输出形状。
2.  **循环神经网络（RNNs）：** 您将接触到使用循环连接和隐藏状态处理序列数据的思路。我们将使用 `nn.RNN` 层构建一个简单的 RNN，并讨论 PyTorch 中序列输入所需的特定数据格式。还会简要提及更高级的变体，如 LSTM 和 GRU。

到本章结束时，您将能够在 PyTorch 中构建这些常用模型的简单版本，为您后续处理更复杂的模型做好准备。
