# 第 7 章：常用模型结构介绍

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-7-introduction-common-architectures)

[返回课程目录](../README.md)

您已经掌握了 PyTorch 的核心构成部分，比如张量（Tensors）、使用 Autograd 的自动求导、通过 `torch.nn` 定义模型，以及实现数据加载和训练步骤。本章将在之前所学知识之上，讲解如何构建特定且应用广泛的神经网络模型。

我们将着重介绍两种重要的模型类别：

1.  **卷积神经网络（CNNs）：** 您将了解卷积和池化的核心思想，学习为何 CNN 对网格状数据（特别是图像）表现出色，并使用 `nn.Conv2d` 和 `nn.MaxPool2d` 等层实现一个简单的 CNN 模型。我们还将说明如何处理这些层的输入和输出形状。
2.  **循环神经网络（RNNs）：** 您将接触到使用循环连接和隐藏状态处理序列数据的思路。我们将使用 `nn.RNN` 层构建一个简单的 RNN，并讨论 PyTorch 中序列输入所需的特定数据格式。还会简要提及更高级的变体，如 LSTM 和 GRU。

到本章结束时，您将能够在 PyTorch 中构建这些常用模型的简单版本，为您后续处理更复杂的模型做好准备。

## 小节

- 1. [卷积神经网络 (CNN) 概述](01-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20%28CNN%29%20%E6%A6%82%E8%BF%B0.md)
- 2. [在PyTorch中构建一个简单的CNN](02-%E5%9C%A8PyTorch%E4%B8%AD%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84CNN.md)
- 3. [理解CNN层的输入/输出形状](03-%E7%90%86%E8%A7%A3CNN%E5%B1%82%E7%9A%84%E8%BE%93%E5%85%A5-%E8%BE%93%E5%87%BA%E5%BD%A2%E7%8A%B6.md)
- 4. [循环神经网络 (RNN) 概述](04-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20%28RNN%29%20%E6%A6%82%E8%BF%B0.md)
- 5. [在PyTorch中构建一个简单的RNN](05-%E5%9C%A8PyTorch%E4%B8%AD%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84RNN.md)
- 6. [循环神经网络（RNN）的序列数据输入处理](06-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%88RNN%EF%BC%89%E7%9A%84%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E8%BE%93%E5%85%A5%E5%A4%84%E7%90%86.md)
- 7. [LSTM 和 GRU 简要介绍](07-LSTM%20%E5%92%8C%20GRU%20%E7%AE%80%E8%A6%81%E4%BB%8B%E7%BB%8D.md)
- 8. [实践：实现基本CNN和RNN](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%9F%BA%E6%9C%ACCNN%E5%92%8CRNN.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-7-introduction-common-architectures/quiz)
