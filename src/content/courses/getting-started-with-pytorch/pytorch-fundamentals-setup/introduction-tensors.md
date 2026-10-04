---
course: "getting-started-with-pytorch"
chapter: "pytorch-fundamentals-setup"
lesson: "introduction-tensors"
sourceId: 2085
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-1-pytorch-fundamentals-setup/introduction-tensors"
title: "张量介绍"
description: "了解 PyTorch 中主要的数据结构：张量。"
order: 3
plots: []
sourceHash: "e3c03339fe9125a0f7ba5bb41e23b92fdc96306f0e19dd0019e21aafbb8a3a23"
sourceCorrections: []
---

\*\*张量（Tensor）\*\*是 PyTorch 中将要使用的主要数据结构。如果你使用过 NumPy，你会发现 PyTorch 张量非常熟悉。从本质上讲，张量是多维数组，与 NumPy 的 `ndarray` 非常相似。

可以把张量看作是常见数学对象的推广：

- 一个**标量**（单个数字，例如 7）是 0 维张量（或 0 阶张量）。
- 一个**向量 (vector)**（一维数字列表或数组，例如 `[1, 2, 3]`）是 1 维张量（或 1 阶张量）。
- 一个**矩阵**（二维数字网格，例如 `[[1, 2], [3, 4]]`）是 2 维张量（或 2 阶张量）。
- 依此类推，可以有 3 维张量（例如数字立方体，常用于 RGB 图像：高 x 宽 x 通道）、4 维张量（常用于图像批次：批大小 x 高 x 宽 x 通道）等。

> 一种将张量看作标量、向量和矩阵的推广的方式，维度逐渐增加。

在深度学习 (deep learning)中，张量用来表示几乎所有数据：

- **输入数据：** 图像批次、文本序列或特征表格。
- **模型参数 (parameter)：** 神经网络 (neural network)层的权重 (weight)和偏置 (bias)。
- **中间激活：** 网络内部各层的输出。
- **梯度：** 反向传播 (backpropagation)过程中计算的值，用于更新模型参数。

与标准 Python 列表甚至 NumPy 数组相比，是什么让 PyTorch 张量特别适合深度学习呢？

1. **GPU 加速：** 张量可以轻松地移动到图形处理器 (GPU) 或其他硬件加速器上进行处理。这为深度学习中常见的计算密集型操作提供了大幅加速。
2. **自动微分：** PyTorch 张量通过 `Autograd` 系统内置了对自动微分的支持（我们将在第 3 章中介绍）。这种机制自动计算梯度，这对通过反向传播训练神经网络来说非常重要。

尽管其原理与 NumPy 数组相似，但这两个特性使得 PyTorch 张量成为高效构建和训练模型的主要工具。在接下来的部分中，我们将介绍如何创建和操作这些重要的数据结构。

## 参考资料

- [PyTorch Tensors](https://pytorch.org/docs/stable/tensors.html) — PyTorch developers (2024)
  Publisher: PyTorch Foundation
  PyTorch核心数据结构的官方文档，涵盖其定义、属性和基本操作。
- [Automatic differentiation with torch.autograd](https://pytorch.org/docs/stable/autograd.html) — PyTorch developers (2016)
  Publisher: PyTorch Foundation
  PyTorch自动微分引擎的官方文档，对神经网络训练至关重要。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 28-34
  一本基础教材，在深度学习背景下介绍张量作为数学对象，包括标量、向量和矩阵。
- [Neural Networks Part 1: Setting up the Architecture](https://cs231n.github.io/neural-networks-1/#datarep) — Andrej Karpathy, Justin Johnson, Serena Yeung, et al. (Stanford CS231n course staff) (2023)
  Journal: Stanford CS231n Course Notes
  斯坦福大学知名课程笔记，提供在深度学习中将张量作为数据表示的实践介绍，特别针对图像处理。
