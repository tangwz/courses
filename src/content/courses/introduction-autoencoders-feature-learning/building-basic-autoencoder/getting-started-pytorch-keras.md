---
course: "introduction-autoencoders-feature-learning"
chapter: "building-basic-autoencoder"
lesson: "getting-started-pytorch-keras"
sourceId: 6429
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-5-building-basic-autoencoder/getting-started-pytorch-keras"
title: "PyTorch和Keras入门"
description: "介绍如何使用TensorFlow及其Keras API来构建神经网络模型，如自编码器。"
order: 2
plots: []
sourceHash: "29a1b5cbbb2ed7f5b7fa9eafdc1ad32ae2b54ba94aef9007e7d770b9f6b3e283"
sourceCorrections: []
---

在已配置好深度学习 (deep learning) Python 环境的情况下，是时候熟悉我们将用于构建自编码器的主要工具了：PyTorch 和 Keras。这些框架是您构建和训练神经网络 (neural network)的主要工具集。

### 什么是 PyTorch？

PyTorch 是一个由 Facebook AI 研究院（FAIR）开发的开源机器学习 (machine learning)框架。PyTorch 的核心是为深度学习 (deep learning)应用而设计，并以其灵活性和易用性而闻名。它允许您定义、训练和部署机器学习模型，范围从简单的模型到非常复杂的神经网络 (neural network)。

设想您需要执行一项复杂的数学运算，特别是那种涉及许多变量，需要调整以达成目标的操作，就像我们的自编码器学习重建图像那样。PyTorch 提供了底层架构，可以高效地处理这些计算，尤其是在专用硬件如 GPU（图形处理单元）的帮助下（如果您有的话）。虽然 PyTorch 功能非常强大，能够处理精细的模型设计，但在这门入门课程中，我们将通过一个更易用的界面来使用它。

### 什么是 Keras？它与 PyTorch 有何关联？

Keras 是一个用于构建和训练深度学习 (deep learning)模型的高级 API（应用程序编程接口）。“高级”这个术语表示它被设计为用户友好且直观，让您无需陷入张量操作或复杂数学实现的低层细节，即可定义神经网络 (neural network)。

Keras 最初是一个独立库，可以与包括 PyTorch 在内的多种后端引擎配合使用。然而，随着 Keras 3 的发布，多后端支持已成为一个主要焦点。这意味着您可以使用相同的 Keras 代码在不同的框架（包括 PyTorch）上运行。

可以这样理解：PyTorch 是强大的引擎，可以为您的神经网络完成所有繁重工作。Keras 提供了一套简化的控制和蓝图（例如预制组件），使组装和运行该引擎变得更加容易。

> 该图说明了 Keras 如何作为强大的 PyTorch 引擎之上的用户友好界面，让您更轻松地定义自编码器模型。

对于我们的自编码器，Keras 将使我们能够：

- 以直接的方式定义网络层（输入、编码器隐藏层、瓶颈、解码器隐藏层、输出）。
- 通过指定损失函数 (loss function)（例如我们讨论过的 MSE）和优化器（辅助模型学习的算法）来编译模型。
- 用一个简单命令在我们的数据上训练模型。

### 为什么为您的第一个自编码器选择这些工具？

我们选择 PyTorch 和 Keras 有几个充分的理由：

1. **方便初学者使用**：特别是 Keras，降低了深度学习 (deep learning)的入门门槛。它的语法清晰，类似于您用普通语言描述网络结构的方式。
2. **行业标准**：PyTorch 是研究和产业方面中应用最广泛的深度学习框架之一。学习它能为您提供有价值的技能。
3. **丰富的生态系统**：有大量的文档、教程和一个庞大的社区，使得查找帮助和示例更加容易。
4. **集成体验**：由于 Keras 可以使用 PyTorch 作为其后端，它们可以顺畅协同工作。

### 在代码中准备它们

在上一节中，您应该已经将 PyTorch 和 Keras 作为环境设置的一部分进行安装。要在您的 Python 脚本或 Jupyter notebooks 中使用它们，您将首先设置 Keras 后端，然后导入必要的库。

设置 Keras 后端的标准方式是在导入 Keras 之前设置一个环境变量。

```python
import os
os.environ["KERAS_BACKEND"] = "torch"
```

然后您将导入 Keras 库。

由于 Keras 是一个多后端 API，您可以通过标准的 `keras` 导入来访问它。例如，当我们开始构建自编码器时，我们将像这样从 Keras 导入特定组件：

```python
from keras.layers import Input, Dense
from keras.models import Model
```

这里，`Input` 和 `Dense` 是我们将要使用的层类型，而 `Model` 是我们用来定义自编码器整体结构的工具。现在不必过多担心这些特定的导入；我们将在后续的“构建简单自编码器模型”一节中，在构建模型时详细说明它们。

总结一下，PyTorch 提供了强大的后端，而 Keras 则提供了一种方便的方式来定义和训练我们的神经网络 (neural network)，包括我们即将构建的自编码器。有了这些工具，您就准备充分，可以将我们学到的自编码器原理转化为可运行的代码了。

接下来，我们将了解如何加载和准备数据集，这将是我们的自编码器进行学习的原始数据。

## 参考资料

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Core Team (2023)
  Publisher: PyTorch Foundation
  PyTorch深度学习框架的官方指南，定期更新，涵盖其核心功能和API。
- [Keras Documentation](https://keras.io/) — Keras Team (2023)
  Keras的官方文档，提供了构建神经网络和配置其多后端支持（包括PyTorch）的详细指南。
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://proceedings.neurips.cc/paper/2019/file/bfad02148a6a575b6510303867c51483-Paper.pdf) — Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, Soumith Chintala (2019)
  Journal: Advances in Neural Information Processing Systems 32; Publisher: Curran Associates, Inc.; Pages: 8024–8035; DOI: [10.5555/3454287.3455008](https://doi.org/10.5555/3454287.3455008)
  本文介绍了PyTorch框架，概述了其设计原则，如命令式编程和动态计算图，及其对深度学习研究的贡献。
- [Deep Learning with Python (2nd Edition)](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  Keras创建者撰写的深度学习综合指南，侧重于使用Keras API进行实际应用和最佳实践。
