# 第 4 章：使用 `torch.nn` 搭建模型

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-4-building-models-torch-nn)

[返回课程目录](../README.md)

在熟悉了PyTorch张量和用于梯度计算的Autograd系统后，我们现在开始构建神经网络本身。本章主要介绍`torch.nn`包，它是PyTorch用于高效构建网络结构的专用库。

你将学习如何使用核心`nn.Module`类作为模型的设计蓝图。我们将使用PyTorch提供的常用构建块来组装网络，包括线性层（`nn.Linear`）、卷积层（`nn.Conv2d`）和循环层（`nn.RNN`）。我们还将整合像激活函数（例如ReLU、Sigmoid）这样的重要组成部分，以引入非线性处理能力。此外，你将学习如何使用`torch.nn`中的损失函数（如$MSELoss$或$CrossEntropyLoss$）来定义目标衡量标准，以及如何从`torch.optim`中选择合适的优化算法（如SGD或Adam）在训练期间迭代地优化模型参数。在本章结束时，你将能够在PyTorch中定义并实例化自己的基本神经网络。

## 小节

- 1. [\`torch.nn.Module\` 基类](01-%60torch.nn.Module%60%20%E5%9F%BA%E7%B1%BB.md)
- 2. [定义自定义网络架构](02-%E5%AE%9A%E4%B9%89%E8%87%AA%E5%AE%9A%E4%B9%89%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84.md)
- 3. [常见层：线性、卷积、循环](03-%E5%B8%B8%E8%A7%81%E5%B1%82%EF%BC%9A%E7%BA%BF%E6%80%A7%E3%80%81%E5%8D%B7%E7%A7%AF%E3%80%81%E5%BE%AA%E7%8E%AF.md)
- 4. [激活函数 (ReLU, Sigmoid, Tanh)](04-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%20%28ReLU%2C%20Sigmoid%2C%20Tanh%29.md)
- 5. [简单模型的顺序容器](05-%E7%AE%80%E5%8D%95%E6%A8%A1%E5%9E%8B%E7%9A%84%E9%A1%BA%E5%BA%8F%E5%AE%B9%E5%99%A8.md)
- 6. [损失函数 (\`torch.nn\` 损失)](06-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%20%28%60torch.nn%60%20%E6%8D%9F%E5%A4%B1%29.md)
- 7. [优化器 (\`torch.optim\`)](07-%E4%BC%98%E5%8C%96%E5%99%A8%20%28%60torch.optim%60%29.md)
- 8. [练习：构建一个简单网络](08-%E7%BB%83%E4%B9%A0%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%BD%91%E7%BB%9C.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-4-building-models-torch-nn/quiz)
