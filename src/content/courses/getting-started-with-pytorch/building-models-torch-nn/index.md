---
course: "getting-started-with-pytorch"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-4-building-models-torch-nn"
sourceId: 557
chapter: "building-models-torch-nn"
title: "使用 `torch.nn` 搭建模型"
order: 4
description: "学习使用 `torch.nn` 在 PyTorch 中构建神经网络。内容涵盖模块、常用层、激活函数和损失函数。"
hasQuiz: true
---

在熟悉了PyTorch张量和用于梯度计算的Autograd系统后，我们现在开始构建神经网络本身。本章主要介绍`torch.nn`包，它是PyTorch用于高效构建网络结构的专用库。

你将学习如何使用核心`nn.Module`类作为模型的设计蓝图。我们将使用PyTorch提供的常用构建块来组装网络，包括线性层（`nn.Linear`）、卷积层（`nn.Conv2d`）和循环层（`nn.RNN`）。我们还将整合像激活函数（例如ReLU、Sigmoid）这样的重要组成部分，以引入非线性处理能力。此外，你将学习如何使用`torch.nn`中的损失函数（如$MSELoss$或$CrossEntropyLoss$）来定义目标衡量标准，以及如何从`torch.optim`中选择合适的优化算法（如SGD或Adam）在训练期间迭代地优化模型参数。在本章结束时，你将能够在PyTorch中定义并实例化自己的基本神经网络。
