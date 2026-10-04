---
course: "getting-started-with-pytorch"
chapter: "implementing-training-loop"
lesson: "anatomy-training-loop"
sourceId: 2186
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-6-implementing-training-loop/anatomy-training-loop"
title: "训练循环的构成"
description: "概述训练神经网络所包含的标准步骤。"
order: 1
plots: []
sourceHash: "2d01a64f1c1d4c0649dcda20f5a3d82576267fec7ad5ba33d1b140b276e10c75"
sourceCorrections: []
---

训练神经网络 (neural network)是一个迭代优化过程。你为模型提供数据，衡量其预测的准确度，然后微调 (fine-tuning)其内部参数 (parameter)（权重 (weight)和偏置 (bias)），以减少这种不准确性。这个循环会重复多次。管理这种重复过程的代码结构通常被称为**训练循环**。

### 整体结构：周期与批次

从宏观上看，训练通常包含两个嵌套循环：

1. **外层循环（周期）：** 一个周期表示对整个训练数据集的一次完整遍历。训练通常跨越多个周期，让模型能够多次查看并从每个数据样本中学习。周期的数量是你根据模型表现何时稳定或停止提升来选择的一个超参数 (parameter) (hyperparameter)。
2. **内层循环（批次）：** 由于内存限制，一次性处理整个数据集在计算上通常是不可行的。因此，在每个周期内，我们以称为批次的更小片段遍历数据集。你之前学过的`DataLoader`负责提供这些批次。与逐个处理样本或一次性使用整个数据集相比，以批次进行训练内存高效，还能带来更稳定的收敛和更好的泛化能力。

### 每次批次迭代中的核心步骤

对于在一个周期内处理的每个批次，训练循环执行一系列明确定义的步骤。让我们分解一下典型迭代中发生的情况：

1. **获取数据：** 从`DataLoader`获取下一批输入数据（特征）及其对应的目标标签。在此阶段，确保数据被传输到模型参数 (parameter)所在的正确计算设备（CPU或GPU）也很重要。
2. **梯度清零：** 在计算当前批次的梯度之前，你必须明确重置从*上一*次迭代积累的梯度。如果忘记这一步，梯度将在批次间累加，导致不正确的更新，并可能在训练期间发散。通过在优化器对象上调用`zero_grad()`方法来完成。

   ```python
   # 代码：在新批次处理前重置梯度
   optimizer.zero_grad()
   ```
3. **前向传播：** 将输入特征批次送入你的模型。模型通过其层处理数据，应用学习到的权重 (weight)和激活函数 (activation function)，最终生成一批预测或输出。

   ```python
   # 代码：获取模型预测
   predictions = model(input_batch)
   ```
4. **计算损失：** 使用你选择的损失函数 (loss function)（准则）将模型的`predictions`与真实的`target_batch`进行比较，例如用于分类的`nn.CrossEntropyLoss`或用于回归的`nn.MSELoss`。损失函数返回一个单一的标量值，表示当前批次的平均误差或差异。这个值显示了模型在这个特定批次上的表现好坏。

   ```python
   # 代码：计算损失
   loss = criterion(predictions, target_batch)
   ```
5. **反向传播 (backpropagation)：** 这是PyTorch的自动微分引擎Autograd计算梯度的地方。调用`loss.backward()`计算损失标量相对于每个`requires_grad=True`的模型参数的梯度（`nn.Module`中参数的默认设置）。这些梯度表示损失对每个参数变化的敏感度；本质上，它们告诉优化器如何调整每个权重以降低损失。

   ```python
   # 代码：通过反向传播计算梯度
   loss.backward()
   ```
6. **更新权重（优化器步进）：** 计算出梯度后，优化器现在可以调整模型的参数了。调用`optimizer.step()`根据计算出的梯度和优化器的特定算法（如带动量的SGD、Adam等）更新每个参数。目标是朝着最小化损失的方向迈出一小步。

   ```python
   # 代码：更新模型参数
   optimizer.step()
   ```

训练循环中的一次迭代是模型处理数据并更新参数的完整单元。这个循环为`DataLoader`提供的每个批次重复进行。一旦所有批次处理完毕，一个周期就完成了，外层循环开始下一个周期，重复整个批次迭代过程。

> 流程图显示了PyTorch训练循环单次批次迭代中的操作顺序。

## 参考资料

- [PyTorch Quickstart Tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) — PyTorch Core Team (2025)
  解释了实现基本训练循环的PyTorch API实用方法，包括`DataLoader`、优化器、损失函数、`zero_grad()`、`backward()`和`step()`。这是一个基础指南。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供了神经网络优化、反向传播、损失函数和通用训练方法的理论背景，这些构成了训练循环的基础。
- [Dive into Deep Learning](https://d2l.ai/) — Aston Zhang, Zachary C. Lipton, Mu Li, and Alex Smola (2024)
  Publisher: Cambridge University Press
  一本交互式开源书籍，涵盖深度学习基础知识，包括带有周期和批次的训练过程，并提供了PyTorch代码示例。
