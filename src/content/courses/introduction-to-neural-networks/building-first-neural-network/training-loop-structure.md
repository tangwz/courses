---
course: "introduction-to-neural-networks"
chapter: "building-first-neural-network"
lesson: "training-loop-structure"
sourceId: 1986
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-5-building-first-neural-network/training-loop-structure"
title: "训练循环的结构"
description: "概述通过数据迭代来训练网络的步骤。"
order: 3
plots: []
sourceHash: "1f28534b5ec496b55956a13a3a30f24f7a1e2ad6ab5014f0fb29a3e2550ea1b2"
sourceCorrections: []
---

训练神经网络 (neural network)并非一次性计算。相反，它是一个迭代过程，网络会逐步从训练数据中学习。这种迭代优化发生在通常称为**训练循环**的过程中。理解这种循环的构成方式对掌握模型学习的过程非常重要。

其核心是，训练循环重复执行一系列步骤，旨在调整网络的参数 (parameter)——权重 (weight) $W$ 和偏置 (bias) $b$，使预测值随时间推移更接近实际目标值。可以将其视为一种系统化的方式，向网络展示样本、评估其表现，并指导它如何调整以在下次做得更好。

典型的训练循环涉及多次遍历数据集。每次完整地遍历整个训练数据集被称为一个**周期**（**epoch**）。在每个周期内，数据通常分小块处理，这些小块称为**批次**（**batches**）或**小批次**（**mini-batches**）。

以下是对训练循环单次迭代中执行的主要步骤的细分，通常以一个数据批次为单位进行操作：

1. **获取批次数据：** 选取一小部分（批次）训练数据（输入 $X_{batch}$ 和相应的目标输出 $Y_{batch}$）。使用批次而不是一次性处理整个数据集，可以使计算更易管理，特别是对于大型数据集，并且由于更新的随机性，通常会带来更快的收敛速度和更好的泛化能力。
2. **前向传播：** 将输入数据（$X_{batch}$）逐层送入网络，应用线性变换（使用当前 $W$ 和 $b$）和激活函数 (activation function)，以生成预测值（$\hat{Y}_{batch}$）。这是第三章中详细说明的过程。
3. **计算损失：** 计算网络预测值（$\hat{Y}_{batch}$）与实际目标值（$Y_{batch}$）之间的差异。这会使用选定的损失函数 (loss function) $L$，例如用于回归的均方误差 (MSE) 或用于分类的交叉熵损失，如第四章所述。损失值衡量了网络对当前批次数据的预测有多么不准确。
4. **反向传播 (backpropagation) (Backprop)：** 计算损失函数相对于网络中每个参数的梯度（例如，所有权重和偏置的 $\frac{\partial L}{\partial W}$ 和 $\frac{\partial L}{\partial b}$）。此过程在第四章中也有介绍，它使用链式法则来确定每个参数对总体损失的贡献程度。这些梯度指出了损失函数上升最快的方向。
5. **更新参数：** 调整网络的权重和偏置以最小化损失。这通常使用优化算法完成，例如梯度下降 (gradient descent)或其变体（例如 Adam, RMSprop）。参数会沿着梯度的反方向更新，更新幅度由学习率（$\eta$）控制：
   $W = W - \eta \frac{\partial L}{\partial W}$
   $b = b - \eta \frac{\partial L}{\partial b}$
   此步骤会推动参数朝向在下一次迭代中减少损失的方向移动。

这五个步骤构成一次**训练迭代**或**训练步**。循环会在一个周期内为所有批次重复这些步骤，然后为多个周期重复整个过程。

整体结构可按如下方式展现：

> 典型的神经网络训练循环结构，遍历周期并以批次处理数据。每个批次都经过前向传播、损失计算、反向传播和参数更新。

在训练期间，需要监控网络表现，通常通过跟踪训练集和单独验证集上的损失和准确率（针对分类任务）来完成，这可以在每个周期之后，甚至在一定数量的迭代之后进行。这有助于诊断过拟合 (overfitting)或欠拟合 (underfitting)等问题，并决定何时停止训练（将在第六章中进一步讨论）。

虽然像TensorFlow和PyTorch这样的深度学习 (deep learning)框架提供了高级抽象，通常向用户隐藏了明确的循环结构，但理解这种基本流程对于调试、自定义训练过程以及解释模型行为非常重要。接下来的章节将引导你实现这些步骤，以构建和训练你的第一个网络。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的综合性教科书，详细学术探讨了神经网络架构、反向传播等训练算法和优化方法。
- [CS231n: Convolutional Neural Networks for Visual Recognition, Lecture Notes](https://cs231n.github.io/) — Fei-Fei Li, Justin Johnson, and Serena Yeung (2024)
  Journal: Stanford University; Publisher: Stanford University
  为学生提供了清晰详细的神经网络训练核心组件解释，包括前向和反向传播、损失函数和优化，以教学方式呈现。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael Nielsen (2015)
  Publisher: Determination Press
  一本易于理解的在线书籍，从第一性原理构建神经网络概念，直观地解释了反向传播和迭代学习过程。
