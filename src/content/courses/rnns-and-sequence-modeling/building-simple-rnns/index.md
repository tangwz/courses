---
course: "rnns-and-sequence-modeling"
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-3-building-simple-rnns"
sourceId: 608
chapter: "building-simple-rnns"
title: "构建简单RNN"
order: 3
description: "学习使用Python和深度学习框架API实现基本的循环神经网络。"
hasQuiz: true
---

在了解了简单循环神经网络的主要思想和数学原理，包括时间反向传播（BPTT）的工作方式之后，我们现在将重点转向实现。本章将指导您完成使用常用工具构建第一个RNN模型的实际步骤。

您将学习如何：

*   配置您的Python环境，安装所需的深度学习库（TensorFlow或PyTorch）。
*   使用框架API来定义 `SimpleRNN` 层。
*   正确管理循环层所需的输入和输出张量形状，通常表示为 `(batch_size, time_steps, features)`。
*   将这些层组合成一个完整的序列模型。
*   构建基本的训练循环，根据损失函数来输入数据并更新模型权重。
*   将这些步骤应用于一个涉及简单序列预测的实践案例。

在本章结束时，您将把RNN的理论知识转化为可运行的代码，为您处理更复杂的架构和应用做好准备。
