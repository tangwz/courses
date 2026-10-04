---
course: "autoencoders-representation-learning"
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-7-applications-training-strategies"
sourceId: 656
chapter: "applications-training-strategies"
title: "应用与训练策略"
order: 7
description: "审视异常检测和预训练等应用场景，并结合优化自编码器训练的高级技术。"
hasQuiz: false
---

在确立了各类自编码器的架构及理论基础之后，从经典设计到变分模型和对抗模型，我们现在将侧重于它们在实践中的部署与完善。本章考察常见的应用场景，自编码器在这些场景中展现出独特优势，并讨论有效训练这些模型所需的方法。

您将研究自编码器如何应用于异常检测等任务，采用重构误差作为重要衡量标准。我们还将介绍它们在非线性降维、数据压缩中的应用，以及作为大型深度学习系统中无监督预训练的一种方式。此外，还将讨论图像去噪等应用。

成功运用自编码器通常需要细致的训练流程。因此，本章也涵盖适用于这些网络的先进优化策略、训练期间调整学习率以获得更好收敛的方法，以及超参数调整的系统化方法。掌握这些技术对于最小化所选损失函数（通常表示为 $L$）并达到实际中可靠的表现来说常常是必不可少的。我们将以实践例子作结，例如构建一个基于自编码器的异常检测系统。
