---
course: "julia-for-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-6-julia-deep-learning-flux-jl"
sourceId: 1212
chapter: "julia-deep-learning-flux-jl"
title: "Flux.jl 深度学习入门"
order: 6
description: "在 Julia 中使用 Flux.jl 学习深度学习。构建、训练并理解神经网络和自动微分。"
hasQuiz: false
---

本章将介绍深度学习的原理以及如何使用 Julia 的主要深度学习库 Flux.jl 来实现它们。在您对机器学习的认识基础上，我们将侧重于神经网络。这类模型在图像识别和自然语言处理等技术中推动了显著进步。

您将学到：
*   回顾神经网络的基本组成部分，例如层、Sigmoid 激活函数 $\sigma(x) = \frac{1}{1 + e^{-x}}$，以及常见结构。
*   上手使用 Flux.jl，了解其核心数据结构，例如张量，以及如何定义网络层。
*   构建前馈神经网络。
*   定义恰当的损失函数，例如均方误差 $$L = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$，并选择用于训练的优化器。
*   理解训练循环，包括前向传播、反向传播和权重更新。
*   学习 Zygote.jl 自动微分，这是一个让 Flux.jl 中基于梯度的优化得以实现的组成部分。
*   在模型训练过程中处理梯度。

在本章结束时，您将能够使用 Flux.jl 在 Julia 中构建和训练基础的神经网络模型。
