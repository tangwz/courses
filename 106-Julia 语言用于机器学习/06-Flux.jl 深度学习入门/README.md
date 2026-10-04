# 第 6 章：Flux.jl 深度学习入门

来源：[原章节](https://apxml.com/zh/courses/julia-for-machine-learning/chapter-6-julia-deep-learning-flux-jl)

[返回课程目录](../README.md)

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

## 小节

- 1. [神经网络基本原理](01-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86.md)
- 2. [Flux.jl 入门：张量与层](02-Flux.jl%20%E5%85%A5%E9%97%A8%EF%BC%9A%E5%BC%A0%E9%87%8F%E4%B8%8E%E5%B1%82.md)
- 3. [构建前馈神经网络](03-%E6%9E%84%E5%BB%BA%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
- 4. [定义损失函数与优化器](04-%E5%AE%9A%E4%B9%89%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E4%B8%8E%E4%BC%98%E5%8C%96%E5%99%A8.md)
- 5. [在 Flux.jl 中训练神经网络](05-%E5%9C%A8%20Flux.jl%20%E4%B8%AD%E8%AE%AD%E7%BB%83%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
- 6. [使用 Zygote.jl 的自动微分](06-%E4%BD%BF%E7%94%A8%20Zygote.jl%20%E7%9A%84%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86.md)
- 7. [使用梯度](07-%E4%BD%BF%E7%94%A8%E6%A2%AF%E5%BA%A6.md)
- 8. [动手实践：搭建和训练一个简单的神经网络](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%90%AD%E5%BB%BA%E5%92%8C%E8%AE%AD%E7%BB%83%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
