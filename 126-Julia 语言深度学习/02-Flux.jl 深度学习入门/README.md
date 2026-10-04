# 第 2 章：Flux.jl 深度学习入门

来源：[原章节](https://apxml.com/zh/courses/julia-deep-learning/chapter-2-intro-flux-jl)

[返回课程目录](../README.md)

Julia机器学习的相关知识已讲解完毕，接下来我们将学习Flux.jl。这个库是Julia中构建神经网络的主要工具。它的设计注重灵活性，并善用Julia的特性以实现高效计算，使其成为深度学习项目的一个有力选项。

本章将带你了解Flux.jl的必要知识。你将学习如何使用它的主要组成部分来构建模型：层、链和激活函数。我们将讲解如何定义损失函数来衡量模型误差，以及优化器如何指导学习过程。Zygote.jl在为Flux提供自动微分方面起到的作用也将得到说明。本章最后将以构建和应用一个基本的神经网络作为总结，让你获得关于这些内容的实际操作经验。

## 小节

- 1. [Flux.jl：设计原则与架构](01-Flux.jl%EF%BC%9A%E8%AE%BE%E8%AE%A1%E5%8E%9F%E5%88%99%E4%B8%8E%E6%9E%B6%E6%9E%84.md)
- 2. [Flux.jl 基本构成：层、模型和链](02-Flux.jl%20%E5%9F%BA%E6%9C%AC%E6%9E%84%E6%88%90%EF%BC%9A%E5%B1%82%E3%80%81%E6%A8%A1%E5%9E%8B%E5%92%8C%E9%93%BE.md)
- 3. [定义简单的神经网络层](03-%E5%AE%9A%E4%B9%89%E7%AE%80%E5%8D%95%E7%9A%84%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%B1%82.md)
- 4. [在 Flux 中使用激活函数](04-%E5%9C%A8%20Flux%20%E4%B8%AD%E4%BD%BF%E7%94%A8%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.md)
- 5. [损失函数：衡量模型误差](05-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%9A%E8%A1%A1%E9%87%8F%E6%A8%A1%E5%9E%8B%E8%AF%AF%E5%B7%AE.md)
- 6. [优化器：引导学习过程](06-%E4%BC%98%E5%8C%96%E5%99%A8%EF%BC%9A%E5%BC%95%E5%AF%BC%E5%AD%A6%E4%B9%A0%E8%BF%87%E7%A8%8B.md)
- 7. [Zygote.jl：Flux中的自动微分](07-Zygote.jl%EF%BC%9AFlux%E4%B8%AD%E7%9A%84%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86.md)
- 8. [构建 Flux 中的基础神经网络](08-%E6%9E%84%E5%BB%BA%20Flux%20%E4%B8%AD%E7%9A%84%E5%9F%BA%E7%A1%80%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
- 9. [实践操作：使用 Flux 构建一个简单回归器](09-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E4%BD%BF%E7%94%A8%20Flux%20%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E5%9B%9E%E5%BD%92%E5%99%A8.md)
