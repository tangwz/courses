---
course: "introduction-to-deep-learning"
chapter: "activation-functions-architecture"
lesson: "role-of-activation-functions"
sourceId: 5025
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/role-of-activation-functions"
title: "激活函数的作用"
description: "解释非线性激活函数对深度学习模型的必要性。"
order: 1
plots: ["plots/5025-0.json"]
sourceHash: "97e0df29447a9c30e11469e3baa6394f379b0ac107dcc85e37675561cb0e9db0"
sourceCorrections: []
---

多层感知机 (MLP) 通过堆叠人工神经元层来获得功能。每个神经元计算其输入的加权和并加上偏置 (bias)项。如果仅仅将这个和直接传递给下一层而不进行额外处理，那么我们的深度网络并不会比单层模型的功能强多少。

这是为什么呢？设想一个双层网络，其中每层只进行线性变换（加权和加偏置）。第一层的输出将是输入的线性函数。第二层的输出将是第一层输出的线性函数。组合两个线性函数，结果仍是一个线性函数。无论我们堆叠多少个线性层，整个网络都将表现得像一个单一的线性变换。它无法对数据中的非线性模式进行建模，就像我们之前遇到的异或问题。

正是在这里，**激活函数 (activation function)**发挥了作用。激活函数是一个固定的非线性函数，应用于神经元内部的加权和（加偏置）计算结果，在结果传递给下一层之前。

在数学上，如果 $z$ 是加权和加偏置 ($z = \sum_i w_i x_i + b$)，神经元的输出 $a$ 由以下公式给出：
$a = f(z)$
其中 $f$ 是激活函数。

$f$ 的重要性质是它的**非线性**。通过在每层线性计算后引入这一非线性步骤，网络打破了线性链条。现在，堆叠层使得MLP能够近似任意形式的函数，使其能够学到线性模型无法捕获的数据中的细致关系和样式。

可以把它想象成弯曲或扭曲数据所存在的空间。线性变换只能拉伸、旋转或剪切这个空间。非线性激活函数使得网络能够在每一步执行更多样的变换，最终使其能够分离那些非线性可分的数据点。



![线性与非线性激活函数对比](plots/5025-0.json)



> 简单线性'激活'（恒等函数）与常见非线性激活函数（ReLU）的对比。非线性使得网络能够对更多样化的关系进行建模。

本质上，激活函数的主要作用是引入深度神经网络 (neural network)学习输入到输出的非线性映射所需的非线性。没有它们，MLP的表示能力将大大降低。虽然有些激活函数有额外的有用特性（比如限制输出范围，我们稍后会讨论），但它们最主要的效用是实现非线性计算。

激活函数的选择可以对网络的训练过程和性能产生很大影响。在接下来的章节中，我们将介绍几种流行的激活函数，并说明它们的特点、优点和缺点。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本综合性教材为深度学习提供了详细的理论基础，包括激活函数的作用和类型。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) — Michael A. Nielsen (2015)
  Publisher: Determination Press
  这本易于理解的在线书籍清晰地解释了神经网络的基本概念，包括非线性激活函数的必要性。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Neural Networks Part 1: Setting up the Architecture](https://cs231n.github.io/convolutional-networks/#activations) — Justin Johnson, Andrej Karpathy, and Fei-Fei Li (2023)
  这是一门知名大学课程的最新讲义，提供了关于神经网络架构组件（包括激活函数）的实用见解。
