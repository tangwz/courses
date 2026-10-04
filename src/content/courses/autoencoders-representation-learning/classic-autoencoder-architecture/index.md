---
course: "autoencoders-representation-learning"
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-2-classic-autoencoder-architecture"
sourceId: 643
chapter: "classic-autoencoder-architecture"
title: "经典自动编码器架构"
order: 2
description: "详细介绍基本自动编码器的主要构成部分：编码器、瓶颈、解码器和重建损失函数。"
hasQuiz: false
---

明确了表示学习的背景之后，我们现在将重点放在自动编码器上。本章介绍标准的自动编码器架构，它是一种用于无监督学习的神经网络。其主要思想是先将输入数据压缩成低维表示（编码），然后从该压缩形式重建原始数据（解码）。这一过程有助于网络学习数据的重要特点。

在本章中，您将审视其主要构成：
*   **编码器网络**：它如何将高维输入 $x$ 映射到潜在代码 $z$。
*   **瓶颈层**：它在承载压缩数据表示 $z$ 中的作用。
*   **解码器网络**：它如何从代码 $z$ 重建数据 $\hat{x}$。
*   **重建损失函数**：衡量 $x$ 和 $\hat{x}$ 之间的差异，使用均方误差 (MSE)
    $$L_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i)^2$$
    或二元交叉熵 (BCE) 等函数，具体取决于输入数据的特点。

我们还将介绍基本自动编码器目标函数的数学定义，并讨论使用当前深度学习库进行实现的相关事宜。最后，通过一个动手实践环节，演示构建一个简单的自动编码器。
