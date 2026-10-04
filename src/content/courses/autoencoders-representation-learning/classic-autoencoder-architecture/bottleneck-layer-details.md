---
course: "autoencoders-representation-learning"
chapter: "classic-autoencoder-architecture"
lesson: "bottleneck-layer-details"
sourceId: 2808
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-2-classic-autoencoder-architecture/bottleneck-layer-details"
title: "瓶颈层"
description: "了解瓶颈在创建压缩数据表示中的作用。"
order: 2
plots: []
sourceHash: "824d891912c8128f737e91c725601c6e80c9fc974db1be1495285b2dccde6886"
sourceCorrections: []
---

自编码器旨在学习输入数据的压缩表示。这种压缩得以体现的核心组成部分是**瓶颈层**。该层位于编码器和解码器之间，根本上是网络架构中最窄的部分，迫使来自输入 $x$ 的信息被挤压成低维潜在空间表示，记作 $z$。

可以将编码器看作一个过程，它接收高维输入向量 (vector) $x \in \mathbb{R}^D$，并通过（可能包含非线性激活函数 (activation function)的）多个中间层，将其映射到低维潜在向量 $z \in \mathbb{R}^d$，且 $d < D$。瓶颈层正是其输出*即*此潜在向量 $z$ 的那一层。

> 瓶颈层在结构上位于编码器和解码器之间，存储着压缩后的潜在表示 $z$。

瓶颈层的维度 $d$ 是一个重要超参数 (parameter) (hyperparameter)。它直接控制着压缩程度。

- **$d$ 过小**：网络可能难以捕获足够的信息以准确重建输入，导致较高的重建损失。它可能只捕获数据中最突出的变化。
- **$d$ 过大（接近 $D$）**：网络可能只是学习一个恒等函数，实际将输入复制到输出，而不学习任何有意义的潜在结构或抽象。这违背了表示学习和降维的目的。

瓶颈施加的限制迫使自编码器高效地学习。它必须优先捕获输入数据中最显著的特征或变化因素，以最小化重建误差。与主成分分析（PCA）等寻找低维*线性*子空间的方法不同，自编码器通过其编码器和解码器中的非线性激活函数，可以学习数据中的复杂非线性流形。瓶颈表示 $z$ 因此代表了此学习到的流形上的坐标。

瓶颈层中存储的这种压缩表示 $z$ 通常被称为**潜在代码**、**潜在变量**，或简称为**嵌入 (embedding)**。它是编码过程的核心输出，也是解码过程的唯一输入。这种潜在空间的质量和结构对于自编码器执行其任务（无论是降维、特征学习，还是在更高级的变体中，生成建模）的表现非常重要。我们将在后续章节中研究这个潜在空间的属性和操作。

在实现方面，瓶颈在简单的自编码器中通常是一个标准的密集（全连接）层，在为特定数据类型设计的高级架构中，也可能是一个卷积层或循环层。其主要特点不一定是层的*类型*，而是其相对于输入和输出的降低的维度，充当编码器和解码器之间的信息通道。理解它的作用对于理解自编码器如何实现表示学习来说是根本的。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供了关于自编码器的全面理论背景，包括瓶颈层和表示学习的目的与功能。
- [Reducing the Dimensionality of Data with Neural Networks](https://www.science.org/doi/10.1126/science.1127647) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  开创性论文，展示了深度自编码器在降维和学习压缩表示方面的有效性。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125968/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 864
  提供实现自编码器的实践见解，包括瓶颈层的细节及其维度超参数的考量。第4版。
- [Pattern Recognition and Machine Learning](https://www.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/b100889](https://doi.org/10.1007/b100889)
  从经典的机器学习角度介绍了降维方法，为自编码器作为PCA等技术的非线性扩展提供了背景。
