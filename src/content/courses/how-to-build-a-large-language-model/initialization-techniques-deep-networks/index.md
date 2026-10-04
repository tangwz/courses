---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-12-initialization-techniques-deep-networks"
sourceId: 1090
chapter: "initialization-techniques-deep-networks"
title: "深度网络的初始化方法"
order: 12
description: "介绍有效的权重初始化策略，例如Xavier/Glorot和Kaiming初始化，这些策略对于训练非常深的模型非常重要。"
hasQuiz: false
---

训练非常深层的神经网络，例如前面讨论过的多层Transformer，带来独特的难题。影响训练稳定性和收敛速度的一个重要因素是模型权重最初如何设置。不当的初始化会导致诸如梯度消失或梯度爆炸的问题，在此情况下，网络中传播的信号变得过小或过大，从而阻碍学习过程。

本章着重介绍旨在减轻这些问题的系统性权重初始化方法。我们将考察有助于在数据前向流动和梯度反向流动穿过深度网络时保持适当信号方差的已有技术。

你将学到：

*   有原则的权重初始化背后的原理。
*   Xavier（或Glorot）初始化，适用于具有对称激活函数的层，例如$tanh$。
*   Kaiming（或He）初始化，专为使用修正线性单元（ReLU）及其变体的层设计。
*   如何将这些方法具体应用于Transformer架构中的不同组成部分，包括嵌入层、注意力机制和前馈网络。
*   针对模型最终层初始化的特殊考量。

掌握和运用这些技术对于成功训练大型语言模型所需的深层架构非常必要。
