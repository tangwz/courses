---
course: "introduction-to-graph-neural-networks"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-2-the-message-passing-mechanism"
sourceId: 1317
chapter: "the-message-passing-mechanism"
title: "消息传递机制"
order: 2
description: "关于 GNN 核心机制（消息传递）的详细章节。包含了聚合、更新函数以及图层叠加的方式。"
hasQuiz: true
---

在上一章中，我们确定了如何用数值表示图数据。接下来的步骤是理解神经网络如何在这种结构上运行。与图像等具有固定网格拓扑的数据不同，图需要专门的处理方式。本章介绍大多数图神经网络的运行机制：这一过程被称为消息传递。

我们将该机制拆解为各个组成部分。你将学习节点如何从直接邻居处收集信息（聚合），然后使用这些信息来更改自身的表达（更新）。我们将通过通用的数学公式使这一过程规范化，并分析这两个步骤中常用的函数。我们还将通过讨论置换不变性，说明为什么这种设计天生适合图数据。最后，你将看到叠加多个消息传递层如何使 GNN 能够获取一跳之外的节点信息，从而扩大其在图上的感受野。

在本章中，我们将使用数学符号来定义这些运算。例如，单个 GNN 层通常可以表示为：
$$
\mathbf{h}_v^{(l+1)} = \text{UPDATE}^{(l)} \left( \mathbf{h}_v^{(l)}, \text{AGGREGATE}^{(l)} \left( \{ \mathbf{h}_u^{(l)} : u \in \mathcal{N}(v) \} \right) \right)
$$
其中 $\mathbf{h}_v^{(l)}$ 是节点 $v$ 在第 $l$ 层的特征向量，而 $\mathcal{N}(v)$ 表示节点 $v$ 的邻居集合。为了巩固这些内容，本章最后提供了一个实践练习，你将使用 NumPy 从零开始实现一个基础的消息传递层。
