# 第 2 章：消息传递机制

来源：[原章节](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-2-the-message-passing-mechanism)

[返回课程目录](../README.md)

在上一章中，我们确定了如何用数值表示图数据。接下来的步骤是理解神经网络如何在这种结构上运行。与图像等具有固定网格拓扑的数据不同，图需要专门的处理方式。本章介绍大多数图神经网络的运行机制：这一过程被称为消息传递。

我们将该机制拆解为各个组成部分。你将学习节点如何从直接邻居处收集信息（聚合），然后使用这些信息来更改自身的表达（更新）。我们将通过通用的数学公式使这一过程规范化，并分析这两个步骤中常用的函数。我们还将通过讨论置换不变性，说明为什么这种设计天生适合图数据。最后，你将看到叠加多个消息传递层如何使 GNN 能够获取一跳之外的节点信息，从而扩大其在图上的感受野。

在本章中，我们将使用数学符号来定义这些运算。例如，单个 GNN 层通常可以表示为：
$$
\mathbf{h}_v^{(l+1)} = \text{UPDATE}^{(l)} \left( \mathbf{h}_v^{(l)}, \text{AGGREGATE}^{(l)} \left( \{ \mathbf{h}_u^{(l)} : u \in \mathcal{N}(v) \} \right) \right)
$$
其中 $\mathbf{h}_v^{(l)}$ 是节点 $v$ 在第 $l$ 层的特征向量，而 $\mathcal{N}(v)$ 表示节点 $v$ 的邻居集合。为了巩固这些内容，本章最后提供了一个实践练习，你将使用 NumPy 从零开始实现一个基础的消息传递层。

## 小节

- 1. [邻域聚合思想](01-%E9%82%BB%E5%9F%9F%E8%81%9A%E5%90%88%E6%80%9D%E6%83%B3.md)
- 2. [通用 GNN 层：聚合与更新](02-%E9%80%9A%E7%94%A8%20GNN%20%E5%B1%82%EF%BC%9A%E8%81%9A%E5%90%88%E4%B8%8E%E6%9B%B4%E6%96%B0.md)
- 3. [常用的聚合函数](03-%E5%B8%B8%E7%94%A8%E7%9A%84%E8%81%9A%E5%90%88%E5%87%BD%E6%95%B0.md)
- 4. [更新函数与非线性](04-%E6%9B%B4%E6%96%B0%E5%87%BD%E6%95%B0%E4%B8%8E%E9%9D%9E%E7%BA%BF%E6%80%A7.md)
- 5. [置换不变性与置换等变性](05-%E7%BD%AE%E6%8D%A2%E4%B8%8D%E5%8F%98%E6%80%A7%E4%B8%8E%E7%BD%AE%E6%8D%A2%E7%AD%89%E5%8F%98%E6%80%A7.md)
- 6. [堆叠层以构建深度图神经网络 (GNN)](06-%E5%A0%86%E5%8F%A0%E5%B1%82%E4%BB%A5%E6%9E%84%E5%BB%BA%E6%B7%B1%E5%BA%A6%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20%28GNN%29.md)
- 7. [实践：使用 NumPy 实现简单的 GNN 层](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20NumPy%20%E5%AE%9E%E7%8E%B0%E7%AE%80%E5%8D%95%E7%9A%84%20GNN%20%E5%B1%82.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-2-the-message-passing-mechanism/quiz)
