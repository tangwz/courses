---
course: "introduction-to-graph-neural-networks"
chapter: "the-message-passing-mechanism"
lesson: "common-aggregation-functions"
sourceId: 7621
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-2-the-message-passing-mechanism/common-aggregation-functions"
title: "常用的聚合函数"
description: "介绍用于聚合节点邻域消息的常用函数，包括求和、均值和最大池化。"
order: 3
plots: []
sourceHash: "e80241df0c67cf42262a9f64032725f65a2d7402c4f4821bbd5354b1e8c340f0"
sourceCorrections: []
---

`AGGREGATE` 函数是图神经网络 (neural network)消息传递机制中的核心操作。它的作用是接收来自节点邻居的一组特征向量 (vector)（其数量可能各不相同），并将其压缩成一个具有代表性的单一向量。这一步操作十分必要，因为虽然图中的一个节点可能有多个邻居，但后续处理聚合信息的 `UPDATE` 步骤需要固定大小的输入。

聚合器必须是一个置换不变性函数，这意味着处理邻居特征的顺序不会影响输出结果。这一特性对图来说非常重要，因为节点的邻居之间并没有天然的排序。下面我们来看一下满足这一要求的最常用聚合函数。

### 均值聚合 (Mean Aggregation)

均值聚合器计算邻居特征向量 (vector)的逐元素平均值。它提供了邻域特征的汇总，并根据邻居数量进行了归一化 (normalization)。

在数学上，目标节点 $v$ 的聚合消息 $\mathbf{m}_{\mathcal{N}(v)}^{(l)}$ 计算如下：


$$
\mathbf{m}_{\mathcal{N}(v)}^{(l)} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l)}
$$


这里，$|\mathcal{N}(v)|$ 是节点 $v$ 的度（即邻居数量），而 $\mathbf{h}_u^{(l)}$ 是第 $l$ 层中邻居节点 $u$ 的特征向量。

均值聚合的主要特点是其平滑效应。通过取平均值，它能确保聚合向量的量级不会随着邻居数量的增加而增长。这使其成为一个稳定且通常有效的默认选择。然而，这一特性也可能是一个缺点。如果某个邻居具有极具信息量的特征，其信号可能会被平均值稀释，特别是在邻居密集的区域。

### 求和聚合 (Sum Aggregation)

求和聚合器只是简单地将所有邻居的特征向量 (vector)相加。这可能是合并邻域信息最直接的方式。

求和聚合的公式为：


$$
\mathbf{m}_{\mathcal{N}(v)}^{(l)} = \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l)}
$$


与均值不同，求和对邻域的大小很敏感。所得聚合向量的量级将随节点的度而缩放。如果连接数量是希望模型学习的一个重要结构属性，那么这种方式会很有用。例如，在社交网络中，拥有许多朋友的用户与朋友较少的用户在表示上可能会有所不同。求和聚合的主要风险在于，对于度数非常高的节点，求和后的数值可能会变得极大，从而在训练过程中导致数值不稳定的问题。

### 最大值聚合 (Max Aggregation)

最大值聚合（也称为最大池化）取所有邻居特征向量 (vector)中的逐元素最大值。它不是汇总整个邻域，而是识别邻居中出现的最显著特征值。

该操作定义为：


$$
\mathbf{m}_{\mathcal{N}(v)}^{(l)} = \max_{u \in \mathcal{N}(v)} \{ \mathbf{h}_u^{(l)} \}
$$


此操作在特征向量的每个维度上独立执行。最大池化能有效地捕捉具有代表性或显著的特征。例如，如果每个特征维度代表某种属性的存在，最大值聚合会识别出该属性在邻域内的最强存在感。它不太受平均值稀释效应的影响，也不像求和聚合器那样对邻居数量敏感。

### 聚合函数对比

为了更好理解这些函数之间的差异，假设目标节点 $v$ 有三个邻居，每个邻居都有一个二维特征向量 (vector)。

> 目标节点 `v` 的均值、求和及最大值聚合对比。每个聚合器根据邻居特征向量 `[1, 5]`、`[4, 2]` 和 `[2, 2]` 生成了不同的汇总结果。

如图所示：

- **均值 (Mean)** 生成了一个“平均”表示：`[2.33, 3.00]`。
- **求和 (Sum)** 生成了一个量级更大的向量，反映了总体的贡献：`[7, 9]`。
- **最大值 (Max)** 选取了每个维度中最显著的特征：`[4, 5]`，分别取自 `u2` 的 `4` 和 `u1` 的 `5`。

### 选择聚合器

聚合函数的选择取决于具体的任务和图数据的性质。

- **均值** 是一个保险且通用的起点，在许多场景下表现良好。
- **求和** 如果节点度数具有实际意义，它可能是一个更具表达力的替代方案，尽管可能需要归一化 (normalization)来保持训练稳定性。
- **最大值** 当任务涉及识别最具影响力的单个邻居或特征时非常有用。

一些先进的 GNN 架构（例如我们将在下一章看到的 GraphSAGE）将聚合器视为可配置的超参数 (parameter) (hyperparameter)，允许你通过实验找到最适合具体问题的方案。下一节将介绍 `UPDATE` 函数，它接收聚合的消息并将其与节点自身的表示相结合，从而产生新的状态。

## 参考资料

- [Inductive Representation Learning on Large Graphs](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs) — William L. Hamilton, Rex Ying, Jure Leskovec (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 1024-1034; DOI: [10.55985/05f23d4e68e4d262](https://doi.org/10.55985/05f23d4e68e4d262)
  介绍了GraphSAGE，形式化了可学习邻域聚合函数的概念，包括均值、池化（最大值）和LSTM聚合器。
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf, Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  这项基础工作介绍了图卷积网络（GCN），在其消息传递机制中利用了一种特定形式的邻居特征归一化聚合。
- [Graph Neural Networks: A Review of Methods and Applications](https://doi.org/10.1016/j.aiopen.2021.01.001) — Jie Zhou, Ganqu Cui, Shengding Hu, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, Maosong Sun (2020)
  Journal: AI Open; Volume: 1; Pages: 57-81; DOI: [10.1016/j.aiopen.2021.01.001](https://doi.org/10.1016/j.aiopen.2021.01.001)
  概述了图神经网络，讨论了不同的消息传递范式和常见的聚合函数。
- [CS224W: Machine Learning with Graphs (Course Materials)](https://cs224w.stanford.edu/) — Jure Leskovec and Stanford University (2024)
  来自领先大学课程的官方材料，解释了图神经网络的基本概念，包括聚合函数。
