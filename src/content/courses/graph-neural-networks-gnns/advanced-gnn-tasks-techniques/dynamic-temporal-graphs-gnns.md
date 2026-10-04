---
course: "graph-neural-networks-gnns"
chapter: "advanced-gnn-tasks-techniques"
lesson: "dynamic-temporal-graphs-gnns"
sourceId: 2724
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-4-advanced-gnn-tasks-techniques/dynamic-temporal-graphs-gnns"
title: "动态和时序图的建模"
description: "了解将GNN应用于随时间变化的图，并结合时间信息的方法。"
order: 2
plots: []
sourceHash: "14deae0240fa04b1295b61f2eca9f1cac9774bf2c311f63113b537bfa1077ad6"
sourceCorrections: []
---

尽管许多系统可以表示为静态图，但许多现象本身就具有动态特性。社交网络随着友谊的建立和瓦解而演变，通信网络流量波动，金融交易图随每次交易而增长，生物交互网络也随时间变化。对这些系统进行建模，需要图神经网络 (neural network)不仅能够捕捉图快照中的复杂关系，还能捕捉这些关系和节点属性随时间的演变。针对动态图和时序图，提供了使GNN适应处理的方法。

### 理解动态图

动态图是指其结构（节点和边）或特征（节点或边属性）随时间变化的图。根据时间处理方式，我们可以大致对表示形式和相应的建模方法进行分类：

1. **离散时间动态图（快照）：** 演变中的图被表示为在离散时间间隔拍摄的一系列静态图快照：$G_1, G_2, ..., G_T$。每个快照 $G_t = (V_t, E_t, X_t)$ 捕捉图在时间 $t$ 的状态。这是一种常用表示形式，尤其当数据定期收集时。然而，它可能错过快照之间发生的细微变化，并假定一个固定（有时是任意的）时间粒度。
2. **连续时间动态图（事件流）：** 图的演变被表示为带有时间戳的连续事件流，例如节点添加/删除、边形成/移除或特征更新。每个事件都关联一个精确的时间戳。这种表示形式捕捉图动态的完整时间分辨率，但也带来处理不规则时间事件和可能巨大事件量的挑战。

表示形式的选择通常取决于数据的可得性和具体任务。建模方法因处理快照序列还是连续事件流而有显著差异。

### 离散时间方法：GNN与序列模型的结合

处理图快照时，一个直接的方法是结合GNN的空间处理能力与序列模型（主要是像LSTM或GRU这样的循环神经网络 (neural network)（RNN））的时间建模能力。

一般思路是，首先在每个时间步 $t$ 使用一个GNN来计算节点嵌入 (embedding)，以捕捉该快照内的图结构和特征：


$$
H_t = GNN(A_t, X_t)
$$


在此，$A_t$ 和 $X_t$ 分别是时间 $t$ 的邻接矩阵和节点特征矩阵，$H_t$ 是节点嵌入矩阵。

这些节点嵌入序列（如果适用，也可以是图级别嵌入）然后输入到RNN中，对时间依赖性进行建模：


$$
S_{t+1}, O_{t+1} = RNN(H_t, S_t)
$$


其中 $S_t$ 是时间 $t$ 时RNN的隐藏状态，$O_{t+1}$ 可能是下一时间步的预测（例如，未来节点标签、预测链接）。

> 离散时间动态GNN模型图示，该模型使用循环神经网络（RNN）组件来处理每个时间步由GNN生成的图嵌入序列。

**模型示例：**

- **GCN-LSTM：** 一个直接的实现，使用GCN进行空间聚合，并使用LSTM进行时间建模。
- **EvolveGCN：** 解决了简单GNN-RNN组合中GNN参数 (parameter)在时间上是静态的局限性。EvolveGCN提出使用RNN（如GRU）在每个时间步动态更新GCN层本身的*参数*，使GNN模型能够根据演变中的图动态调整其行为。

这些离散时间模型在存在清晰快照且快照间动态变化不那么重要时是有效的。然而，它们计算成本较高，因为它们需要为每个快照运行一个GNN。

### 连续时间方法：基于事件的建模

连续时间模型旨在通过直接处理带有时间戳的事件流，以更精细的粒度捕捉图动态。这些模型通常避免离散化为快照，原则上可以处理任何时间点发生的交互。

一个主要的类别包括**基于记忆的方法**，以\*\*时序图网络（TGN）\*\*框架为例。TGN为每个节点维护一个记忆向量 (vector)（或状态），表示其历史信息。当涉及某些节点的交互（事件）在时间 $t$ 发生时：

1. 检索交互节点的当前记忆状态。
2. 这些记忆状态，连同事件特征和时间信息，被处理（通常使用注意力机制 (attention mechanism)和MLP层）以计算消息。
3. 这些消息被用于更新相关节点的记忆状态。
4. 交互信息和更新后的记忆随后可用于计算时序嵌入 (embedding)，以进行链接预测或节点分类等下游任务。

**TGN等连续时间模型中常见组件：**

- **记忆模块：** 随时间存储和更新节点状态。
- **嵌入模块：** 根据节点记忆和最近交互计算节点的时序嵌入。
- **时间编码：** 表示事件连续时间的方法，通常涉及将时间差（$\Delta t$）映射到特征向量（例如，使用径向基函数或习得的时间嵌入）。
- **交互处理：** 处理事件的模块（通常是神经网络 (neural network)），使用节点记忆和时间信息。

连续时间模型提供更高的时间精度，但实现和训练起来可能更复杂。它们需要高效地管理记忆和处理异步事件流的方式。

### 动态GNN的技术考量

无论采用何种方法，在动态图建模时，有几个技术方面非常重要：

- **时间信息编码：** 有效表示时间非常重要。这可以包括编码绝对时间戳、事件之间的时间差，或使用适应时间维度的位置编码 (positional encoding)。这种选择会影响模型对新近度和持续时间的感知。
- **时序注意力：** 注意力机制 (attention mechanism)可以调整以根据时间接近度或相关性来衡量过去交互或快照的影响，使模型能够关注相关历史。
- **结构变化处理：** 模型必须适应节点和边的出现或消失。这可能涉及初始化新节点记忆/嵌入 (embedding)的策略，处理邻域变化的节点的消息传递，或明确建模节点/边的存在概率。
- **计算可伸缩性：** 时间维度增加了显著的计算开销。处理长序列快照或海量事件流需要高效的实现，可能涉及适应时间域的采样技术（例如，时间邻域采样）或近似方法。
- **评估：** 评估动态GNN需要仔细考虑时间分割策略（例如，在 $t_0$ 到 $t_k$ 进行训练，在 $t_{k+1}$ 到 $t_m$ 进行验证，在 $t_{m+1}$ 到 $T$ 进行测试），以防止未来信息泄露。未来链接预测等任务是常见的基准测试。

### 应用

动态和时序GNN应用于各种随时间演变是其定义特征的问题：

- **链接预测：** 预测社交网络、协作网络或交易图中的未来交互。
- **节点分类/回归：** 对节点进行分类（例如，检测异常用户/交易），或预测节点属性，其中行为随时间演变。
- **事件时间预测：** 预测涉及节点的下一次交互可能何时发生。
- **推荐系统：** 建模用户偏好和物品交互的时间演变。
- **流行病预测：** 建模随时间变化的接触网络上的疾病传播。

与静态图分析相比，动态图建模增加了一层复杂性，但这能帮助我们理解和预测那些不断变化的系统。在离散时间方法和连续时间方法之间做出选择，很大程度上取决于数据的性质和任务的具体要求。随着研究的进展，用于捕捉时空依赖性的混合方法和更复杂的技术持续出现。

## 参考资料

- [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graph Representation Learning](https://doi.org/10.1609/aaai.v34i04.5984) — Aldo Pareja, Giacomo Domeniconi, Jie Chen, Tengfei Ma, Toyotaro Suzumura, Hiroki Kanezashi, Tim Kaler, Tao Schardl, Charles Leiserson (2020)
  Journal: Proceedings of the AAAI Conference on Artificial Intelligence; Publisher: Association for the Advancement of Artificial Intelligence; Volume: 34; Pages: 5363-5370; DOI: [10.1609/aaai.v34i04.5984](https://doi.org/10.1609/aaai.v34i04.5984)
  提出一种利用RNN动态更新GNN参数以处理离散时间动态图的方法。
- [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637) — Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, Michael Bronstein (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 33; DOI: [10.48550/arXiv.2006.10637](https://doi.org/10.48550/arXiv.2006.10637)
  提出时间图网络 (TGN) 框架，一种用于连续时间动态图的基于内存的方法。
