# 图卷积网络 (GCN)

来源：[原文](https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-2-advanced-gnn-architectures/gcn-exploration)

[返回章节目录](README.md) · [返回课程目录](../README.md)

对几种特定的高级图神经网络 (neural network)架构进行了探讨。虽然存在许多创新设计，但图卷积网络（GCN）依然是一种核心模型，也是理解更复杂模型的常用起始点。其谱域渊源将得到更正式的分析，并与广泛使用的基于消息传递的空间实现方法关联起来。

### 从谱域滤波器到空间聚合

回顾第1章内容，谱域图卷积在傅里叶域中操作，由图拉普拉斯算子$L$的特征向量 (vector)定义。谱域滤波器$g_\theta$应用于图信号$x$的定义如下：


$$
g_\theta * x = U g_\theta(\Lambda) U^T x
$$


这里，$U$包含$L = D - A$的特征向量，$\Lambda$是对角特征值矩阵，$g_\theta(\Lambda)$将滤波器函数逐元素应用于特征值。尽管理论上很优美，但这种表述存在两个主要缺陷：

1. **计算开销：** 对大型图计算$L$的特征分解计算成本高昂，通常为$O(N^3)$，其中$N$是节点数量。
2. **非局部性：** 特征向量$U$通常是稠密的，这意味着滤波器在顶点域中不具有局部性；改变单个节点的特征会影响整个图的输出。

为解决这些问题，提出了使用拉普拉斯算子多项式的近似方法。例如，ChebNet使用切比雪夫多项式$T_k$来近似滤波器：


$$
g_\theta(\Lambda) \approx \sum_{k=0}^K \theta_k T_k(\tilde{\Lambda})
$$


这里，$\tilde{\Lambda}$是$\Lambda$的缩放版本，使其值位于$[-1, 1]$内，这是切比雪夫多项式定义的定义域。这种多项式近似使滤波器成为$K$-局部化，意味着它只依赖于节点的$K$跳邻域，并避免了特征向量的显式计算。

### GCN的简化

图卷积网络，由Kipf和Welling（2017）提出，可以看作是这种谱域方法的一种特定且高效的简化。它主要进行了两个重要的近似：

1. **线性滤波器：** 它将多项式滤波器限制在$K=1$。这意味着滤波器只考虑了直接（1跳）邻域。滤波器函数相对于拉普拉斯算子的特征值变为线性：$g_\theta(\Lambda) \approx \theta_0 I + \theta_1 \Lambda$。
2. **参数 (parameter)减少：** 它通过设置$\theta_0 = -\theta_1 = \theta$进一步简化，将滤波器简化为$g_\theta(\Lambda) \approx \theta (I - \Lambda)$。然而，实际应用中的GCN使用基于重新归一化 (normalization)邻接矩阵的略微不同表述。

实际GCN表述中的主要步骤涉及一种再归一化技巧。取代直接使用拉普拉斯算子$L = I - D^{-1/2} A D^{-1/2}$（归一化拉普拉斯算子），GCN采用修改后的邻接矩阵$\hat{A}$：


$$
\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}
$$


这里，$\tilde{A} = A + I$是添加自环的邻接矩阵，$\tilde{D}$是$\tilde{A}$的对角度数矩阵（即，$\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$）。

为何采用这种特殊形式？

- **添加自环 ($A+I$)：** 确保节点自身的特征被包含在消息传递过程的聚合操作中。若没有自环，节点的更新表示将只依赖于其邻居。
- **对称归一化 ($\tilde{D}^{-1/2} \dots \tilde{D}^{-1/2}$)：** 这种归一化防止节点特征向量 (vector)的尺度在聚合过程中根据节点度数发生剧烈变化。乘以$A$会汇总邻居特征，导致高连接度节点有更大的特征值。除以$\tilde{D}$（如在$\tilde{D}^{-1}\tilde{A}$中）会平均邻居特征，但可能导致其他数值不稳定问题。对称归一化$\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$平衡了这些影响，并具有有利的谱域属性。当其源自$I+D^{-1/2}AD^{-1/2}$时（假设$A$具有自环或最初使用$A+I$），其特征值被限制在$[0, 2]$范围内，这有助于稳定训练，尤其是在更深的网络中。

### GCN层传播规则

结合这些要素，GCN的逐层传播规则呈现出非常简单的形式。给定第$l$层的节点特征$H^{(l)} \in \mathbb{R}^{N \times F_l}$（其中$N$是节点数量，$F_l$是特征维度），下一层的特征$H^{(l+1)} \in \mathbb{R}^{N \times F_{l+1}}$计算如下：


$$
H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})
$$


我们来分解一下：

- $H^{(l)}$：来自上一层（或$l=0$时的初始节点特征）的节点嵌入 (embedding)矩阵。
- $W^{(l)} \in \mathbb{R}^{F_l \times F_{l+1}}$：一个层特有的可训练权重 (weight)矩阵。该矩阵对每个节点的特征进行变换。
- $\hat{A} \in \mathbb{R}^{N \times N}$：预先计算好的、带自环的归一化 (normalization)邻接矩阵。乘以$\hat{A}$执行核心的邻域聚合。具体来说，$(\hat{A} X)_i = \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} X_j$，其中$X = H^{(l)} W^{(l)}$且$\tilde{d}_i$是节点$i$在$\tilde{A}$中的度数。
- $\sigma(\cdot)$：一个逐元素的非线性激活函数 (activation function)，例如ReLU或ELU。

多个GCN层可以堆叠，使信息在图中传播更远距离。一个用于半监督节点分类的典型两层GCN可能如下所示：


$$
Z = \text{softmax}(\hat{A} \text{ ReLU}(\hat{A} X W^{(0)}) W^{(1)})
$$


这里，$X$是输入特征，$W^{(0)}$和$W^{(1)}$是权重矩阵，$Z$包含每个节点的输出类别概率。

### GCN作为消息传递

GCN传播规则很好地符合第1章中讨论的消息传递框架。对于单个节点$i$，其更新可以看作是：

1. **变换：** 每个节点$j$（由于$\tilde{A}$中的自环，包括$i$自身）使用权重 (weight)矩阵变换其特征向量 (vector)：$m_j^{(l)} = h_j^{(l)} W^{(l)}$。
2. **聚合：** 节点$i$聚合来自其邻居（和自身）的变换后的消息，并由$\hat{A}$中的归一化 (normalization)常数加权：
   
   $$
   a_i^{(l)} = \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} m_j^{(l)}
   $$
   
   这种聚合通过矩阵乘法$\hat{A} (H^{(l)} W^{(l)})$隐式完成。
3. **更新：** 应用非线性激活函数 (activation function)：
   
   $$
   h_i^{(l+1)} = \sigma(a_i^{(l)})
   $$
   

这种视角表明，GCN执行一种由图结构（$\hat{A}$）决定的特定、固定形式的邻域聚合，然后是共享的线性变换（$W^{(l)}$）和非线性处理。

> 节点$i$的GCN层更新的消息传递视角。来自邻居和节点自身的特征（$h_j, h_i$）由$W^{(l)}$变换，使用从归一化邻接矩阵$\hat{A}$派生的权重进行聚合，然后经过非线性函数$\sigma$处理。

### 优点与局限性

GCN因其简洁性、高效性（与早期谱域方法相比）以及在引文网络（如Cora、Citeseer、Pubmed）上的半监督节点分类等基准任务中表现良好而受到欢迎。它们提供了一个有效的基线模型。

然而，GCN也存在局限性：

- **表达能力有限：** 基于$\hat{A}$的固定聚合机制（归一化 (normalization)后）平等对待所有邻居，限制了模型区分特定图结构的能力，正如第1章中与WL测试相关讨论的那样。
- **过平滑：** 堆叠多个GCN层往往会使节点表示趋向于一个共同值，丧失区分能力。邻域间的重复平均操作使节点特征变得平滑。
- **同质性假设：** GCN隐式地在同质性假设下表现最佳，即连接的节点倾向于具有相似特征或标签。平均机制强化了这一点。在存在显著异质性（即连接节点通常不相似）的图上，性能可能下降。
- **原生处理带权边能力不足：** 标准GCN表述使用无权邻接矩阵$A$。尽管存在修改版本，但基本模型并未固有地将边权重 (weight)或特征纳入聚合加权中。

这些局限性促使了更精巧架构的出现。例如，接下来讨论的图注意力网络（GAT）引入了可学习的注意力机制 (attention mechanism)，在聚合过程中为邻居分配不同的重要性权重，直接解决了GCN固定的加权机制问题。其他架构解决了过平滑问题，或使GNN适用于不同的图类型和任务，我们将在本课程中进行探讨。理解GCN的谱域起源和实际消息传递实现方法，为领会后续进展提供了根本依据。

## 参考资料

- [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf) — Thomas N. Kipf, Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); Publisher: OpenReview.net; DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍图卷积网络（GCN）模型的基础论文，详细阐述了其架构、传播规则和半监督学习应用。
- [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/pdf/1606.09375.pdf) — Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst (2016)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1606.09375](https://doi.org/10.48550/arXiv.1606.09375)
  介绍切比雪夫图卷积网络（ChebNet），这是一种使用切比雪夫多项式近似图卷积的谱方法，是简化GCN的直接前身。
- [A Comprehensive Survey on Graph Neural Networks](https://doi.org/10.1109/TNNLS.2020.2978386) — Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, Philip S. Yu (2020)
  Journal: IEEE Transactions on Neural Networks and Learning Systems; Publisher: IEEE; Volume: 32; Pages: 4-24; DOI: [10.1109/TNNLS.2020.2978386](https://doi.org/10.1109/TNNLS.2020.2978386)
  一篇综述文章，概述了图神经网络，对不同架构进行分类，讨论了GCN，并强调了该领域的趋势和挑战。

---

[上一节](../01-%E9%87%8D%E6%B8%A9%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E7%A1%80/05-%E8%A1%A8%E8%BE%BE%E8%83%BD%E5%8A%9B%E4%B8%8EWL%E6%B5%8B%E8%AF%95.md) · [下一节](02-%E5%9B%BE%E6%B3%A8%E6%84%8F%E5%8A%9B%E7%BD%91%E7%BB%9C%20%28GAT%29.md)
