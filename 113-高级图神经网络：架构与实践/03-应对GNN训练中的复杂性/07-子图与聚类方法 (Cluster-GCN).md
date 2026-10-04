# 子图与聚类方法 (Cluster-GCN)

来源：[原文](https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-3-gnn-training-complexities/subgraph-clustering-cluster-gcn)

[返回章节目录](README.md) · [返回课程目录](../README.md)

邻域和图采样方法通过小批量操作，提供了在大图上训练GNN的途径，但它们通常涉及每个批次大量的预处理，或者因采样过程引入方差。一种不同的策略，旨在通过对图本身进行划分来简化批次构建并减少计算负担，即子图和聚类方法，Cluster-GCN就是其代表。

Cluster-GCN背后的主要想法非常直接：如果能将图的节点划分为较少的几个簇，使得簇内的节点紧密相连，也许我们就可以用这些簇形成批次来训练GNN。不再是采样单个节点或边，我们采样整个簇（或多个簇的组合），并在采样到的簇内节点所形成的子图上进行GNN计算。

## Cluster-GCN方法

Cluster-GCN包含两个主要阶段：

1. **图聚类（预处理）**: 在训练开始前，大型图 $G=(V, E)$ 被划分为 $c$ 个簇 $V_1, V_2, ..., V_c$，使得 $V = V_1 \cup V_2 \cup ... \cup V_c$ 且当 $i \neq j$ 时 $V_i \cap V_j = \emptyset$。这种划分通常使用成熟的图聚类算法，如METIS或Louvain来完成，它们旨在最小化簇*之间*的边数（边切），同时保持簇的大小相对均衡。目标是创建这样的划分：图中大部分连接性都保留在簇*内部*。
2. **随机批量训练**: 在训练过程中，我们不再采样节点，而是采样一个或多个簇来形成一个迷你批次。设 $V_b = V_{t_1} \cup ... \cup V_{t_k}$ 是为批次 $b$ 选定的 $k$ 个簇所包含的节点集合。接着我们构建由 $V_b$ 中节点诱导的子图 $G_b = (V_b, E_b)$。这意味着 $E_b$ 只包含原始图 $E$ 中连接 $V_b$ 内两个节点的边。该批次的GNN前向和反向传播 (backpropagation)只使用这个子图 $G_b$ 及其对应的邻接矩阵 $A_b$ 来计算。

> 一个图被划分为三个簇。在Cluster-GCN中，一个迷你批次可能由单个簇的节点和簇内边构成（例如簇2，用双圆圈标出）。簇间边（虚线）在该批次的GNN计算中被忽略。

## Cluster-GCN为何高效

效率提升来自于GNN计算（例如图卷积 $A' H W$，其中 $A'$ 是归一化 (normalization)邻接矩阵）是在子图 $G_b$ 更小的邻接矩阵 $A_b$ 上进行的，而不是在完整图的邻接矩阵 $A$ 上。如果簇的规模适中且簇的数量 $c$ 可控，每个训练步骤都会显著加快，并且与全图训练相比需要更少的内存。

批次的构建本身也得到了简化。一旦完成了初始聚类，创建批次仅涉及选择一个预定义的簇（或几个簇），并获取相应的节点特征和子图结构。这与邻域采样形成对比，后者在训练期间需要对批次中的每个节点进行可能耗费资源的邻域查找。

## 解决近似：簇间边问题

Cluster-GCN中的主要简化，也是近似的来源，是省略了连接当前批次内节点 ($V_b$) 与批次外节点 ($V \setminus V_b$) 的边。当GNN在 $G_b$ 内进行消息传递时，它只聚合来自 $G_b$ *内部*邻居的信息。来自其他簇邻居的信息在该特定批次更新中会丢失。

这有多不利？这在很大程度上取决于聚类的质量。如果聚类算法成功地将节点分组，使得大多数连接都在簇内部（即图具有很强的社区结构且算法找到了它），那么每个批次丢弃的边相对较少，并且近似误差可能很小。如果图缺乏清晰的社区结构或者聚类效果不佳，每个批次可能会忽略许多边，这可能影响GNN学习全局模式的能力。

原始的Cluster-GCN论文提出了一种缓解此问题的方法：不再是每个批次只使用一个簇，而是将少量随机选择的簇（$k > 1$）合并到一个批次中，即 $V_b = V_{t_1} \cup ... \cup V_{t_k}$。这增加了位于一个簇边界附近的节点仍然能从碰巧落在同一批次中另一个簇的邻居那里接收到消息的机会，从而在该步骤的计算中包含更多的簇间边。

## 数学角度

让我们考虑完整图的标准GCN层更新：
$H^{(l+1)} = \sigma(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W^{(l)})$
其中 $\hat{A} = A+I$ 是带自环的邻接矩阵，$\hat{D}$ 是对应的对角度矩阵，$H^{(l)}$ 是第 $l$ 层的节点特征矩阵，且 $W^{(l)}$ 是该层的权重 (weight)矩阵。

在Cluster-GCN中，对于由节点 $V_b$ 定义的批次 $b$，更新是在子图 $G_b=(V_b, E_b)$ 上执行的。设 $A_b$ 是这个子图的邻接矩阵（只包含 $V_b$ 内的边）。批次中节点的更新变为：
$H^{(l+1)}_{V_b} = \sigma(\hat{D}_b^{-1/2} \hat{A}_b \hat{D}_b^{-1/2} H^{(l)}_{V_b} W^{(l)}).$
这里，$\hat{A}_b = A_b+I_{V_b}$ 是为 $V_b$ 中的节点添加自环的子图邻接矩阵，$\hat{D}_b$ 是其度矩阵，而 $H^{(l)}_{V_b}$ 表示 $H^{(l)}$ 中与 $V_b$ 中节点对应的行。请注意，计算只涉及特征 $H^{(l)}_{V_b}$ 和结构 $\hat{A}_b$，如果 $|V_b| \ll |V|$，则计算成本会显著降低。

## 权衡与比较

**优点：**

- **可扩展性**: 能够在大规模图上进行训练，这些图在使用全批次或简单小批量时无法完全载入GPU内存。通常，每步内存效率高于邻居采样，后者由于多跳邻域可能占用大量内存。
- **计算效率**: 在较小的子图上执行GNN计算，与全批次方法相比，这使得训练步骤更快。初始聚类完成后，批次创建的开销很低。
- **实现简化**: 一旦完成聚类，训练循环结构相对简单，相比复杂的邻居或图采样过程。

**缺点：**

- **近似误差**: 忽略簇间边会引入偏差。其有效性在很大程度上取决于图结构和聚类质量。
- **聚类开销**: 需要一个初始的、可能耗时的图聚类步骤。该聚类的质量直接影响性能。
- **潜在的收敛速度较慢**: 由于近似，收敛有时可能更慢或达到稍低的精度，相比那些更精确地采样邻域的方法（如GraphSAINT）或全批次训练（如果可行）。

**与采样方法的比较：**

- **邻域采样（例如GraphSAGE）**: 为批次中的每个节点采样邻居。避免了结构近似，但可能面临高方差和计算开销（“邻居爆炸”）的问题。Cluster-GCN固定了批次的计算图，减少了方差和每批次的开销，但引入了结构近似。
- **图采样（例如GraphSAINT）**: 使用节点或边采样来采样整个子图。旨在创建使GNN计算以低方差近似全批次梯度的批次。实现和调整采样概率可能很复杂。Cluster-GCN采用确定性划分方法，简化了批次创建，但代价是固定的结构近似。

## 实现说明

像PyTorch Geometric (PyG) 这样的库为Cluster-GCN提供了方便的数据加载器。例如，`torch_geometric.loader.ClusterLoader` 接收一个图数据对象，执行聚类（使用与METIS接口的 `torch_cluster` 包），并生成与簇或簇组对应的批次。这简化了划分和子图构建的许多复杂性。

```python
# 使用PyTorch Geometric的示例
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.datasets import Planetoid # 或您的大型图数据集

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# 执行聚类（预处理）
cluster_data = ClusterData(data, num_parts=100, recursive=False, save_dir=dataset.processed_dir)

# 创建加载器，它会生成簇批次
train_loader = ClusterLoader(cluster_data, batch_size=20, shuffle=True)

# 训练循环
model = GCN(...) # 您的GNN模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for batch in train_loader:
    optimizer.zero_grad()
    # 'batch' 对象已经是所选簇的子图
    output = model(batch.x, batch.edge_index)
    # 只计算当前批次中节点的损失
    loss = F.nll_loss(output[batch.train_mask], batch.y[batch.train_mask])
    loss.backward()
    optimizer.step()
```

总而言之，Cluster-GCN提供了一种通过图聚类来扩展GNN训练的有效策略。它牺牲了簇边界处消息传递的准确性，以换取计算速度和内存效率上的显著提升，使其成为处理真正大规模图学习问题（尤其是那些具有固有社区结构的问题）的一种有益技术。

## 参考资料

- [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://dl.acm.org/doi/10.1145/3292500.3330989) — Wei-Lin Chiang, Meng Qu, Xisen Jin, Stein Lim, Jure Leskovec, S. P. K. Shen (2019)
  Journal: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD); Publisher: ACM; Pages: 257–266; DOI: [10.1145/3292500.3330989](https://doi.org/10.1145/3292500.3330989)
  介绍 Cluster-GCN 方法的原始论文，概述了其图分区策略和用于大型图的随机训练方法。
- [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf, Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  引入图卷积网络 (GCN) 架构，为 GNN 数学更新公式奠定基础。
- [A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs](https://doi.org/10.1137/S1064827595287997) — George Karypis, Vipin Kumar (1998)
  Journal: SIAM Journal on Scientific Computing; Publisher: Society for Industrial and Applied Mathematics; Volume: 20; Pages: 359-392; DOI: [10.1137/S1064827595287997](https://doi.org/10.1137/S1064827595287997)
  详细介绍了 METIS 算法，这是一种在 Cluster-GCN 预处理阶段使用的图分区方法。

---

[上一节](06-%E5%9B%BE%E9%87%87%E6%A0%B7%E6%8A%80%E6%9C%AF%20%28GraphSAINT%2C%20ShaDow-GNN%29.md) · [下一节](08-%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E4%BC%98%E5%8C%96%E7%AD%96%E7%95%A5.md)
