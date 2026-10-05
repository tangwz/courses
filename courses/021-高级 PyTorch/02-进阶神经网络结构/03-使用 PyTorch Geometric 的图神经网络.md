# 使用 PyTorch Geometric 的图神经网络

来源：[原文](https://apxml.com/zh/courses/advanced-pytorch/chapter-2-advanced-network-architectures/graph-neural-networks)

[返回章节目录](README.md) · [返回课程目录](../README.md)

许多数据集本质上是关系型的，最适合用图来表示。例子有社交网络、分子结构、引用网络、知识图谱和推荐系统。传统深度学习 (deep learning)结构，如 CNN 和 RNN，假定数据结构是网格状或序列状的，这使它们不适合处理图中存在的任意连接。图神经网络 (neural network) (GNN) 专门设计用于直接处理图结构数据，它们学习到的表示会同时包含节点特征和图的拓扑结构。

PyTorch Geometric (PyG) 是一个功能强大且被广泛采用的库，它建立在 PyTorch 之上，用于开发和应用 GNN。它提供了多种 GNN 层的优化实现、高效的图数据处理以及常见的图基准数据集。本节将指导您如何使用 PyG 来实现和理解不同的 GNN 结构。

### 在 PyTorch Geometric 中表示图

在构建 GNN 模型之前，我们需要一种标准化的方式来表示图数据。PyG 使用 `torch_geometric.data.Data` 对象。一个 `Data` 对象包含描述单个图的各种属性：

- `x`: 节点特征矩阵，形状为 `[num_nodes, num_node_features]`。每行代表一个节点，列代表其特征。
- `edge_index`: 图的连接信息，采用 COO (坐标) 格式，形状为 `[2, num_edges]`。它存储每条边的源节点和目标节点索引。对于从节点 `j` 到节点 `i` 的边，其列为 `[j, i]`。这种表示对于稀疏图是高效的。
- `edge_attr`: 边特征矩阵，形状为 `[num_edges, num_edge_features]`。表示与每条边相关的可选特征。
- `y`: 目标标签或值，取决于具体任务。对于节点级任务，形状为 `[num_nodes, ...]`；对于图级任务，形状为 `[1, ...]`。
- `pos`: 节点位置特征，形状为 `[num_nodes, num_dimensions]`。常用于几何深度学习 (deep learning)。

下面是创建简单 `Data` 对象的方法：

```python
import torch
from torch_geometric.data import Data

# 节点特征：3 个节点，每个节点 2 个特征
x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)

# 边：(0 -> 1), (1 -> 0), (1 -> 2), (2 -> 1)
# 表示为源节点和目标节点
edge_index = torch.tensor([[0, 1, 1, 2],  # 源节点
                           [1, 0, 2, 1]], # 目标节点
                          dtype=torch.long)

# 可选的边特征：4 条边，每条边 1 个特征
edge_attr = torch.tensor([[0.5], [0.5], [0.8], [0.8]], dtype=torch.float)

# 可选的节点标签（例如，用于节点分类）
y = torch.tensor([0, 1, 0], dtype=torch.long)

# 创建 Data 对象
graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

print(graph_data)
# 输出：Data(x=[3, 2], edge_index=[2, 4], edge_attr=[4, 1], y=[3])
```

PyG 还提供了 `torch_geometric.data.Dataset` 和 `torch_geometric.loader.DataLoader`，用于高效处理图集合并创建小批量数据。`DataLoader` 会自动将不同大小的图整理成更大的批处理对象。

### 消息传递方法

大多数 GNN 层都基于消息传递原理运行。核心思想是每个节点通过聚合来自其局部邻域的信息，迭代地更新其特征表示（嵌入 (embedding)）。这个过程通常包含对层 $l$ 中每个节点 $i$ 的三个步骤：

1. **消息计算：** 每个邻居节点 $j \in \mathcal{N}(i)$ 根据其自身特征 $\mathbf{h}_j^{(l-1)}$，以及目标节点特征 $\mathbf{h}_i^{(l-1)}$ 和边特征 $\mathbf{e}_{j,i}$，计算一个消息 $\mathbf{m}_{j \to i}^{(l)}$。

   
   $$
   \mathbf{m}_{j \to i}^{(l)} = \phi^{(l)}(\mathbf{h}_i^{(l-1)}, \mathbf{h}_j^{(l-1)}, \mathbf{e}_{j,i})
   $$
   

   其中 $\phi^{(l)}$ 是一个可微分的消息函数（例如，一个神经网络 (neural network)）。
2. **聚合：** 节点 $i$ 使用一个置换不变函数 $\bigoplus$（如求和、平均或最大值）来聚合来自其邻居的所有传入消息。

   
   $$
   \mathbf{a}_i^{(l)} = \bigoplus_{j \in \mathcal{N}(i)} \mathbf{m}_{j \to i}^{(l)}
   $$
   
3. **更新：** 节点 $i$ 根据其先前的表示 $\mathbf{h}_i^{(l-1)}$ 和聚合后的消息 $\mathbf{a}_i^{(l)}$ 来更新其特征向量 (vector) $\mathbf{h}_i^{(l)}$。

   
   $$
   \mathbf{h}_i^{(l)} = \gamma^{(l)}(\mathbf{h}_i^{(l-1)}, \mathbf{a}_i^{(l)})
   $$
   

   其中 $\gamma^{(l)}$ 是一个可微分的更新函数（例如，另一个神经网络或简单地添加聚合消息）。

初始特征 $\mathbf{h}_i^{(0)}$ 通常是输入节点特征 `data.x`。堆叠多个消息传递层可以使信息在图中传播更远的距离。

> 此图表呈现了更新节点 $i$ 的消息传递理念。来自邻居 $j_1, j_2, j_3$ 的信息（消息 $m_1, m_2, m_3$）被聚合，并与节点的先前状态 $h_i^{(l-1)}$ 结合，以计算出新状态 $h_i^{(l)}$。

PyG 在其层类中提供了这些步骤的优化实现。

### PyTorch Geometric 中常用的 GNN 层

PyG 提供了多种预实现的 GNN 层。让我们看看三个流行的例子：GCN、GraphSAGE 和 GAT。

#### 图卷积网络 (GCN)

GCN 层由 Kipf & Welling (2017) 提出，执行基于谱的图卷积。GCN 层的消息传递更新规则可以简化为：


$$
\mathbf{H}^{(l+1)} = \sigma\left(\hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)
$$


其中，$\mathbf{H}^{(l)}$ 是层 $l$ 的节点嵌入 (embedding)矩阵，$\mathbf{W}^{(l)}$ 是一个可训练的权重 (weight)矩阵，$\sigma$ 是一个激活函数 (activation function)（如 ReLU），$\hat{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ 是添加了自循环的邻接矩阵，$\hat{\mathbf{D}}$ 是 $\hat{\mathbf{A}}$ 的对角度矩阵。项 $\hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-1/2}$ 表示邻接矩阵的对称归一化 (normalization)。该层平均邻居节点（包括节点自身）的特征，然后应用线性变换，再进行非线性处理。

在 PyG 中，您使用 `torch_geometric.nn.GCNConv`：

```python
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training) # 经常使用 Dropout
        x = self.conv2(x, edge_index)

        # 对于节点分类，通常使用 LogSoftmax
        return F.log_softmax(x, dim=1)
```

#### GraphSAGE

GraphSAGE (Hamilton et al., 2017) 专注于学习聚合函数，而不是固定的卷积。它被设计为归纳式的，这意味着它可以在推断时推广到未见过的节点。GraphSAGE 为每个节点采样固定大小的邻域，然后使用平均、最大值或 LSTM 池化等函数聚合邻居特征。

主要步骤包括：

1. 为节点 $i$ 采样一个邻域 $\mathcal{N}(i)$。
2. 聚合邻居特征：$\mathbf{a}_{\mathcal{N}(i)}^{(l)} = \text{AGGREGATE}^{(l)}(\{\mathbf{h}_j^{(l-1)} \mid j \in \mathcal{N}(i)\})$
3. 更新节点 $i$ 的嵌入：$\mathbf{h}_i^{(l)} = \sigma\left( \mathbf{W}^{(l)} \cdot \text{CONCAT}(\mathbf{h}_i^{(l-1)}, \mathbf{a}_{\mathcal{N}(i)}^{(l)}) \right)$

PyG 使用 `torch_geometric.nn.SAGEConv` 实现此功能：

```python
from torch_geometric.nn import SAGEConv

class SimpleGraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super().__init__()
        # 默认聚合器是 'mean'
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
```

在创建 `SAGEConv` 层时，您可以指定聚合器类型（例如，`aggr='max'`、`aggr='mean'`）。

#### 图注意力网络 (GAT)

GAT 层 (Veličković et al., 2018) 引入了注意力机制 (attention mechanism)，使得节点在聚合过程中可以为其邻居分配不同的重要性权重。这使得聚合过程更加灵活，并通常带来更好的性能。

节点 $i$ 和邻居 $j$ 之间的注意力系数 $e_{ij}$ 基于它们的特征计算，通常使用共享的线性变换和一个注意力机制（例如，一个单层前馈网络）：


$$
e_{ij} = \text{attention}(\mathbf{W}^{(l)}\mathbf{h}_i^{(l-1)}, \mathbf{W}^{(l)}\mathbf{h}_j^{(l-1)})
$$


然后，这些系数使用 softmax 函数对节点 $i$ 的所有邻居进行归一化：


$$
\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}
$$


聚合后的消息是转换后的邻居特征的加权和：


$$
\mathbf{a}_i^{(l)} = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}^{(l)} \mathbf{h}_j^{(l-1)}
$$


更新步骤将此聚合消息与节点自身的特征结合，通常使用拼接后跟激活函数：


$$
\mathbf{h}_i^{(l)} = \sigma\left( \mathbf{a}_i^{(l)} \right) \quad \text{or} \quad \mathbf{h}_i^{(l)} = \sigma\left( \text{CONCAT}(\mathbf{h}_i^{(l-1)}, \mathbf{a}_i^{(l)}) \right)
$$


GAT 经常使用多头注意力 (multi-head attention)，其中计算多个独立的注意力机制，并将其结果进行拼接或平均。

PyG 使用 `torch_geometric.nn.GATConv` 实现此功能：

```python
from torch_geometric.nn import GATConv

class SimpleGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels, heads=8):
        super().__init__()
        # 在第一层中使用多头注意力
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=0.6)
        # 多头注意力的输出特征为 heads * hidden_channels
        # 对于最后一层，通常平均各头或使用单头
        self.conv2 = GATConv(hidden_channels * heads, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training) # 对输入特征应用 Dropout
        x = self.conv1(x, edge_index)
        x = F.elu(x) # ELU 激活在 GAT 中很常见
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
```

### 构建和训练 GNN 模型

使用 PyG 层在 PyTorch 中构建 GNN 遵循标准的 PyTorch 实践。您定义一个继承自 `torch.nn.Module` 的类，在 `__init__` 中初始化 PyG 层，并在 `forward` 中定义前向传播逻辑。`forward` 方法通常接受 `Data` 或 `Batch` 对象作为输入，并提取 `x`、`edge_index`，以及可能的 `edge_attr` 和 `batch` 索引。

训练循环也类似于标准的 PyTorch 循环：遍历 `DataLoader`，执行前向传播，计算损失（例如，用于节点分类的 `F.nll_loss`，配合 `log_softmax`），使用 `loss.backward()` 计算梯度，并使用优化器更新参数 (parameter)。

### 常见的 GNN 应用

GNN 功能多样，可应用于各种图相关任务：

1. **节点分类：** 预测图中单个节点的标签或属性（例如，对社交网络中的用户进行分类，预测蛋白质功能）。上述示例 (`SimpleGCN`、`SimpleGraphSAGE`、`SimpleGAT`) 均适用于节点分类。
2. **图分类：** 预测整个图的标签或属性（例如，将分子分类为有毒或无毒，对社交群体进行分类）。这需要在 GNN 层之后使用图池化层（例如，`torch_geometric.nn.global_mean_pool`、`global_max_pool`）将节点嵌入 (embedding)聚合成单个图嵌入。
3. **链接预测：** 预测两个节点之间是否存在或将存在边（例如，在社交网络中推荐朋友，预测蛋白质-蛋白质相互作用）。这通常包括学习节点嵌入，然后使用评分函数（例如，点积）对节点嵌入对进行处理。
4. **图生成：** 生成具有期望属性的新图。
5. **社区检测：** 在大型图中识别连接紧密的节点群组。

PyTorch Geometric 为应对这些任务提供了全面的工具。通过结合其优化层、数据处理工具和标准 PyTorch 功能，您可以有效地构建和训练处理复杂图问题的精巧 GNN 模型。请记住，选择合适的 GNN 结构（GCN、GAT、SAGE 或其他）通常取决于您的图数据的具体特征和当前的任务。进行实验和理解每个层的基本原理对于成功应用非常重要。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  本文介绍了图卷积网络（GCN）架构，它是本节讨论的核心GNN模型。
- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) — William L. Hamilton, Rex Ying, and Jure Leskovec (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1706.02216](https://doi.org/10.48550/arXiv.1706.02216)
  这篇论文介绍了GraphSAGE框架，该框架侧重于学习图上的归纳表示学习的聚合函数。
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) — Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)
  这篇基础论文介绍了图注意力网络（GAT）架构，该架构利用注意力机制对邻居信息进行加权。
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/) — PyTorch Geometric Team (2023)
  PyTorch Geometric的官方文档，提供了GNN实现中数据结构、层和实用程序的详细信息。

---

[上一节](02-%E9%AB%98%E7%BA%A7%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md) · [下一节](04-%E7%94%A8%E4%BA%8E%E7%94%9F%E6%88%90%E5%BB%BA%E6%A8%A1%E7%9A%84%E5%BD%92%E4%B8%80%E5%8C%96%E6%B5%81.md)
