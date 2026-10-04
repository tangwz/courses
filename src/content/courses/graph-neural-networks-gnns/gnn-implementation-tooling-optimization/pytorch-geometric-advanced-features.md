---
course: "graph-neural-networks-gnns"
chapter: "gnn-implementation-tooling-optimization"
lesson: "pytorch-geometric-advanced-features"
sourceId: 2747
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-5-gnn-implementation-tooling-optimization/pytorch-geometric-advanced-features"
title: "PyTorch Geometric (PyG) 高级功能"
description: "使用 PyG 的高级能力，包括数据集、变换和优化的 GNN 层。"
order: 2
plots: []
sourceHash: "fb8217d5cea98a5f800f82a97e24645b27014e900cfc35225139d828f0f792dd"
sourceCorrections: []
---

PyTorch Geometric (PyG) 提供实现图神经网络 (neural network)的框架，通过专用工具扩展 PyTorch 以处理图数据。PyG 提供一系列高级功能，这些功能扩展并提升了定义层和处理 `Data` 对象等基础操作。这些功能旨在简化复杂模型的开发，高效处理大型数据集，并支持多样化的图结构。掌握这些功能对于构建高性能、研究级别的 GNN 非常重要。

### 高级数据处理和数据集

PyG 通过 `torch_geometric.datasets` 简化对各种基准图数据集的访问。除了像 Cora 或 CiteSeer 这样的标准数据集，它还包括用于大型图（例如，来自 Open Graph Benchmark 的 ogbn-arxiv）、社交网络、分子数据集等的加载器。许多数据集支持延迟加载，这意味着它们不会一次性将整个图加载到内存中，这对于处理大规模图非常必要。

```python
# 示例：加载大型 OGB 数据集
from torch_geometric.datasets import Planetoid, OGB_MAG
from torch_geometric.transforms import ToUndirected, NormalizeFeatures

# 带有变换的标准数据集
dataset_cora = Planetoid(root='/tmp/Cora', name='Cora', 
                         transform=NormalizeFeatures())
data_cora = dataset_cora[0]

# 大型异构数据集（需要 ogb 包）
# dataset_ogb = OGB_MAG(root='/tmp/OGB_MAG', preprocess='metapath2vec',
#                       transform=ToUndirected())
# hetero_data_ogb = dataset_ogb[0] 
# print(hetero_data_ogb) # 示例输出结构
```

> OGB\_MAG 示例显示了加载一个大型异构图。请注意，处理此类大型数据集通常需要大量的计算资源。

PyG 提供用于创建适合各种图学习任务的数据集划分的工具。`torch_geometric.transforms.RandomNodeSplit` 和 `torch_geometric.transforms.RandomLinkSplit` 是分别生成节点分类的训练、验证和测试掩码或链接预测的分区的有效工具。它们提供转导式和归纳式设置的选项。

对于自定义数据集，您可以继承自 `torch_geometric.data.Dataset` 或 `torch_geometric.data.InMemoryDataset` 以实现您自己的加载和处理逻辑，并与 PyG 的生态系统集成。

### 有效的数据变换

变换 (`torch_geometric.transforms`) 是应用于 `Data` 或 `HeteroData` 对象的功能，在它们被传递给模型或保存*之前*执行。它们通常用于预处理或数据增强。PyG 提供丰富的变换集合：

- **几何变换：** `AddSelfLoops`、`ToUndirected`、`RemoveIsolatedNodes`、`Cartesian`、`LocalCartesian`、`KNNGraph`。
- **特征变换：** `NormalizeFeatures`、`AddLaplacianEigenvectorPE`（位置编码 (positional encoding)）、`AddRandomWalkPE`。
- **格式转换：** `ToSparseTensor`（将边索引转换为 `torch_sparse.SparseTensor`，通常可以提升性能）、`ToDense`。
- **划分：** `RandomNodeSplit`、`RandomLinkSplit`。

变换可以通过 `torch_geometric.transforms.Compose` 进行组合。

```python
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

# 组合变换的示例
transform = T.Compose([
    T.NormalizeFeatures(),
    T.AddSelfLoops(),
    T.ToSparseTensor() 
])

dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
data = dataset[0]

# 访问稀疏邻接矩阵
# adj_t = data.adj_t 
# print(adj_t)
```

> 使用 `ToSparseTensor` 可以通过利用优化的稀疏矩阵乘法例程，显著加速许多 GNN 层中的计算。

### 使用 DataLoader 进行高效小批量处理

高效处理图或子图的批次非常重要。PyG 的 `DataLoader`（来自 `torch_geometric.loader`）智能地将多个 `Data` 对象批量处理成一个巨大的图（`torch_geometric.data.Batch` 对象），其中包含不连通的子图。它自动调整节点索引，并提供一个 `batch` 属性，将批次内的每个节点映射到其原始图索引。这种整理过程对于处理各种大小的图非常高效。

```python
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    print(batch) 
    # 输出：Batch(batch=[num_nodes_in_batch], x=[num_nodes_in_batch, num_node_features], 
    #             edge_index=[2, num_edges_in_batch], y=[batch_size])
    print(batch.num_graphs) 
    # 输出：32（或最后一批次更少）
```

对于全图训练不可行的大型图，PyG 提供实现邻域采样或聚类的专用数据加载器：

- **`NeighborLoader`：** 执行逐层邻域采样，创建适合训练 GraphSAGE 等模型的小批量数据。它为批次中每个层中的每个节点采样固定数量的邻居。
- **`LinkNeighborLoader`：** 类似于 `NeighborLoader`，但专为链接预测任务设计。它采样节点对（正边和负边）及其计算邻域。
- **`ClusterLoader`：** 通过将图划分为子图（簇）并加载这些子图的批次来实现 Cluster-GCN 算法。
- **`GraphSAINTLoader`：** 实现 GraphSAINT 论文中的各种图采样技术（例如，节点、边、随机游走采样器）。

这些加载器处理采样、子图创建和批量处理的复杂性，使您能够将 GNN 应用于大规模数据集。

```python
from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

# 假设 'data' 是一个大型 Data 对象（例如，来自 OGB）
# data = ... 

# 示例：为节点分类设置 NeighborLoader
train_loader = NeighborLoader(
    data,
    # 为第一层采样 15 个邻居，为第二层采样 10 个
    num_neighbors=[15, 10], 
    batch_size=128,
    input_nodes=data.train_mask, # 要采样的节点
    shuffle=True
)

# 迭代采样的迷你批次（子图）
# for batch in train_loader:
#    # batch 是一个较小的 Data 对象，表示采样的计算图
#    # model(batch.x, batch.edge_index) 
#    pass 
```

### 原生异构图支持

PyG 通过 `HeteroData` 对象为异构图（具有多种节点和边类型的图）提供一流支持。`HeteroData` 对象为每种类型单独存储节点特征、边索引和边特征。节点类型由字符串标识（例如，`'author'`、`'paper'`），边类型表示为元组 `('source_node_type', 'relation_type', 'destination_node_type')`，例如 `('author', 'writes', 'paper')`。

```python
from torch_geometric.data import HeteroData

# 示例：创建 HeteroData 对象
data = HeteroData()

# 节点特征
data['paper'].x = torch.randn(num_papers, paper_features)
data['author'].x = torch.randn(num_authors, author_features)

# 边索引（注意边类型的元组表示法）
data['author', 'writes', 'paper'].edge_index = # shape [2, num_write_edges]
data['paper', 'cites', 'paper'].edge_index = # shape [2, num_cite_edges]

# 可选的边特征
data['author', 'writes', 'paper'].edge_attr = torch.randn(num_write_edges, edge_features)

print(data)
# 示例输出：
# HeteroData(
#  paper={ x=[num_papers, paper_features] },
#  author={ x=[num_authors, author_features] },
#  (author, writes, paper)={ edge_index=[2, num_write_edges], edge_attr=[num_write_edges, edge_features] },
#  (paper, cites, paper)={ edge_index=[2, num_cite_edges] }
#)
```

PyG 提供用于异构图的专用层，其中最著名的是 `HeteroConv`。`HeteroConv` 充当一个包装器，将不同的 GNN 层（由您指定）应用于图中的不同边类型。它自动处理不同关系类型之间的消息传递和聚合。其他专用层，如 `HGTConv`（异构图 Transformer），也可用。

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, HeteroConv

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('paper', 'cites', 'paper'): SAGEConv((-1, -1), hidden_channels),
                ('author', 'writes', 'paper'): GCNConv(-1, hidden_channels),
                ('paper', 'rev_writes', 'author'): GCNConv(-1, hidden_channels), 
                # 根据需要添加其他边类型
            }, aggr='sum') # 聚合来自不同边类型的结果
            self.convs.append(conv)

        # 示例输出层（根据任务调整）
        self.lin = torch.nn.Linear(hidden_channels, out_channels) 

    def forward(self, x_dict, edge_index_dict):
        # x_dict: {'paper': tensor, 'author': tensor}
        # edge_index_dict: {('paper','cites','paper'): tensor, ...}

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()} # 对每种节点类型应用激活函数

        # 示例：返回用于节点分类的论文嵌入
        return self.lin(x_dict['paper']) 

# 示例用法（假设模型已定义）
# model = HeteroGNN(...)
# out = model(data.x_dict, data.edge_index_dict) 
```

> 本示例呈现了如何定义一个 `HeteroConv` 层，该层根据边类型应用不同的卷积（`SAGEConv`、`GCNConv`）。请注意，输入/输出特征大小通常可以使用 `-1` 来推断。我们还添加了一个反向边类型 `('paper', 'rev_writes', 'author')`，这可能取决于所需的消息传递方向。在 `HeteroData` 对象上使用 `T.ToUndirected(merge=False)` 变换通常可以自动化添加反向边。

### 与优化后端集成：`torch-sparse` 和 `torch-scatter`

在内部，PyG 使用高度优化的库：

- **`torch-sparse`：** 在 GPU 和 CPU 上提供稀疏矩阵操作（如 SpMM - 稀疏矩阵乘法）的高效实现。许多 PyG 层在操作 `SparseTensor` 邻接格式（通过 `T.ToSparseTensor()` 获得）时使用 `torch-sparse`。
- **`torch-scatter`：** 提供用于 scatter 操作（scatter\_add、scatter\_mean、scatter\_max 等）的优化例程，这些对于消息传递 GNN 中的聚合步骤是基本的。

虽然您可能不经常直接与这些库交互，但了解它们的作用有助于编写高性能代码，并认识到 PyG 相较于朴素实现所带来的效率提升。使用 `SparseTensor` 输入等功能通常会隐式调用这些优化后端。

通过使用 PyG 的高级数据集、变换、数据加载器（特别是用于采样的）和异构图能力，并依赖其优化的后端，您可以构建和训练复杂的 GNN 模型，这些模型可以扩展到研究和工业中遇到的复杂、大规模图问题。

## 参考资料

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/) — Matthias Fey, Jan E. Lenssen, et al. (2024)
  Publisher: PyTorch Geometric Contributors
  PyTorch Geometric的官方文档，提供了其功能的详细说明、API参考和示例，包括高级数据处理、转换和加载器。
- [Open Graph Benchmark: Datasets for Machine Learning on Graphs](https://arxiv.org/abs/2005.00687) — Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, Jure Leskovec (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 33; Publisher: NeurIPS; DOI: [10.48550/arXiv.2005.00687](https://doi.org/10.48550/arXiv.2005.00687)
  介绍了开放图基准（Open Graph Benchmark），一个包含挑战性、大规模图数据集的集合，包括文中提及的数据集（如ogbn-arxiv, OGB_MAG），用于GNN的开发和评估。
- [Inductive Representation Learning on Large Graphs](https://papers.nips.cc/paper/2017/file/5ddf0048ac127e77d24dc0f195bc7648-Paper.pdf) — William L. Hamilton, Rex Ying, Jure Leskovec (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Neural Information Processing Systems; Volume: 30; Pages: 1024-1034; DOI: [10.48550/arXiv.1706.02216](https://doi.org/10.48550/arXiv.1706.02216)
  介绍了GraphSAGE，一个通过采样和聚合节点局部邻域特征来生成节点嵌入的归纳框架，是PyG中NeighborLoader的基础。
- [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332) — Ziniu Hu, Yuxiao Dong, Kuansan Wang, Yizhou Sun (2020)
  Journal: The Web Conference (WWW); DOI: [10.48550/arXiv.2003.01332](https://doi.org/10.48550/arXiv.2003.01332)
  介绍了异构图Transformer (HGT)，一种用于异构图的GNN架构，与PyG的HGTConv层直接相关。
