# 实践：应用可扩展GNN训练

来源：[原文](https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-3-gnn-training-complexities/practice-scalable-gnn-training)

[返回章节目录](README.md) · [返回课程目录](../README.md)

一个动手练习将实现并训练在大规模图数据集上的图神经网络 (neural network) (GNN)。针对大规模图固有的可扩展性挑战，实践中应用了邻域采样或图聚类等技术。此实践旨在揭示将GNN应用于网络规模数据时，实际的影响、权衡和必要的调整，从而有效部署GNN并超越小型图的局限。

我们假设您熟悉PyTorch Geometric (PyG) 或 Deep Graph Library (DGL) 中的基本GNN模型定义（如GCN或GraphSAGE）和标准训练循环。本练习特别侧重于整合可扩展的数据加载和训练策略。

### 准备工作：数据集和框架选择

首先，选择一个合适的大规模图数据集。像Open Graph Benchmark的`ogbn-products`或`ogbn-arxiv`，或者像`Reddit`这样的数据集，都是很好的选择。这些图通常拥有数百万个节点和边，使得在标准硬件上进行全批次训练不可行。

```python
# 使用PyG加载ogbn-products的示例
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

# 加载数据集
dataset = PygNodePropPredDataset(name='ogbn-products', root='./dataset/')
split_idx = dataset.get_idx_split()
data = dataset[0]

# 如果需要（例如用于标签传播），预计算节点特征
# data = T.ToSparseTensor()(data) # 可选：如果偏好，转换为SparseTensor格式

print(f'Dataset: {dataset.name}')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of features: {data.num_node_features}')
print(f'Number of classes: {dataset.num_classes}')
```

选择您偏好的库，PyG或DGL，因为两者都提供了可扩展训练方法的实现。我们将阐述适用于两者的思路，主要使用PyG的API提供代码片段以保持简洁，但会在适用时注明DGL的对应部分。

### 方法一：邻域采样 (GraphSAGE风格)

邻域采样通过处理节点的小批次，并且仅在采样邻域上执行消息传递，而非整个图，从而应对可扩展性问题。这使得每个批次的计算图保持小巧且易于管理。

**使用 `NeighborLoader` (PyG) 实现：**

PyG的`NeighborLoader`（或旧版本/DGL中的`NeighborSampler`）自动处理采样过程。您定义加载器，指定每层要采样的邻居数量。

```python
# PyG示例
from torch_geometric.loader import NeighborLoader

# 定义NeighborLoader
train_loader = NeighborLoader(
    data,                                  # 完整的图Data对象
    num_neighbors=[15, 10],                # 第1层采样15个邻居，第2层采样10个
    batch_size=1024,                       # 小批次大小（目标节点数量）
    input_nodes=split_idx['train'],        # 用于采样目标节点的节点（训练节点）
    shuffle=True,                          # 每个epoch打乱节点
    num_workers=4                          # 数据加载的子进程数量
)

# 在DGL中，设置涉及创建图对象，然后使用
# dgl.dataloading.NeighborSampler 类似地使用。
```

**参数 (parameter)：**

- `num_neighbors`: 一个列表，指定每个GNN层（从最外层到最内层）要采样的邻居数量。较小的数字意味着更快的计算和更少的内存，但可能导致更高的采样方差和信息丢失。较大的数字会增加成本但可能提高准确性。这是一个重要的超参数 (hyperparameter)，需要调整。
- `batch_size`: 每次迭代中计算嵌入 (embedding)的目标节点数量。这直接影响GPU内存使用。
- `input_nodes`: 指定从中抽取`batch_size`个目标节点的节点集合（例如，训练节点）。

**训练循环修改：**

训练循环结构保持相似，但GNN模型现在在`NeighborLoader`生成的`batch`对象上操作。该对象表示一个子图，包含目标节点及其采样的多跳邻域。

```python
# 使用NeighborLoader的训练循环示例片段
model = YourGNNModel(...) # 定义您的GNN（例如GraphSAGE）
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def train():
    model.train()
    total_loss = total_examples = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # 模型直接在采样的子图（batch）上操作
        # 注意：输出大小与NeighborLoader中指定的batch_size匹配
        out = model(batch.x, batch.edge_index, size=batch.size())[:batch.batch_size]
        # 获取目标节点的真实标签
        y = batch.y[:batch.batch_size].view(-1).long()
        loss = F.nll_loss(out, y) # 假设分类使用NLLLoss
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_examples += batch.batch_size
    return total_loss / total_examples

# --- 评估通常需要为验证/测试节点使用单独的加载器 ---
# 通常，评估是逐层进行的，以避免内存占用过大，
# 或者使用shuffle=False的NeighborLoader。
```

> 注意：`model`的前向传播接收采样子图的特征（`batch.x`）和邻接信息（`batch.edge_index`）。输出`out`仅对应于小批次中包含的`batch_size`个目标节点，而非采样子图中存在的所有节点。

### 方法二：图聚类 (Cluster-GCN)

Cluster-GCN采用一种不同的方法。它首先使用图聚类算法（如METIS）将图的节点划分成簇。训练随后以小批次进行，每个批次包含一个或多个簇。GNN在为该批次选择的簇内节点所形成的子图上操作。

**使用 `ClusterLoader` (PyG) 实现：**

PyG的`ClusterLoader`处理聚类（如果未预计算）和簇的批处理。

```python
# PyG示例
from torch_geometric.loader import ClusterData, ClusterLoader

# 1. 执行图聚类（预处理步骤）
# 这将图数据划分为num_parts个簇
cluster_data = ClusterData(data, num_parts=1500, recursive=False, save_dir=dataset.processed_dir)

# 2. 创建ClusterLoader
# 每个批次将包含由'batch_size'个簇形成的子图
train_loader = ClusterLoader(
    cluster_data,
    batch_size=32, # 每个批次中的簇数量
    shuffle=True,
    num_workers=4
)

# DGL提供了类似的功能，通常首先需要显式分区
# 使用METIS等库，然后创建特定的采样器。
```

**参数 (parameter)：**

- `num_parts`: 将图划分成的簇的总数。更多簇意味着每个批次的子图更小，但也可能在簇之间切割更多边。
- `batch_size`: 组合形成单个小批次的簇数量。

**训练循环修改：**

训练循环遍历`ClusterLoader`提供的批次。每个`batch`对象是表示采样子簇中节点形成的子图的标准`Data`对象。

```python
# 使用ClusterLoader的训练循环示例片段
model = YourGNNModel(...) # GNN模型（例如GCN, GAT）
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def train():
    model.train()
    total_loss = total_examples = 0
    for batch in train_loader: # 遍历簇批次
        batch = batch.to(device)
        optimizer.zero_grad()
        # 模型在当前簇批次定义的子图上操作
        out = model(batch.x, batch.edge_index)
        y = batch.y.view(-1).long()
        loss = F.nll_loss(out, y) # 损失仅在批次内的节点上计算
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.num_nodes
        total_examples += batch.num_nodes
    return total_loss / total_examples

# --- 评估通常使用整个图或单独的ClusterLoader ---
# 用于验证/测试集。Cluster-GCN评估可以通过迭代所有簇批次来近似
# 整个图的性能。
```

**区别：**

以下是对比这两种方法的一个简单示意图：

> 邻域采样和Cluster-GCN的流程图。采样侧重于目标节点的自我网络，而聚类则首先对整个图进行划分。

### 运行实验与分析

1. \*\*实现：\*\*选择一种方法（邻域采样或Cluster-GCN），并使用您选择的GNN架构，按照上文所示实现数据加载和训练循环。
2. \*\*训练：\*\*对模型进行合理数量的epoch训练。监控：
   - \*\*GPU内存使用：\*\*使用`nvidia-smi`等工具。与尝试全批次加载（如果您尝试过）相比如何？像`batch_size`和`num_neighbors`（用于采样）或`num_parts`（用于聚类）这样的参数 (parameter)如何影响内存？
   - \*\*每个Epoch的时间：\*\*训练需要多长时间？将其与估计的全批次时间进行比较。
   - \*\*训练损失/准确率：\*\*监控收敛情况。
3. \*\*评估：\*\*实现一个评估函数。对于邻域采样，推理 (inference)通常需要仔细实现，以计算所有节点的嵌入 (embedding)（可能逐层进行或使用不打乱的`NeighborLoader`）。对于Cluster-GCN，评估有时可以通过对所有簇批次运行推理来近似。计算验证集和测试集上的最终准确率。
4. \*\*比较（可选）：\*\*如果可能，尝试实现*另一种*可扩展方法。性能（准确率、速度、内存）和实现复杂度如何比较？调整您选择的方法的超参数 (hyperparameter)（`num_neighbors`、`num_parts`、`batch_size`），并观察其影响。

### 总结思考

“这个实践练习表明，使用正确的技术，在大规模图上训练GNN是可行的。邻域采样通过控制每个节点的计算图大小提供灵活性，而Cluster-GCN则通过预分区运用图结构。两种方法相较于全批次训练都引入了近似，导致在可扩展性、速度、内存使用和最终模型性能之间存在权衡。理解如何实现、调整和评估这些可扩展策略，对于将GNN应用于许多涉及大规模图数据集的重要问题非常重要。”

## 参考资料

- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) — William L. Hamilton, Rex Ying, Jure Leskovec (2017)
  Journal: NIPS 2017; DOI: [10.48550/arXiv.1706.02216](https://doi.org/10.48550/arXiv.1706.02216)
  介绍了GraphSAGE，通过邻域采样实现归纳学习和可扩展GNN训练的开创性工作。
- [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://doi.org/10.1145/3292500.3330925) — Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, Cho-Jui Hsieh (2019)
  Journal: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD); Publisher: ACM; Pages: 257-266; DOI: [10.1145/3292500.3330925](https://doi.org/10.1145/3292500.3330925)
  提出了Cluster-GCN，一种将大型图划分为簇以进行高效GNN训练的算法。
- [PyTorch Geometric Documentation: Scalable Graph Data Loading](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html) — PyTorch Geometric Developers (2024)
  Publisher: PyTorch Geometric
  PyG可扩展数据加载器的官方文档，包括`NeighborLoader`和`ClusterLoader`，对于实际实现至关重要。
- [A Survey on Large-Scale Graph Neural Networks](https://ieeexplore.ieee.org/document/9768276) — Jingtun Zhang, Liang Zhao, Yu Rong, Donghan Yu, Wei Xu, Yunsheng Jiang, Haishuai Wang, Junzhou Huang (2022)
  Journal: IEEE Transactions on Knowledge and Data Engineering; Publisher: IEEE; Volume: 45; Pages: 3260-3283; DOI: [10.1109/TKDE.2022.3168276](https://doi.org/10.1109/TKDE.2022.3168276)
  关于在大型图上训练GNN的可扩展性挑战和各种技术的全面调查，涵盖了广泛的方法。

---

[上一节](08-%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E4%BC%98%E5%8C%96%E7%AD%96%E7%95%A5.md) · [下一节](../04-%E9%AB%98%E7%BA%A7GNN%E4%BB%BB%E5%8A%A1%E4%B8%8E%E6%8A%80%E6%9C%AF/01-%E5%A4%84%E7%90%86%E5%BC%82%E6%9E%84%E5%9B%BE%20%28RGCN%2C%20HAN%29.md)
