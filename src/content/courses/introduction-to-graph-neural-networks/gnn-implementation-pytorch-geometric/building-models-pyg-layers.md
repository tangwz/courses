---
course: "introduction-to-graph-neural-networks"
chapter: "gnn-implementation-pytorch-geometric"
lesson: "building-models-pyg-layers"
sourceId: 7644
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-5-gnn-implementation-pytorch-geometric/building-models-pyg-layers"
title: "使用 PyG GNN 层构建模型"
description: "学习如何使用 PyG 的模块化 GNN 层（如 GCNConv 和 GATConv）快速构建模型。"
order: 4
plots: []
sourceHash: "97b6db8084363af6204b58e8ef1ca6eebf3ab20dac7db7b60bca5215f000a31e"
sourceCorrections: []
---

在 PyTorch Geometric 中构建基于图的模型涉及组装其核心组件。这一过程与在 PyTorch 中构建标准神经网络 (neural network)非常相似，因为它使用了相同的 `torch.nn.Module` 类结构。不同于使用 `nn.Linear` 或 `nn.Conv2d` 等层，这里使用的是 PyG 提供的专门 GNN 层。

### GNN 作为 PyTorch 模块

PyG 中的 GNN 模型是一个继承自 `torch.nn.Module` 的 Python 类。网络的层在 `__init__` 方法中定义，而前向传播逻辑（描述数据如何流经网络）在 `forward` 方法中实现。这种结构为定义各种架构提供了一种熟悉且有条理的方式。

主要区别在于 `forward` 方法的签名。虽然标准前馈网络可能只需要特征张量 `x`，但 GNN 还需要图结构来执行消息传递。因此，`forward` 方法通常接收节点特征 `x` 和 `edge_index` 张量。

### 了解 PyG 的 GNN 层

PyTorch Geometric 在其 `torch_geometric.nn` 模块中提供了丰富的 GNN 层。这些是我们在第三章中讨论的架构的高效实现。让我们看看如何使用其中一些基础的层。

#### GCNConv

`GCNConv` 层实现了图卷积网络（Graph Convolutional Network）算子。它的构造函数很直接，需要每个节点的输入特征数量和期望的输出特征数量。

`gnn.GCNConv(in_channels: int, out_channels: int)`

- `in_channels`：输入节点特征的维度。对于第一层，这是数据集中每个节点拥有的特征数。
- `out_channels`：该层生成的节点嵌入 (embedding)的维度。

#### SAGEConv

`SAGEConv` 层实现了 GraphSAGE 算子。它遵循类似的模式，但在聚合方案上提供了更多灵活性。

`gnn.SAGEConv(in_channels: int, out_channels: int, aggr: str = 'mean')`

- `in_channels` 和 `out_channels` 与 `GCNConv` 中的含义相同。
- `aggr`：指定所用聚合方法的字符串，例如 `'mean'`、`'max'` 或 `'add'`。

#### GATConv

`GATConv` 层实现了图注意力网络（Graph Attention Network）算子。该层引入了注意力机制 (attention mechanism)，允许节点对邻居的权重 (weight)进行衡量。

`gnn.GATConv(in_channels: int, out_channels: int, heads: int = 1, concat: bool = True)`

- `heads`：使用的并行注意力机制或“头”的数量。多头注意力 (multi-head attention)通常能使学习过程更稳定。
- `concat`：如果设置为 `True`，则将多个注意力头的嵌入拼接起来，导致输出特征维度为 `heads * out_channels`。如果为 `False`，则对它们取平均值。

### 构建双层 GCN 模型

让我们将这些组件组合起来，构建一个用于节点分类的 GNN。我们的模型将由两个 `GCNConv` 层组成。第一层将初始节点特征转换为中间隐藏表示，第二层将该隐藏表示转换为最终输出，输出维度对应于类别数量。我们将在各层之间使用 ReLU 激活函数 (activation function)来引入非线性。

以下是模型的完整定义：

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 1. 第一个 GCN 层 + ReLU 激活
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # 可选：添加 dropout 以进行正则化
        # x = F.dropout(x, p=0.5, training=self.training)

        # 2. 第二个 GCN 层
        x = self.conv2(x, edge_index)

        return x
```

让我们分解 `forward` 方法：

1. 初始节点特征 `x` 和图结构 `edge_index` 被传递给第一个卷积层 `self.conv1`。
2. 输出是一组维度为 `hidden_channels` 的新节点嵌入 (embedding)。
3. 对这些嵌入逐元素应用 ReLU 激活函数。
4. 激活后的嵌入随后被传递给第二个卷积层 `self.conv2`，该层产生大小为 `out_channels` 的最终嵌入。对于分类任务，`out_channels` 将等于类别数量。
5. 模型返回这些最终嵌入，通常被称为“logits”。稍后通常会在损失计算中对这些 logits 应用 softmax 函数。

> 我们的双层 GCN 架构图。节点特征 `x` 和图结构 `edge_index` 被送入第一个 GCN 层。生成的嵌入在进入第二个 GCN 层之前经过 ReLU 激活，以产生最终的输出 logits。

### 模块化的优势

PyG 的模块化设计使得尝试不同架构变得简单。想看看 GraphSAGE 模型是否表现更好？你只需要在 `__init__` 方法中更换层类型即可。`forward` 传递逻辑保持不变。

例如，要将我们的 GCN 更改为 GraphSAGE 模型，我们可以进行以下修改：

```python
# from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # 将 GCNConv 替换为 SAGEConv
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
```

这种灵活性是使用像 PyTorch Geometric 这样专门库的显着优点。它让你能够专注于高层模型设计和迭代，而不是每一层的底层数学实现。定义好模型结构后，下一步就是编写脚本来为模型提供数据并训练其参数 (parameter)。

## 参考资料

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/) — Matthias Fey and Jan E. Lenssen (2024)
  PyTorch Geometric的官方文档，提供了GNN层和模块构建的API详情。
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍了图卷积网络（GCN）架构，是图卷积领域的奠基性论文。
- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) — William L. Hamilton, Rex Ying, Jure Leskovec (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1706.02216](https://doi.org/10.48550/arXiv.1706.02216)
  提出了GraphSAGE，一个通过采样和聚合节点局部邻域特征来生成节点嵌入的归纳框架。
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) — Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)
  介绍了图注意力网络（GAT），它利用注意力机制聚合邻域信息。
