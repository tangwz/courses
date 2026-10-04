---
course: "advanced-pytorch"
chapter: "advanced-network-architectures"
lesson: "practice-custom-gnn-layer"
sourceId: 2704
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-2-advanced-network-architectures/practice-custom-gnn-layer"
title: "实践：实现自定义GNN层"
description: "从头开始为图神经网络构建自定义消息传递层。"
order: 7
plots: []
sourceHash: "0c6a96dfa8b395caabfa3d6a76478a9ea727fe2be8f53a067805e9517ef7fcf7"
sourceCorrections: []
---

PyTorch Geometric等库提供强大的预构建层来构建图神经网络 (neural network)（GNN）。虽然这些层非常有用，但了解如何使用核心PyTorch操作从头开始构建GNN层，能提供更全面的理解，并具备实现新颖或定制化消息传递方案的灵活性。本次实践练习将指导你创建一个简单的自定义GNN层。

许多GNN层背后的基本思想是**消息传递**，即节点迭代地从邻居节点聚合信息并更新自身的表示。我们可以将其分解为每个节点 $i$ 的两个主要步骤：

1. **聚合：** 从邻居节点 $j \in \mathcal{N}(i)$ 收集特征或“消息”。
2. **更新：** 将聚合的信息与节点当前的特征向量 (vector) $h_i$ 结合，以生成更新后的特征向量 $h'_i$。

让我们实现一个执行这些步骤的基本层。我们将定义一个层，它使用可学习的权重 (weight)矩阵转换节点特征，使用简单的求和从邻居聚合转换后的特征，然后应用激活函数 (activation function)。

从数学上讲，对于节点 $i$，此操作可以描述为：


$$
a_i = \sum_{j \in \mathcal{N}(i) \cup \{i\}} W h_j
$$


$$
h'_i = \sigma(a_i)
$$


这里，$h_j$ 表示节点 $j$ 的特征向量，$W$ 是一个可学习权重矩阵，$\mathcal{N}(i)$ 是节点 $i$ 的邻居集合，$\sigma$ 是一个非线性激活函数（如ReLU）。注意，我们将节点自身（$i$）也包含在聚合中，这通常被称为添加自环。这确保了节点原始特征在更新时得到考量。

### 设置自定义层

首先，请确保已导入PyTorch。我们将把自定义层定义为一个继承自 `torch.nn.Module` 的Python类。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGNNLayer(nn.Module):
    """
    一个实现消息传递的基本图神经网络层。
    Args:
        in_features (int): 每个输入节点特征向量的大小。
        out_features (int): 每个输出节点特征向量的大小。
    """
    def __init__(self, in_features, out_features):
        super(SimpleGNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 定义可学习的权重矩阵
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # 初始化权重（可选但通常是好的做法）
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, edge_index):
        """
        定义每次调用时执行的计算。
        Args:
            x (torch.Tensor): 节点特征张量，形状为 [num_nodes, in_features]。
            edge_index (torch.Tensor): COO格式的图连接信息，形状为 [2, num_edges]。
                                       edge_index[0] = 源节点，edge_index[1] = 目标节点。
        Returns:
            torch.Tensor: 更新后的节点特征张量，形状为 [num_nodes, out_features]。
        """
        num_nodes = x.size(0)

        # 1. 为edge_index表示的邻接矩阵添加自环
        # 创建节点索引张量 [0, 1, ..., num_nodes-1]
        self_loops = torch.arange(0, num_nodes, device=x.device).unsqueeze(0)
        self_loops = self_loops.repeat(2, 1) # 形状 [2, num_nodes]
        # 将原始边与自环拼接
        edge_index_with_self_loops = torch.cat([edge_index, self_loops], dim=1)

        # 提取源节点和目标节点索引
        row, col = edge_index_with_self_loops

        # 2. 线性变换节点特征
        x_transformed = self.linear(x) # 形状: [num_nodes, out_features]

        # 3. 聚合来自邻居（包括自身）的特征
        # 我们希望对每个目标节点（col）求和源节点（row）的特征
        # 使用零初始化输出张量
        aggregated_features = torch.zeros(num_nodes, self.out_features, device=x.device)

        # 使用 index_add_ 进行高效聚合（散列求和）
        # 将 x_transformed[row] 的元素添加到 aggregated_features 中由 col 指定的索引处
        # index_add_(维度, 索引张量, 要添加的张量)
        aggregated_features.index_add_(0, col, x_transformed[row])

        # 4. 应用最终激活函数（可选）
        # 在此示例中，我们使用ReLU
        output_features = F.relu(aggregated_features)

        return output_features

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features}, {self.out_features})'
```

### 理解实现

- **初始化 (`__init__`)**：我们定义一个 `nn.Linear` 层。此层将把可学习权重 (weight)变换 $W$ 应用于输入节点特征。为简单起见，我们设置 `bias=False`，这与一些GNN公式（如基本GCN）一致。使用 `nn.init.xavier_uniform_` 进行权重初始化有助于稳定训练。
- **前向传播 (`forward`)**：这是消息传递逻辑的所在。
  - **自环**：我们显式地将自环添加到 `edge_index`。这确保了在为节点聚合邻居特征时，节点自身的转换特征也包含在内。我们创建一个表示从每个节点到自身的边的边索引，并将其与原始 `edge_index` 拼接。
  - **特征变换**：我们同时对所有节点特征 `x` 应用线性变换 (`self.linear`)。
  - **聚合**：这是GNN的主要步骤。我们需要对每个目标节点 (`col`) 求和源节点 (`x_transformed[row]`) 的转换特征。`torch.index_add_` 是一种高效执行此“散列-求和”操作的方法。它接受要累积到的张量 (`aggregated_features`)、进行索引的维度（节点为 `0`）、要添加到的索引 (`col`，即目标节点），以及要添加的值 (`x_transformed[row]`，即源节点的转换特征）。
  - **激活**：最后，逐元素应用一个非线性激活函数 (activation function) (`F.relu`)。

这里有一个小型图可视化，用以显示 `edge_index` 格式和邻居的思想：

> 对于上面的图，一个可能的 `edge_index`（表示用于消息传递的有向边，假设无向原始边意味着消息双向传递）可能是：
> `tensor([[0, 0, 1, 2, 1, 2, 3, 3], [1, 2, 0, 0, 3, 3, 1, 2]])`。
> 第一行包含源节点，第二行包含目标节点。当为节点3聚合时，我们会查看来自源节点1和2的消息。

### 使用自定义层

现在，让我们看看如何使用这个 `SimpleGNNLayer`。我们需要一些示例节点特征和一个 `edge_index`。

```python
# 示例用法

# 定义图数据
num_nodes = 4
num_features = 8
out_layer_features = 16

# 节点特征（随机）
x = torch.randn(num_nodes, num_features)

# 边索引表示连接（例如，0->1, 0->2, 1->3, 2->3；对于无向图则反之）
edge_index = torch.tensor([
    [0, 0, 1, 2, 1, 2, 3, 3],  # 源节点
    [1, 2, 0, 0, 3, 3, 1, 2]   # 目标节点
], dtype=torch.long)

# 实例化层
gnn_layer = SimpleGNNLayer(in_features=num_features, out_features=out_layer_features)
print(f"已实例化层: {gnn_layer}")

# 将数据通过该层
output_node_features = gnn_layer(x, edge_index)

# 检查输出形状
print(f"\n输入节点特征形状: {x.shape}")
print(f"边索引形状: {edge_index.shape}")
print(f"输出节点特征形状: {output_node_features.shape}")

# 验证输出形状是否符合预期: [num_nodes, out_features]
assert output_node_features.shape == (num_nodes, out_layer_features)

print("\n数据已成功通过自定义GNN层。")
# 显示节点0的前几个输出特征
print(f"节点0的输出特征（前5维）: {output_node_features[0, :5].detach().numpy()}")
```

此示例展示了创建随机节点特征和示例 `edge_index`，实例化我们的 `SimpleGNNLayer`，并执行前向传播。输出形状 `[num_nodes, out_features]` 确认该层按预期运行，为每个节点根据其邻域生成新的嵌入 (embedding)。

### 潜在的扩展

这个简单的层可作为根本。你可以通过多种方式对其进行扩展：

1. **不同聚合方式**：将 `index_add_`（求和聚合）替换为平均或最大值聚合。平均聚合通常需要知道每个节点的度。
2. **边特征**：修改 `forward` 传播以接受和运用边特征，并可能在聚合前将其加入到消息计算中。
3. **标准化**：添加标准化步骤，例如GCN层中常见的对称标准化，这通常涉及节点度。
4. **偏置 (bias)项**：在 `nn.Linear` 层中包含一个偏置项，或在聚合后添加。
5. **多层堆叠**：堆叠这些层，可能加入标准化或跳跃连接，以构建更深的GNN模型。

构建这样的自定义层是一项很有价值的技能。它使你能够直接根据研究论文实现前沿GNN架构，或在必要时精确地根据问题需求定制消息传递方案。构建自定义 `nn.Module` 组件的这一相同原理，也适用于在本课程中实现的Transformer、归一化 (normalization)流或其他高级架构中的独特机制。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://doi.org/10.48550/arXiv.1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍了图卷积网络（GCN）及其消息传递机制，这是本节自定义层实现的基础模型。
- [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) — William L. Hamilton (2020)
  Journal: Synthesis Lectures on Artificial Intelligence and Machine Learning; Publisher: Morgan and Claypool; Volume: 14; Pages: 1-159
  提供了关于图神经网络的全面且易懂的介绍，详细阐述了消息传递范式和多种GNN架构。
- [torch.Tensor.index_add_](https://pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html) — PyTorch Documentation (2024)
  Publisher: PyTorch
  `index_add_`操作的官方文档，该操作对自定义GNN层实现中的高效聚合步骤至关重要。
