# 在GAT中实现多头注意力

来源：[原文](https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-2-advanced-gnn-architectures/gat-multi-head-attention)

[返回章节目录](README.md) · [返回课程目录](../README.md)

图注意力网络（GAT）架构的核心组成部分是自注意力 (self-attention)机制 (attention mechanism)。该机制允许节点在聚合特征时，为其邻居分配不同的重要性分数。尽管它有效，但仅依赖单一注意力机制有时会导致训练过程不稳定，或限制模型捕捉节点邻域中不同关系的能力。

为解决这些局限并提高GAT的鲁棒性和表达力，我们可以采用*多头注意力 (multi-head attention)*，这是一种借鉴自Transformer模型成功的技术。

## 多头的基本原理

多头注意力 (multi-head attention)的基本思路是，独立并行地运行多次自注意力 (self-attention)过程。每次独立运行都称作一个“注意力头”。可以将每个头想象成一个独立的专家，关注邻域信息的不同方面，或学习不同的注意力模式。通过汇集这些专家们的见解，模型可以形成更全面、更稳定的表征。

## 机制解析

让我们正式说明多头注意力 (multi-head attention)在GAT层内如何工作。如果我们要使用$K$个独立的注意力头，更新节点$i$的表征$h_i'$（记作$h_i'$）的过程如下：

1. **独立变换和注意力计算：** 对于每个头$k$（其中$k = 1, ..., K$），我们引入一组独立的学习参数 (parameter)：权重 (weight)矩阵$W^{(k)}$和一个注意力机制 (attention mechanism)参数向量 (vector)$\vec{a}^{(k)}$。每个头根据其特定参数计算自己的注意力系数$\alpha_{ij}^{(k)}$：

   - 应用特定于头的线性变换：$z_i^{(k)} = W^{(k)} h_i$。
   - 使用头$k$的注意力机制计算边$(j, i)$的未归一化 (normalization)注意力分数：
     $e_{ij}^{(k)} = \text{LeakyReLU}(\vec{a}^{(k)T} [W^{(k)} h_i || W^{(k)} h_j])$
     或等价地
     $e_{ij}^{(k)} = \text{LeakyReLU}(\vec{a}^{(k)T} [z_i^{(k)} || z_j^{(k)}])$
   - 在节点$i$的邻域$\mathcal{N}_i$内（包括自环）使用softmax函数对分数进行归一化：
     $\alpha_{ij}^{(k)} = \text{softmax}_j(e_{ij}^{(k)}) = \frac{\exp(e_{ij}^{(k)})}{\sum_{l \in \mathcal{N}_i \cup \{i\}} \exp(e_{il}^{(k)})}$
2. **特定于头的特征聚合：** 每个头$k$接着使用其计算出的注意力系数进行加权聚合，从而计算节点$i$的中间输出表征：
   $h_i'^{(k)} = \sigma\left( \sum_{j \in \mathcal{N}_i \cup \{i\}} \alpha_{ij}^{(k)} W^{(k)} h_j \right)$
   这里，$\sigma$表示一个激活函数 (activation function)，如ReLU或ELU。请注意，每个头$h_i'^{(k)}$的输出维度通常是$F'$，即每个头期望的输出特征维度。
3. **组合头输出：** 最后，所有$K$个头（$h_i'^{(1)}, h_i'^{(2)}, ..., h_i'^{(K)}$）的输出需要组合以产生最终层输出$h_i'$。有两种常见策略：

   - **拼接：** 这是中间GAT层最常用的方法。所有头的输出被拼接在一起：
     $h_i' = \Vert_{k=1}^K h_i'^{(k)} = \Vert_{k=1}^K \sigma\left( \sum_{j \in \mathcal{N}_i \cup \{i\}} \alpha_{ij}^{(k)} W^{(k)} h_j \right)$
     得到的特征向量$h_i'$的维度将是$K \times F'$。如果后续层期望不同维度，拼接后可能需要应用额外的线性变换。
   - **平均：** 这常用于GAT模型的最后一层，特别是对于需要特定输出维度的节点分类任务。输出被平均：
     $h_i' = \sigma\left( \frac{1}{K} \sum_{k=1}^K \sum_{j \in \mathcal{N}_i \cup \{i\}} \alpha_{ij}^{(k)} W^{(k)} h_j \right)$
     在这种情况下，输出维度保持为$F'$。

下图说明了多头并行计算及其随后的聚合过程。

> 多头GAT层中的计算流程。输入特征$h_i, h_j$由$K$个注意力头独立处理，每个头使用其自身的权重（$W^{(k)}$）和注意力机制（$Attn_k$）来计算注意力分数（$\alpha_{ij}^{(k)}$）和聚合特征（$h_i'^{(k)}$）。这些特定于头的输出然后被组合（例如，通过拼接或平均）以产生最终输出$h_i'$。

## GAT中多头注意力 (multi-head attention)的优点

使用多个头提供多项重要优势：

1. **稳定的学习过程：** 通过平均注意力输出（无论是在最后一层明确地，还是通过拼接后的下游处理隐式地），多头注意力可以使学习过程更稳定，减少受噪声或初始化不良的单一注意力机制 (attention mechanism)影响。
2. **提升模型能力和更丰富的表征：** 每个头可以学习关注邻域内不同类型的信息或关系。例如，一个头可能更多地关注特征相似的节点，而另一个头可能关注结构重要性。这使模型能够捕捉图结构和节点特征中更复杂、多方面的模式。这种组合（特别是拼接）会产生更丰富的表征。
3. **并行处理能力：** 每个头的计算是独立的，这使得该方法非常适合在GPU等现代硬件上进行并行化，从而减轻了计算成本的增加。

## 实现说明

在使用PyTorch Geometric (PyG) 或 Deep Graph Library (DGL) 等库实现GAT中的多头注意力 (multi-head attention)时，通常不需要从头开始构建并行头和聚合逻辑。标准的GAT层实现（例如PyG中的`GATConv`）通常包含一个`heads`参数 (parameter)。

```python
# PyTorch Geometric 示例
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# 假设输入特征的维度为 'in_channels'
# 我们希望每个头的输出特征维度为 'out_channels'
# 我们使用 K 个头
in_channels = 16
out_channels = 8 # 每个头的输出特征
K = 4 # 头的数量

# 使用拼接的中间层
gat_layer1 = GATConv(in_channels, out_channels, heads=K, dropout=0.6)

# 使用平均的最终层
# 输入维度为前一层的 K * out_channels
# 输出维度为 num_classes
num_classes = 7
gat_layer_final = GATConv(K * out_channels, num_classes, heads=1, concat=False, dropout=0.6)

# 节点分类任务的前向传播示例
# x: 节点特征 [节点数量, 输入通道数]
# edge_index: 图连接性 [2, 边数量]
def forward(x, edge_index):
    x = F.dropout(x, p=0.6, training=True)
    # gat_layer1 的输出维度将是 [节点数量, K * out_channels]
    x = gat_layer1(x, edge_index)
    x = F.elu(x)
    x = F.dropout(x, p=0.6, training=True)
    # gat_layer_final 的输出维度将是 [节点数量, num_classes]
    x = gat_layer_final(x, edge_index)
    return F.log_softmax(x, dim=1)
```

实现时需考虑的因素包括：

- **参数数量：** 使用$K$个头意味着你有$K$组权重 (weight)（$W^{(k)}$）和注意力参数（$\vec{a}^{(k)}$），这会增加参数的总数，相比于单头GAT。
- **输出维度：** 使用拼接时，请注意输出维度的变化。特征维度变为每个头指定的`out_channels`的$K$倍。确保后续层正确处理这种维度变化。平均操作会保持`out_channels`维度。
- **计算成本：** 虽然可以并行化，但计算和内存需求与头的数量$K$大致呈线性关系。

通过加入多头注意力，GAT变得更可靠，能够通过同时关注邻域的不同方面来学习图结构数据的表征。这项技术是许多先进GAT实现中的标准组成部分。

## 参考资料

- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) — Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)
  本文介绍了图注意力网络（GAT）的基础架构，详细阐述了其原始提议以及使用多头注意力机制聚合节点特征的方法。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇重要论文介绍了Transformer模型和多头注意力机制，该机制显著影响了包括图神经网络在内的许多领域中基于注意力模型的设计。

---

[上一节](02-%E5%9B%BE%E6%B3%A8%E6%84%8F%E5%8A%9B%E7%BD%91%E7%BB%9C%20%28GAT%29.md) · [下一节](04-%E5%9B%BETransformer.md)
