# 动手实践：从零开始实现 GCN

来源：[原文](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-3-foundational-gnn-architectures/hands-on-implementing-gcn)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将图卷积网络（GCN）的数学公式转化为可运行的模型。通过实际操作，巩固对 GCN 消息传递机制运行方式的理解。我们将使用 PyTorch 构建一个双层 GCN，重点关注定义图卷积的核心矩阵运算。

这种方法能让你在第 5 章使用更高级的库（这些库会隐藏许多细节）之前，清晰地看到模型的内部运作方式。

### 组件设置

要构建一个 GCN，我们需要三个主要组件：

1. **图结构：** 由归一化 (normalization)邻接矩阵表示。
2. **节点特征：** 一个矩阵，其中每一行对应一个节点的特征向量 (vector)。
3. **GCN 层：** 执行特征转换的神经网络 (neural network)模块。

我们先定义一个简单的小型图，并准备好计算所需的矩阵。我们将使用一个包含四个节点的图。

#### 第 1 步：准备图数据

首先，我们定义图的邻接矩阵 `A` 和初始特征矩阵 `X`。然后，按照“图卷积网络 (GCN)”部分讨论的归一化过程进行处理。GCN 层的传播规则为：


$$
H^{(l+1)} = \sigma(\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})
$$


我们的首要任务是计算归一化邻接矩阵 $\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}$。我们将使用 PyTorch 完成所有张量运算。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. 定义图结构和特征
adj = torch.tensor([
    [0., 1., 1., 0.],
    [1., 0., 1., 1.],
    [1., 1., 0., 0.],
    [0., 1., 0., 0.]
])

features = torch.tensor([
    [10., 2.],
    [5., 15.],
    [20., 10.],
    [1., 1.]
])

# 2. 为邻接矩阵添加自环
# A_hat = A + I
adj_hat = adj + torch.eye(adj.shape[0])

# 3. 计算度矩阵 D_hat
# D_ii = Sum_j A_ij
degree_hat = torch.diag(torch.sum(adj_hat, dim=1))

# 4. 计算 D_hat^(-1/2)
# 添加一个极小值 epsilon 以保证数值稳定性
degree_hat_inv_sqrt = torch.pow(degree_hat.diag() + 1e-5, -0.5)
degree_hat_inv_sqrt = torch.diag(degree_hat_inv_sqrt)

# 5. 计算归一化邻接矩阵
# norm_A = D_hat^(-1/2) * A_hat * D_hat^(-1/2)
norm_adj = torch.matmul(torch.matmul(degree_hat_inv_sqrt, adj_hat), degree_hat_inv_sqrt)

print("归一化邻接矩阵：")
print(norm_adj)
```

输出的 `norm_adj` 是用于在节点间传播信息的矩阵。它是对称的，其值根据节点度进行缩放。该矩阵在模型训练过程中保持不变，只需计算一次。

### 第 2 步：定义单个 GCN 层

接下来，我们为单个 GCN 层创建一个自定义 PyTorch 模块。这个类将包含一个可训练的权重 (weight)矩阵 `W`，其 `forward` 方法将执行核心乘法运算：`norm_adj @ features @ W`。

```python
class GCNLayer(nn.Module):
    """
    单个图卷积网络层。
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        # 定义可训练权重矩阵 W
        self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # 使用标准方法初始化权重
        nn.init.xavier_uniform_(self.weights)

    def forward(self, adj, features):
        # 首先转换特征：X * W
        support = torch.matmul(features, self.weights)
        # 然后传播特征：norm_A * (X * W)
        output = torch.matmul(adj, support)
        return output
```

这个 `GCNLayer` 类清晰地封装了逻辑。它将预先计算好的归一化 (normalization)邻接矩阵和节点特征作为输入，并生成新的节点嵌入 (embedding)。`nn.Parameter` 包装器确保 PyTorch 的自动微分系统会在训练期间跟踪权重矩阵的梯度。

### 第 3 步：构建完整的双层 GCN 模型

定义好 `GCNLayer` 后，将它们堆叠形成深度 GNN 就很直接了。我们将创建一个用于节点分类的双层 GCN。该模型将：

1. 将输入特征传递给第一个 GCN 层。
2. 对输出应用 ReLU 非线性激活。
3. 将结果传递给第二个 GCN 层以获得最终输出（logits）。

> 双层 GCN 的数据流。归一化 (normalization)邻接矩阵被传递到每一层，用以指导消息的聚合。

以下是完整模型的实现：

```python
class GCN(nn.Module):
    """
    双层图卷积网络。
    """
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(in_features, hidden_features)
        self.gc2 = GCNLayer(hidden_features, out_features)

    def forward(self, adj, features):
        # 第一层及 ReLU 激活
        x = F.relu(self.gc1(adj, features))
        # 第二层
        x = self.gc2(adj, x)
        return x
```

这个 `GCN` 类组合了两个 `GCNLayer` 实例。`forward` 方法定义了计算流：第一层的输出成为第二层的输入，中间加入 ReLU 激活以引入非线性，从而允许模型学习更复杂的函数。

### 运行前向传播

现在我们可以将所有部分组合在一起。我们将实例化 `GCN` 模型，并将准备好的图数据传递给它以获得最终的节点嵌入 (embedding)。假设我们的任务是将每个节点分为三类之一。

```python
# 从数据中获取维度
num_nodes = features.shape[0]
in_features = features.shape[1]
hidden_features = 16  # 隐藏层大小的常用选择
out_features = 3      # 类别数量

# 实例化模型
model = GCN(in_features, hidden_features, out_features)

# 执行前向传播
logits = model(norm_adj, features)

print("\n模型输出 (Logits)：")
print(logits)
print("\n输出形状：", logits.shape)
```

该脚本的输出将是一个形状为 `[4, 3]` 的张量。这代表了 4 个节点中每个节点属于 3 个类别中每一个的原始未归一化 (normalization)分数（logits）。

```
模型输出 (Logits)：
tensor([[-1.9965, -3.2057,  2.0163],
        [-2.3150, -4.5161,  3.7142],
        [-2.2965, -4.0189,  2.9785],
        [-1.3857, -2.9754,  2.4187]], grad_fn=<MmBackward0>)

输出形状：torch.Size([4, 3])
```

我们已经成功地从零开始构建了一个 GCN。输出张量中的每一行都是节点的新表示，是通过聚合图结构定义的局部邻域信息计算得出的。在下一章中，我们将学习如何将这些 logits 与损失函数 (loss function)结合使用，来训练模型权重 (weight)，并完成节点分类等任务的准确预测。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf, Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍图卷积网络（GCN）架构及其核心传播规则的原创研究论文，本节中对此进行了实现。
- [torch.nn - PyTorch 2.1 documentation](https://pytorch.org/docs/stable/nn.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  PyTorch神经网络模块的官方文档，提供了`nn.Module`、`nn.Parameter`以及构建自定义层的必要细节，这些是GCN实现的基础。
- [CS224W: Machine Learning with Graphs](http://cs224w.stanford.edu/) — Jure Leskovec (2024)
  一门备受推崇的大学课程，提供关于图神经网络的全面讲义和作业，其中包括对GCN的详细讨论和实际应用。
- [Graph Representation Learning](https://doi.org/10.2200/S01045ED1V01Y202009AIM046) — William L. Hamilton (2020)
  Journal: Synthesis Lectures on Artificial Intelligence and Machine Learning; Publisher: Morgan & Claypool Publishers; Volume: 46; Pages: 1-159; DOI: [10.2200/S01045ED1V01Y202009AIM046](https://doi.org/10.2200/S01045ED1V01Y202009AIM046)
  一本简明的学术专著，提供了图上表示学习方法的理论和实践概述，其中有专门章节解释了图卷积网络。

---

[上一节](07-GCN%E3%80%81GraphSAGE%20%E4%B8%8E%20GAT%20%E7%9A%84%E6%AF%94%E8%BE%83.md) · [下一节](../04-GNN%20%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83/01-%E4%B8%BA%E8%8A%82%E7%82%B9%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1%E6%90%AD%E5%BB%BA%20GNN.md)
