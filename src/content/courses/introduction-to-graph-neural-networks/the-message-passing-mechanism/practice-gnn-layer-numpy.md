---
course: "introduction-to-graph-neural-networks"
chapter: "the-message-passing-mechanism"
lesson: "practice-gnn-layer-numpy"
sourceId: 7625
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-2-the-message-passing-mechanism/practice-gnn-layer-numpy"
title: "实践：使用 NumPy 实现简单的 GNN 层"
description: "仅使用 Python 和 NumPy 从头构建一个基础的消息传递层，以巩固你对其运作机制的理解。"
order: 7
plots: []
sourceHash: "cfac359767a95e38b1bbb163eccc405cf52ebc6eea762facbc2c2fa8a1af4417"
sourceCorrections: []
---

仅使用 Python 和 NumPy 从头开始实现一个简化的图神经网络 (neural network)（GNN）层。这包括将消息传递机制的核心聚合和更新步骤直接转换为代码。其目的是通过实际应用，让 GNN 层的运作机制变得直观具体。

### 示例图

首先，我们需要一个图。让我们使用一个包含四个节点和四条边的简单小图。这个规模非常适合检查矩阵并通过手动计算进行验证。

> 一个具有四个节点及其连接关系的简单无向图。

我们的目标是编写一个函数，输入该图的结构和初始节点特征，并计算经过一层消息传递后的更新节点特征。

### 使用 NumPy 表示图

首先，我们将图的组成部分（结构和特征）表示为 NumPy 数组。正如我们在第 1 章中所学到的，这涉及邻接矩阵 `A` 和节点特征矩阵 `X`（我们也可以将其称为隐藏状态 `H`）。

假设我们节点的初始特征维度为 2。例如，这可能是某些初始特征提取或嵌入 (embedding)过程的结果。

```python
import numpy as np

# 邻接矩阵 (A)
# 表示节点之间的连接。如果节点 i 和 j 相连，则 A[i, j] = 1。
A = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 0]
])

# 节点特征矩阵 (X 或 H)
# 每一行对应一个节点，每一列是一个特征。
# 这里我们有 4 个节点，每个节点有 2 个特征。
X = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8]
])

# 我们还为 GNN 层定义一个权重矩阵。
# 该层将把 2 维特征转换为 3 维特征。
# 输入维度 = 2, 输出维度 = 3
W = np.random.rand(2, 3)

print("邻接矩阵 (A):\n", A)
print("\n节点特征 (X):\n", X)
print("\n权重 (W):\n", W)
```

### 步骤 1：聚合步骤

消息传递的核心思想是聚合来自节点邻居的信息。同时为所有节点执行此聚合的一种直接且有效的方法是通过矩阵乘法。将邻接矩阵 `A` 与特征矩阵 `X` 相乘，正好可以得到我们需要的结果。


$$
\mathbf{M} = \mathbf{A} \mathbf{X}
$$


让我们看看这个运算产生了什么：

```python
# 聚合邻居特征
M = A @ X

print("聚合特征 (M = A @ X):\n", M)
```

得到的矩阵 `M` 与 `X` 的维度相同。每一行 `M[i]` 是节点 `i` 的邻居特征向量 (vector)之和。例如，节点 0 与节点 1 和 2 相连。因此，`M` 的第一行是节点 1 的特征向量 (`[0.3, 0.4]`) 和节点 2 的特征向量 (`[0.5, 0.6]`) 之和，结果为 `[0.8, 1.0]`。这种单一的矩阵乘法高效地并行完成了所有节点的 `AGGREGATE`（聚合）步骤。

### 更有效的聚合方式

简单的聚合方式 `A @ X` 有两个缺点：

1. **缺失自身信息：** 节点的聚合特征不包括其自身的特征向量 (vector)。节点的当前状态通常是预测其下一状态最有效的信息。
2. **度偏差：** 具有大量邻居（高填充度）的节点将拥有数值大得多的聚合特征向量，而低度节点的聚合特征向量则较小。这可能导致训练不稳定，并使模型以不理想的方式对图结构产生敏感性。

我们可以通过两个简单的修改来解决这些问题。

#### 添加自环

为了在聚合中包含节点自身的特征，我们只需为每个节点添加一个自环。在邻接矩阵方面，这意味着加上一个单位矩阵 `I`。


$$
\hat{\mathbf{A}} = \mathbf{A} + \mathbf{I}
$$


```python
# 为邻接矩阵添加自环
A_hat = A + np.eye(A.shape[0])

print("带有自环的邻接矩阵 (A_hat):\n", A_hat)
```

现在，当我们计算 `A_hat @ X` 时，每个节点的聚合将包含其自身的特征以及邻居的特征。

#### 标准化特征

为了解决度偏差问题，我们可以对聚合特征进行标准化。由图卷积网络（GCN）推广的一种常用技术是将特征除以每个节点的度。这实际上是计算邻居特征的*平均值*而不是总和。

对称标准化公式为：


$$
\mathbf{A}_{\text{标准化}} = \hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-1/2}
$$


这里，$\hat{\mathbf{A}}$ 是带有自环的邻接矩阵，$\hat{\mathbf{D}}$ 是从 $\hat{\mathbf{A}}$ 计算得出的对角度矩阵。

让我们来实现它：

```python
# 计算度矩阵 (D_hat)
D_hat = np.diag(np.sum(A_hat, axis=1))

print("度矩阵 (D_hat):\n", D_hat)

# 计算度矩阵的平方根倒数
# 我们添加一个极小值 epsilon 以避免孤立节点出现除以零的情况
D_hat_inv_sqrt = np.linalg.inv(D_hat)**0.5

# 标准化邻接矩阵
A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

print("\n标准化邻接矩阵:\n", A_norm)
```

这个标准化的邻接矩阵 `A_norm` 现在可以用于执行更稳定且更有效的聚合。

### 步骤 2：更新步骤

聚合之后，`UPDATE`（更新）步骤会应用一个可学习的线性变换，随后是一个非线性激活函数 (activation function)。这与神经网络 (neural network)标准稠密层内部的操作完全相同。

我们简单的 GNN 层的完整公式为：


$$
\mathbf{H}^{(l+1)} = \sigma \left( \hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)
$$


其中 $\sigma$ 是像 ReLU 这样的非线性激活函数。

### 将所有部分整合到函数中

让我们将此逻辑封装到一个可重用的 Python 函数中。该函数代表了一个完整的消息传递层。

```python
def gnn_layer(A, X, W):
    """
    执行一层图神经网络消息传递。

    参数:
        A (np.array): 图的邻接矩阵。
        X (np.array): 节点特征矩阵。
        W (np.array): 该层的权重矩阵。

    返回:
        np.array: 更新后的节点特征矩阵。
    """
    # 1. 为邻接矩阵添加自环
    A_hat = A + np.eye(A.shape[0])

    # 2. 计算度矩阵
    D_hat = np.diag(np.sum(A_hat, axis=1))

    # 3. 计算度矩阵的平方根倒数
    D_hat_inv_sqrt = np.linalg.inv(D_hat)**0.5

    # 4. 标准化邻接矩阵
    A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

    # 5. 执行聚合和更新步骤
    # (A_norm @ X) 是邻居特征的聚合
    # (... @ W) 是线性变换（更新）
    H_prime = A_norm @ X @ W

    # 6. 应用非线性激活函数 (ReLU)
    H_next = np.maximum(0, H_prime)

    return H_next

# 将我们的图数据通过 GNN 层运行
H_next = gnn_layer(A, X, W)

print("原始节点特征 (X):\n", X)
print("\n经过一层 GNN 层更新后的节点特征 (H_next):\n", H_next)
```

输出结果 `H_next` 是一个 `4x3` 的矩阵。每一行是对应节点的新特征向量 (vector)。这个新向量是通过对其自身及其邻居的特征进行标准化平均，然后将结果转换到新的高维空间 (high-dimensional space)中计算得出的。这个过程成功地基于每个节点的局部邻域更新了其表示。

通过从头实现这个层，你已经构建了许多强效 GNN 架构的基础组件。在下一章中，我们将看到这一逻辑如何构成图卷积网络（GCN）的基础，并研究 `AGGREGATE` 和 `UPDATE` 函数的其他变体。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR 2017); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  这篇开创性论文介绍了图卷积网络（GCN）架构，本节中实现的GCN核心层，包括对称归一化和消息传递机制，正是基于此。
- [Graph Representation Learning](https://doi.org/10.2200/S01045ED1V01Y202009AIM046) — William L. Hamilton (2020)
  Journal: Synthesis Lectures on Artificial Intelligence and Machine Learning; Publisher: Morgan & Claypool Publishers; Volume: 14; Pages: 1-159; DOI: [10.2200/S01045ED1V01Y202009AIM046](https://doi.org/10.2200/S01045ED1V01Y202009AIM046)
  为图表示学习提供了全面的理论基础，详细讨论了消息传递神经网络和各种GNN架构，为本节实现的GCN层提供了背景。
- [\`torch_geometric.nn.conv.GCNConv\`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv) — PyTorch Geometric Developers (2024)
  PyTorch Geometric库中图卷积网络（GCN）层实现的官方文档，展示了核心逻辑如何转化为高性能深度学习框架。
- [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596) — Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, Philip S. Yu (2020)
  Journal: IEEE Transactions on Neural Networks and Learning Systems; Publisher: IEEE; Volume: 32; Pages: 4-24; DOI: [10.1109/TNNLS.2020.2978386](https://doi.org/10.1109/TNNLS.2020.2978386)
  这篇综述提供了图神经网络的广泛概览，对不同架构进行了分类，并详细介绍了包括本节GCN在内的许多GNN所采用的通用消息传递框架。
