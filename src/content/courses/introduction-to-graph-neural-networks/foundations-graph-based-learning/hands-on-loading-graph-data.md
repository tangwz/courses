---
course: "introduction-to-graph-neural-networks"
chapter: "foundations-graph-based-learning"
lesson: "hands-on-loading-graph-data"
sourceId: 7618
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-1-foundations-graph-based-learning/hands-on-loading-graph-data"
title: "动手实践：加载并检查图数据集"
description: "这是一份逐步操作指南，介绍如何使用 NetworkX 加载标准图数据集并检查其基本属性。"
order: 7
plots: []
sourceHash: "79c85dd6d2c249754acd1a8bf0d81976515b6cec6000feba72c73a4253ae6cc5"
sourceCorrections: []
---

理论提供了蓝图，但真正的理解源于直接操作数据。我们将使用 Python 库 NetworkX 加载一个经典的图数据集并检查其属性，以此说明图的抽象表示（如邻接矩阵和特征矩阵）。这一过程将数学定义与实际应用联系起来。

### 搭建环境

在开始之前，你需要安装 NetworkX 和 `matplotlib` 用于可视化。NetworkX 是一个用于创建、操作以及研究复杂网络结构及其动态的强力工具库。你可以使用 pip 安装这些包：

```bash
pip install networkx matplotlib
```

环境就绪后，我们就可以加载并检查第一个图了。

### Zachary's Karate Club：经典的图数据集

我们将使用一个著名的社交网络数据集，称为“扎卡里空手道俱乐部”（Zachary's Karate Club）。该图表示了 20 世纪 70 年代一家大学空手道俱乐部 34 名成员之间的社交关系。俱乐部管理员与教练之间的冲突导致俱乐部最终分裂成两个派系。图中的边表示俱乐部外的友谊关系，常见的任务是预测每个成员在分裂后加入了哪个派系。

幸运的是，NetworkX 内置了该数据集，加载非常方便。

```python
import networkx as nx
import matplotlib.pyplot as plt

# 加载扎卡里空手道俱乐部图
G = nx.karate_club_graph()

# 打印图的一些基本信息
print(f"节点数: {G.number_of_nodes()}")
print(f"边数: {G.number_of_edges()}")
```

运行这段代码会得到以下输出：

```
节点数: 34
边数: 78
```

这告诉我们该图有 34 个节点（俱乐部成员）和 78 条边（友谊关系）。

### 观察节点及其属性

在 NetworkX 中，节点不仅是数字，还可以携带包含相关信息的属性或特征。在空手道俱乐部图中，每个节点都有一个 'club' 属性，指明该成员加入的派系（“Mr. Hi” 或 “Officer”）。

让我们检查节点 0（教练）和节点 33（管理员）。

```python
# 访问特定节点的属性
node_0_data = G.nodes[0]
print(f"节点 0 数据: {node_0_data}")

# 'club' 属性表示所属派系
print(f"节点 0 加入了 '{node_0_data['club']}' 派系。")
print(f"节点 33 加入了 '{G.nodes[33]['club']}' 派系。")
```

输出：

```
节点 0 数据: {'club': 'Mr. Hi'}
节点 0 加入了 'Mr. Hi' 派系。
节点 33 加入了 'Officer' 派系。
```

这个 'club' 属性是我们在节点分类任务中尝试预测的真实标签。GNN 会利用图结构以及可能的其他节点特征来进行这些预测。

### 可视化图结构

建立对图的直观感受的一种好方法是将其可视化。我们可以配合使用 `matplotlib` 和 NetworkX 的绘图功能来绘制该图。

图的一小部分可能如下所示，其中节点 0 和节点 33 是两个派系的核心人物。

> 空手道俱乐部图展示了由节点 0 (“Mr. Hi”) 和节点 33 (“Officer”) 领导的两个派系成员之间的连接。

现在，让我们绘制完整的图。我们可以根据节点的派系所属关系为节点着色，使可视化包含更多信息。这将让我们清楚地看到两个社群。

```python
# 根据 'club' 属性创建颜色映射
colors = []
for node in G:
    if G.nodes[node]['club'] == 'Mr. Hi':
        colors.append('#9775fa')  # 紫色
    else:
        colors.append('#ffa94d')  # 橙色

# 绘制图形
plt.figure(figsize=(8, 6))
nx.draw_spring(G, with_labels=True, node_color=colors, node_size=500)
plt.title("Zachary's Karate Club Social Network")
plt.show()
```

这段脚本生成的图像中，两个派系清晰可见。你可以看到友谊关系（边）是如何倾向于在两个群体内部聚集，而群体之间的连接较少。这种结构正是 GNN 学习的对象。

### 从图对象到矩阵表示

正如我们之前讨论过的，GNN 不直接对图对象进行操作。它们需要数值张量：邻接矩阵 $A$ 和节点特征矩阵 $X$。使用 NetworkX 可以很方便地导出邻接矩阵。

```python
import numpy as np

# 获取 NumPy 数组格式的邻接矩阵
A = nx.to_numpy_array(G)

print("邻接矩阵形状:", A.shape)
print("A 的部分条目:")
print(A[:5, :5])
```

输出：

```
邻接矩阵形状: (34, 34)
A 的部分条目:
[[0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 0. 0. 0. 1. 0. 1. 0. 1.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [1. 0. 1. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 1. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 1. 0. 1. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 0.]
 [1. 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
```

邻接矩阵的形状是 $(34, 34)$，即 $N \times N$（$N$ 为节点数）。$A[i, j]$ 为 `1` 表示节点 $i$ 和节点 $j$ 之间存在边，而 `0` 表示没有直接连接。

那么节点特征矩阵 $X$ 呢？对于这个特定的数据集，并没有提供显式的节点特征。在这种情况下，通常的策略是根据图结构本身创建特征。例如，我们可以使用每个节点的度（它拥有的连接数）作为简单的特征。另一种方法是使用单位矩阵，为每个节点分配一个唯一的独热编码（one-hot encoded）向量 (vector)。

为了演示，我们创建一个简单的特征矩阵，其中每个节点的特征就是它的度。

```python
# 创建特征矩阵，其中每个特征是节点的度
degrees = [G.degree(n) for n in G.nodes()]
X = np.array(degrees).reshape(-1, 1)

print("特征矩阵形状:", X.shape)
print("前 5 个特征（节点度）:")
print(X[:5])
```

输出：

```
特征矩阵形状: (34, 1)
前 5 个特征（节点度）:
[[16]
 [ 9]
 [10]
 [ 6]
 [ 3]]
```

现在我们的图由两个矩阵表示：描述结构的 $A$ 和描述节点属性的 $X$。这些是大多数 GNN 模型所需的核心输入。有了这个起点，我们就可以准备研究 GNN 如何使用这些矩阵从图数据中学习了。

## 参考资料

- [NetworkX Documentation](https://networkx.org/documentation/stable/) — NetworkX Developers (2024)
  NetworkX Python库的官方文档，详细介绍了图的创建、操作、分析和可视化功能。
- [An Information Flow Model for Conflict and Fission in Small Groups](https://www.jstor.org/stable/3629752) — Wayne W. Zachary (1977)
  Journal: Journal of Anthropological Research; Publisher: University of New Mexico; Volume: 33; Pages: 452-473; DOI: [10.1086/jar.33.4.3629752](https://doi.org/10.1086/jar.33.4.3629752)
  介绍扎卡里空手道俱乐部数据集的原始研究论文，描述了社交网络和观察到的俱乐部分裂。
- [Networks: An Introduction](https://doi.org/10.1093/oso/9780198805090.001.0001) — Mark E. J. Newman (2018)
  Publisher: Oxford University Press; DOI: [10.1093/oso/9780198805090.001.0001](https://doi.org/10.1093/oso/9780198805090.001.0001)
  一本关于网络科学的综合教材，涵盖图论、网络属性和与图数据集相关的矩阵表示。
- [Graph Neural Networks: Foundations, Frontiers, and Applications](https://doi.org/10.1007/978-981-16-6054-2) — Lingfei Wu, Peng Cui, Jian Pei, Liang Zhao (2022)
  Publisher: Springer Nature; Pages: 689; DOI: [10.1007/978-981-16-6054-2](https://doi.org/10.1007/978-981-16-6054-2)
  一本提供图神经网络广泛概述的教材，解释了图表示和数据结构在GNNs中的作用。
