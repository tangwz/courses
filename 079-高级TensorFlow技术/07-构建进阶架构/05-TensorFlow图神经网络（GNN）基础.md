# TensorFlow图神经网络（GNN）基础

来源：[原文](https://apxml.com/zh/courses/advanced-tensorflow/chapter-7-implementing-advanced-architectures/gnn-basics-tensorflow)

[返回章节目录](README.md) · [返回课程目录](../README.md)

“传统的神经网络 (neural network)，例如卷积神经网络（CNN）和循环神经网络（RNN），擅长处理网格状结构（图像）或序列模式（文本、时间序列）的数据。然而，许多数据集更适合表示为图，即互连实体的网络。例子有社交网络、分子结构、知识图和推荐系统。图神经网络（GNNs）是一类专门设计用于处理图结构数据的深度学习 (deep learning)模型。”

本节介绍图神经网络的核心原理，并说明如何开始使用TensorFlow的核心API来实现它们。我们假设您熟悉图论基础：节点（顶点）、边（连接）和相关特征。

### 在TensorFlow中表示图

在构建GNN之前，我们需要一种数值方式来表示图数据。常见的表示方法包括：

“1. **邻接矩阵 ($A$):** 一个方阵，其中如果节点 $i$ 和节点 $j$ 之间存在边，则 $A_{ij} = 1$，否则为 $0$。对于加权图，$A_{ij}$ 可以表示边的权重 (weight)。对于图来说，邻接矩阵可能变得非常大且稀疏。`tf.SparseTensor` 通常适合高效地表示稀疏邻接矩阵。”
2. **节点特征矩阵 ($X$):** 一个矩阵，其中每行 $X_i$ 表示节点 $i$ 的特征向量 (vector)。这通常是一个稠密的 `tf.Tensor`。
3. **边列表:** 一个由元组 $(i, j)$ 组成的列表，表示连接节点 $i$ 到节点 $j$ 的边。这可以表示为形状为 `[num_edges, 2]` 的 `tf.Tensor`。边特征可以存储在一个并行的张量中。

这是一个简单的图的可视化：

> 一个包含四个节点（A、B、C、D）和多条连接边的无向小图。每个节点可以有自己的特征（例如，用户资料信息、原子类型），边也可以有特征（例如，关系类型、键合强度）。

### 核心思想：消息传递

大多数图神经网络 (neural network)基于一种称为**消息传递**或**邻域聚合**的原理运行。其直觉是一个节点的表示应受到其邻居表示的影响。这个过程通常在层之间迭代发生。在每一层中，一个节点会进行以下操作：

1. **收集:** 从其直接邻居那里收集信息（特征向量 (vector)或“消息”）。
2. **聚合:** 使用聚合函数（如求和、平均值或最大值）组合收集到的信息。
3. **更新:** 根据其聚合的邻域信息和先前的表示来更新其自身的表示，通常使用一个神经网络层。

经过 $k$ 层消息传递后，节点的表示将包含来自其 $k$ 跳邻域的信息。

### 基本图神经网络 (neural network)层公式

设 $h_v^{(l)}$ 是节点 $v$ 在第 $l$ 层的特征向量 (vector)（嵌入 (embedding)）。一个简化的图神经网络层更新规则可以表示为：


$$
h_v^{(l+1)} = \sigma \left( W^{(l)} \cdot \text{聚合} \left( \{ h_u^{(l)} \mid u \in \mathcal{N}(v) \} \right) + B^{(l)} \cdot h_v^{(l)} \right)
$$


其中：

- $\mathcal{N}(v)$ 是节点 $v$ 的邻居集合。
- $\text{聚合}$ 是一个排列不变函数（例如，求和、平均值、最大值），用于组合邻居表示。
- $W^{(l)}$ 和 $B^{(l)}$ 是可学习的权重 (weight)矩阵，分别用于变换聚合后的邻居特征和节点自身的特征。通常，它们被组合成一个单一的变换，应用于聚合的邻居特征和自身特征的拼接或求和结果。
- $\sigma$ 是一个非线性激活函数 (activation function)（例如，ReLU）。

聚合和更新函数的不同选择会产生各种图神经网络架构，例如图卷积网络（GCN）、GraphSAGE 或图注意力网络（GAT）。

### 在TensorFlow中实现图神经网络 (neural network)层

您可以通过继承 `tf.keras.layers.Layer` 来实现图神经网络层。核心挑战在于使用TensorFlow操作高效地实现收集和聚合步骤。

我们考虑实现一个简单的GCN风格层。假设我们有节点特征 `X` (形状 `[num_nodes, input_dim]`) 和一个稀疏邻接矩阵 `A_sparse` (表示 $\hat{A} = A + I$，即添加自环的邻接矩阵)。

一个简化的GCN层传播规则是：


$$
H^{(l+1)} = \sigma(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W^{(l)})
$$


其中 $\hat{D}$ 是 $\hat{A}$ 的对角度数矩阵。这种归一化 (normalization)可以防止梯度爆炸/消失。

在TensorFlow中，这可能涉及：

1. 计算归一化的邻接矩阵 $\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}$。这通常需要对稀疏张量 `A_sparse` 进行操作以计算度并应用归一化。
2. 通过稀疏矩阵乘法执行邻域聚合：`tf.sparse.sparse_dense_matmul(normalized_A_sparse, H_l)`。
3. 应用线性变换和激活：`activation(tf.matmul(aggregated_features, W_l))`。

以下是一个自定义图神经网络层的结构：

```python
import tensorflow as tf

class SimpleGNNLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        # 可以在这里初始化权重矩阵 W 和 B
        # self.kernel = self.add_weight(...)
        # self.bias = self.add_weight(...)

    def build(self, input_shape):
        # 根据输入形状（节点特征维度）初始化权重
        node_feature_shape = input_shape[0] # 假设输入为 (node_features, adj_info)
        input_dim = node_feature_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.output_dim),
            initializer='glorot_uniform',
            name='kernel')
        # 如果需要，添加其他权重（例如，用于单独变换自身特征）

    def call(self, inputs):
        node_features, adjacency_info = inputs # 例如，adj_info 可以是稀疏张量

        # 1. 聚合邻居特征
        # 此步骤在很大程度上取决于所选的 GNN 变体和图表示
        # 使用稀疏矩阵乘法的例子（如果 adj_info 是归一化的稀疏邻接）
        # aggregated_neighbors = tf.sparse.sparse_dense_matmul(adjacency_info, node_features)

        # 通用聚合逻辑的占位符
        # 您将在此处实现具体的收集/聚合逻辑
        aggregated_neighbors = self._aggregate(node_features, adjacency_info)

        # 2. 变换聚合特征（可能包括自身特征）
        transformed_features = tf.matmul(aggregated_neighbors, self.kernel)

        # 可选：包含自身特征（例如，为 node_features 添加另一个变换）
        # transformed_self = tf.matmul(node_features, self.self_kernel)
        # combined = transformed_neighbors + transformed_self

        # 3. 应用激活函数
        output = self.activation(transformed_features)
        return output

    def _aggregate(self, node_features, adjacency_info):
        # 根据 adjacency_info 格式，在此处实现具体的聚合逻辑
        # 例子：如果 adjacency_info 是归一化的稀疏矩阵：
        if isinstance(adjacency_info, tf.SparseTensor):
             return tf.sparse.sparse_dense_matmul(adjacency_info, node_features)
        # 根据边列表、稠密矩阵等添加其他聚合方法
        else:
             # 对不支持的格式进行回退或报错
             # 为简单起见，只返回 node_features（无实际聚合）
             print("警告：使用了聚合占位符。请实现具体逻辑。")
             return node_features

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'activation': tf.keras.activations.serialize(self.activation)
        })
        return config
```

这个基本结构强调了各个组件：定义权重 (weight)（`build`）、实现前向传播（`call`），包括聚合和变换。`_aggregate` 的实际实现特定于图神经网络变体和数据表示。

### TensorFlow中的图神经网络 (neural network)库

虽然使用TensorFlow核心API从头实现图神经网络能带来深刻理解，但有几个库可以简化这个过程：

- **TensorFlow Geometric (tf\_geometric):** 一个专门为TensorFlow中的图神经网络设计的库，提供许多流行层和基准数据集的实现。
- **Spektral:** 另一个为图神经网络提供Keras层的库，专注于易用性以及在Keras生态系统中的集成。

这些库抽象了许多底层实现细节，使您能够更快地构建复杂的图神经网络模型。然而，理解此处讨论的底层原理对于自定义和问题解决仍然很重要。

图神经网络是学习图结构数据的强大工具。通过理解如何在TensorFlow中表示图并实现核心的消息传递机制，您可以开始将这些技术应用于数据点之间关系和连接重要的各种问题。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍了图卷积网络（GCN）的基础架构，包括谱图卷积和本节讨论的简化层传播规则。
- [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596) — Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, Philip S. Yu (2020)
  Journal: IEEE Transactions on Neural Networks and Learning Systems; Publisher: IEEE; Volume: 32; Pages: 4-24; DOI: [10.1109/TNNLS.2020.2978386](https://doi.org/10.1109/TNNLS.2020.2978386)
  提供了各种图神经网络模型、分类、应用和挑战的广泛概述，是进一步学习的好选择。
- [tf.SparseTensor](https://www.tensorflow.org/api_docs/python/tf/SparseTensor) — TensorFlow Developers (2024)
  Publisher: TensorFlow
  `tf.SparseTensor` 的官方文档，对于在 TensorFlow 中高效表示和处理稀疏图数据（如邻接矩阵）至关重要。
- [Custom layers and models](https://www.tensorflow.org/guide/custom_layers) — TensorFlow Developers (2024)
  Publisher: TensorFlow
  创建自定义 Keras 层的官方指南，提供了实现本节中演示的 `build` 和 `call` 方法的详细说明。
- [CS224W: Machine Learning with Graphs](https://cs224w.stanford.edu/) — Jure Leskovec (2024)
  Publisher: Stanford University
  一门备受推崇的大学课程，提供讲座视频、笔记和作业，以深入了解图机器学习，包括图神经网络。

---

[上一节](04-%E5%9C%A8TensorFlow%E4%B8%AD%E7%BC%96%E5%86%99%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84GAN.md) · [下一节](06-%E4%BD%BF%E7%94%A8%20TF-Agents%20%E7%9A%84%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E6%99%BA%E8%83%BD%E4%BD%93.md)
