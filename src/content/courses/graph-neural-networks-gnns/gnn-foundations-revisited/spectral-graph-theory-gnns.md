---
course: "graph-neural-networks-gnns"
chapter: "gnn-foundations-revisited"
lesson: "spectral-graph-theory-gnns"
sourceId: 2661
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-1-gnn-foundations-revisited/spectral-graph-theory-gnns"
title: "图神经网络的谱图理论基础知识"
description: "学习图傅里叶变换和谱卷积如何成为谱图神经网络的依据。"
order: 3
plots: []
sourceHash: "d85428d7dae23a1cf0fee301a4c2bda803b4653b150f42d7929320656574269b"
sourceCorrections: []
---

虽然消息传递框架提供了信息如何在节点间局部传播的空间视角，谱图理论通过其频率成分的角度分析图的属性，提供了另一种补充的视角。这种方法对于掌握第一代图神经网络 (neural network)（称为谱图神经网络）至关重要，并且有助于阐明图结构和图上的信号处理。

谱图理论的核心是**图拉普拉斯矩阵**。它是图的矩阵表示，能够体现重要的结构信息。它有几种变体，但其中两种对图神经网络尤为重要：

1. **组合拉普拉斯矩阵 ($L$):** 定义为 $L = D - A$，在此 $D$ 是对角线度矩阵 ($D_{ii} = \text{度}(v_i)$)，$A$ 是邻接矩阵。
2. **对称归一化 (normalization)拉普拉斯矩阵 ($L_{sym}$):** 定义为 $L_{sym} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$，且 $I$ 是单位矩阵。此版本常受青睐，因为其特征值介于 0 和 2 之间，使得分析和学习更为稳定。$D^{-1/2}$ 是对角矩阵，其元素为 $(D^{-1/2})_{ii} = 1 / \sqrt{\text{度}(v_i)}$。对于孤立节点（度为 0），我们通常将对应的元素设置为 0。

拉普拉斯矩阵，特别是 $L_{sym}$，是一个实对称半正定矩阵。这意味着它拥有一整套实数、非负的特征值和对应的正交特征向量 (vector)。

### 拉普拉斯矩阵的特征分解

谱图理论的核心思想是，我们可以利用归一化 (normalization)拉普拉斯矩阵 $L_{sym}$ 的特征值和特征向量 (vector)对其进行分解：


$$
L_{sym} = U \Lambda U^T
$$


让我们分解一个具有 $N$ 个节点的图的这些组成部分：

- $\Lambda = \text{diag}(\lambda_1, \lambda_2, ..., \lambda_N)$ 是包含 $L_{sym}$ **特征值**的对角矩阵。这些特征值是实数且非负，通常排序为 $0 = \lambda_1 \le \lambda_2 \le ... \le \lambda_N \le 2$。特征值常被解释为图的**频率**。
- $U = [\mathbf{u}_1, \mathbf{u}_2, ..., \mathbf{u}_N]$ 是一个正交矩阵 ($U^T U = U U^T = I$)，其列向量 $\mathbf{u}_i$ 是对应的**特征向量**。这些特征向量构成了 $\mathbb{R}^N$ 的正交基。它们有时被称为**图傅里叶模式**。

特征向量代表图上的基本模式或变动方式。与小特征值（低频率）相关的特征向量在连接节点间平滑变化，而与大特征值（高频率）相关的特征向量在相邻节点间快速振荡。特征值 $\lambda_i$ 的大小表示对应特征向量 $\mathbf{u}_i$ 在边上的平滑程度。具体而言，$\lambda_i = \mathbf{u}_i^T L_{sym} \mathbf{u}_i$ 用于衡量 $\mathbf{u}_i$ 的变动。

> 环图 (C4) 的一个特征向量示例。节点颜色表示特征向量值（例如，红色表示负值，蓝色表示正值）。这个特征向量对应一个较高频率（$\lambda=2$），并且在邻居之间显示交替值。较低频率的特征向量会呈现更平滑的过渡。

### 图傅里叶变换 (GFT)

正如经典的傅里叶变换使用正弦和余弦基函数将信号分解为其频率成分一样，图傅里叶变换 (GFT) 使用拉普拉斯特征向量 (vector)作为依据，分解**图信号**。

图信号只是在图节点上定义的一个函数，表示为一个向量 $\mathbf{x} \in \mathbb{R}^N$，其中 $x_i$ 为与节点 $v_i$ 相关的值或特征。

信号 $\mathbf{x}$ 的 GFT 定义为其在特征向量 $U$ 上的投影：


$$
\hat{\mathbf{x}} = U^T \mathbf{x}
$$


得到的向量 $\hat{\mathbf{x}}$ 是信号 $\mathbf{x}$ 在图谱域中的表示。每个分量 $\hat{x}_i = \mathbf{u}_i^T \mathbf{x}$ 表示信号 $\mathbf{x}$ 中第 $i$ 个图傅里叶模式 $\mathbf{u}_i$ 的强度。

逆 GFT 允许我们从其谱表示重建原始信号：


$$
\mathbf{x} = U \hat{\mathbf{x}} = U U^T \mathbf{x}
$$


这在经典傅里叶分析和图上信号处理之间建立了一种直接的类比。

### 图上的谱卷积

卷积在经典信号处理中十分重要，常用于滤波。卷积定理指出，空间（或时间）域中的卷积等同于频率（或谱）域中的逐元素乘法。

我们可以使用 GFT 为图定义一个类似的运算。设 $\mathbf{x}$ 是一个图信号，$\mathbf{g}$ 是一个滤波器信号。它们的谱卷积 $*_{\mathcal{G}}$ 定义为在谱域中执行乘法，然后使用逆 GFT 转换回来：


$$
\mathbf{g} *_{\mathcal{G}} \mathbf{x} = U ( (U^T \mathbf{g}) \odot (U^T \mathbf{x}) ) = U (\hat{\mathbf{g}} \odot \hat{\mathbf{x}})
$$


这里，$\odot$ 代表逐元素的哈达玛积。

在图神经网络 (neural network)的背景下，我们通常希望将可学习的滤波器应用于节点特征（我们的图信号）。谱图神经网络不是直接在空间域定义滤波器 $\mathbf{g}$，而是通过由可学习权重 (weight) $\theta$ 参数 (parameter)化的函数 $g_\theta(\cdot)$ 在谱域中定义滤波器。此函数直接作用于图频率（特征值）。

我们将谱滤波器定义为 $\hat{\mathbf{g}}_\theta = g_\theta(\Lambda)$，它是一个对角矩阵，函数 $g_\theta$ 应用于每个特征值：$\text{diag}(g_\theta(\lambda_1), g_\theta(\lambda_2), ..., g_\theta(\lambda_N))$。

通过谱卷积将此滤波器 $g_\theta(\Lambda)$ 应用于信号 $\mathbf{x}$ 得到：


$$
g_\theta(\Lambda) *_{\mathcal{G}} \mathbf{x} = U g_\theta(\Lambda) \hat{\mathbf{x}} = U g_\theta(\Lambda) U^T \mathbf{x}
$$


表达式 $U g_\theta(\Lambda) U^T$ 代表滤波器在空间域中的作用。

### 在谱图神经网络 (neural network)中的作用

这个公式是谱图神经网络的依据。一个谱图神经网络层主要是在谱域中将参数 (parameter)化的滤波器 $g_\theta(\Lambda)$ 应用于输入节点特征 $\mathbf{H}^{(k-1)}$（它们是图信号）：


$$
\mathbf{H}^{(k)} = \sigma \left( U g_\theta^{(k)}(\Lambda) U^T \mathbf{H}^{(k-1)} \right)
$$


$\sigma$ 是非线性激活函数 (activation function)。网络学习滤波器函数 $g_\theta$ 的参数 $\theta$。

然而，这种直接的谱方法面临显著的实际难题：

1. **计算成本：** 计算特征分解 $U \Lambda U^T$ 的计算成本很高，通常按 $O(N^3)$ 比例增长，这对于大型图而言是难以承受的。涉及 $U$ 和 $U^T$ 的乘法每层也需要 $O(N^2)$ 的成本。
2. **非局部性：** 学习到的滤波器 $g_\theta(\Lambda)$ 依赖于图的全局结构（编码在 $U$ 和 $\Lambda$ 中），并且在顶点域中不具有局部性。这意味着图的远处部分发生变化可能会影响滤波器在特定节点的输出。
3. **可迁移性：** 依据 $U$ 对给定图结构是特定的。在一个图上学习到的滤波器不能直接应用于不同大小或连接性的另一个图，因为它们的拉普拉斯特征向量 (vector)会有所不同。

这些限制促成了空间图神经网络（如前面讨论的消息传递模型）和谱卷积近似（如 GCN、ChebNet）的出现，这些方法在空间域中运行更高效，我们将在下一章中考察。

然而，掌握谱视角仍然具有价值。它为图卷积提供了理论依据，有助于将图神经网络的操作解释为频率滤波，并将图神经网络与更广泛的图信号处理领域联系起来。它构成了许多后续改进的起点，这些改进旨在克服其局限性，同时保留其理想特性。

## 参考资料

- [Spectral Graph Theory](https://bookstore.ams.org/cbms-92) — Fan R. K. Chung (1997)
  Publisher: American Mathematical Society
  这本开创性的教材奠定了谱图理论的数学基础，提供了图拉普拉斯算子及其特征分解的重要背景知识。
- [The Emerging Field of Signal Processing on Graphs: Extending High-Dimensional Data Analysis to Networks and Other Irregular Domains](https://ieeexplore.ieee.org/document/6368853) — David I. Shuman, S. K. Narang, Pascal Frossard, Antonio Ortega, Pierre Vandergheynst (2012)
  Journal: IEEE Signal Processing Magazine; Publisher: IEEE; Volume: 30; Pages: 83-98; DOI: [10.1109/MSP.2012.2235192](https://doi.org/10.1109/MSP.2012.2235192)
  这篇有影响力的综述文章介绍了图信号处理，正式定义了图傅里叶变换和谱图卷积。
- [A Comprehensive Survey on Graph Neural Networks](https://ieeexplore.ieee.org/document/8786938) — Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and S. Yu Philip (2020)
  Journal: IEEE Transactions on Neural Networks and Learning Systems; Publisher: IEEE; Volume: 32; Pages: 4-24; DOI: [10.1109/TNNLS.2020.2978070](https://doi.org/10.1109/TNNLS.2020.2978070)
  一篇全面的综述，回顾了各种图神经网络架构，包括对基于谱的图神经网络及其演变的讨论。
