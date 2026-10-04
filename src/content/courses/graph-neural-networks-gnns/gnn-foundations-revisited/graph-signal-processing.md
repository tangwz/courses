---
course: "graph-neural-networks-gnns"
chapter: "gnn-foundations-revisited"
lesson: "graph-signal-processing"
sourceId: 2664
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-1-gnn-foundations-revisited/graph-signal-processing"
title: "图信号处理思想"
description: "介绍图信号处理中与GNN滤波器设计及分析相关的一些思想。"
order: 4
plots: ["plots/2664-0.json"]
sourceHash: "c7ddb6decb018dfbf10244c33ea658d2405d51f3f9f5d07c30b15b0c5340b566"
sourceCorrections: []
---

图信号处理（GSP）提供了一个强大的数学框架，用于分析在图节点上定义的函数。GSP基于图拉普拉斯算子和谱图理论等概念，使我们能够将图操作，特别是GNN内部的操作，视为信号滤波，这与经典信号处理处理时间序列或图像数据的方式非常相似。这种观点对于理解谱GNN以及分析过平滑等现象尤其有帮助。

## 图上的信号

在GSP中，一个“图信号”只是一个为图中每个节点赋值的函数。如果有一个包含$N = |V|$个节点的图$G = (V, E)$，一个图信号$\mathbf{x}$是$\mathbb{R}^N$中的一个向量 (vector)，其中$x_i$是节点$v_i$处的信号值。


$$
\mathbf{x} = [x_1, x_2, \dots, x_N]^T \in \mathbb{R}^N
$$


这个定义与我们在GNN中使用的节点特征直接一致。所有节点上的单个特征可以被视为一个图信号。因此，节点特征矩阵$\mathbf{X} \in \mathbb{R}^{N \times F}$可以被看作是$F$个不同图信号的集合，每个信号对应一个特征维度。

## 图傅里叶变换

与将时域信号分解为频率分量的经典傅里叶变换类似，图傅里叶变换（GFT）将图信号分解为与图结构相关的分量，特别是其“频率”。这些频率与图拉普拉斯算子$L$的特征值相关联。

回想一下，归一化 (normalization)图拉普拉斯算子$\mathcal{L} = I - D^{-1/2} A D^{-1/2}$（或未归一化的$L=D-A$）是一个对称半正定矩阵。它具有一套完整的正交特征向量 (vector)$\mathbf{U} = [\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_N]$，以及对应的实数非负特征值$0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_N$，存储在对角矩阵$\mathbf{\Lambda}$中。

### 定义

图信号$\mathbf{x}$的GFT定义为其在拉普拉斯算子特征向量 (vector)上的投影：


$$
\hat{\mathbf{x}} = \mathbf{U}^T \mathbf{x}
$$


这里，$\hat{\mathbf{x}} = [\hat{x}_1, \hat{x}_2, \dots, \hat{x}_N]^T$是傅里叶系数向量。每个系数$\hat{x}_i = \mathbf{u}_i^T \mathbf{x}$衡量信号$\mathbf{x}$与对应特征向量$\mathbf{u}_i$的一致性。

逆图傅里叶变换（iGFT）根据其谱系数重构原始信号：


$$
\mathbf{x} = \mathbf{U} \hat{\mathbf{x}} = \sum_{i=1}^N \hat{x}_i \mathbf{u}_i
$$


这表明信号$\mathbf{x}$可以表示为拉普拉斯算子特征向量的线性组合，并由傅里叶系数$\hat{x}_i$加权。

### 频率解释

拉普拉斯算子的特征值$\lambda_i$量化 (quantization)了它们对应特征向量 (vector)$\mathbf{u}_i$在图上的“平滑度”或“变化”。

- 小特征值（$\lambda_i \approx 0$）对应低频率。相关的特征向量$\mathbf{u}_i$在连接的节点间变化缓慢；它们代表平滑信号。与$\lambda_1 = 0$相关的特征向量$\mathbf{u}_1$通常是一个常数向量（对于连通图），代表最平滑的信号（零变化）。
- 大特征值对应高频率。相关的特征向量$\mathbf{u}_i$在相邻节点间快速振荡，代表具有高变化的信号。

因此，GFT系数$\hat{x}_i$表示信号在与$\lambda_i$相关频率处分量的强度。集中在低频率（小$\lambda_i$对应大$\hat{x}_i$）的信号在图上是平滑的，而具有显著高频分量的信号在邻居之间变化剧烈。



![图特征值（频率）示例](plots/2664-0.json)



> 图拉普拉斯算子特征值的典型分布，显示了从零（最平滑）到代表更高频率的较大值的范围。

## 图滤波

图滤波涉及修改图信号的谱系数。图滤波器通过放大或衰减由拉普拉斯算子特征值定义的特定频率分量来操作。

### 谱域滤波

线性、移不变图滤波器由一个函数$h(\lambda)$定义，称为滤波器的频率响应。在谱域中将此滤波器应用于信号$\mathbf{x}$，意味着将每个傅里叶系数$\hat{x}_i$乘以滤波器在对应频率$\lambda_i$处的响应$h(\lambda_i)$。

如果$\mathbf{y}$是滤波后的信号，其GFT $\hat{\mathbf{y}}$由下式给出：


$$
\hat{y}_i = h(\lambda_i) \hat{x}_i
$$


以矩阵形式，这可以写为：


$$
\hat{\mathbf{y}} = h(\mathbf{\Lambda}) \hat{\mathbf{x}}
$$


其中$h(\mathbf{\Lambda})$是一个对角矩阵，其对角线上是$h(\lambda_i)$：$h(\mathbf{\Lambda}) = \text{diag}(h(\lambda_1), h(\lambda_2), \dots, h(\lambda_N))$。

### 空间域等效性（卷积）

要查看滤波器如何在节点（空间）域中直接作用于信号$\mathbf{x}$，我们可以使用iGFT将谱滤波操作转换回去：


$$
\mathbf{y} = \mathbf{U} \hat{\mathbf{y}} = \mathbf{U} h(\mathbf{\Lambda}) \hat{\mathbf{x}} = \mathbf{U} h(\mathbf{\Lambda}) \mathbf{U}^T \mathbf{x}
$$


这定义了空间域中的图卷积操作：


$$
\mathbf{y} = (\mathbf{U} h(\mathbf{\Lambda}) \mathbf{U}^T) \mathbf{x} = \mathbf{H} \mathbf{x}
$$


矩阵$\mathbf{H} = \mathbf{U} h(\mathbf{\Lambda}) \mathbf{U}^T$表示节点域中的滤波器操作。

这里的一个主要挑战是直接计算此操作需要：

1. 计算拉普拉斯算子的完整特征分解（$\mathbf{U}$，$\mathbf{\Lambda}$），其代价是$O(N^3)$。
2. 执行涉及$\mathbf{U}$和$\mathbf{U}^T$的矩阵乘法，每个信号向量 (vector)的代价是$O(N^2)$。
3. 由此产生的滤波器矩阵$\mathbf{H}$通常是稠密的，这意味着节点$v_i$处的滤波值$y_i$取决于*所有*其他节点$v_j$处的信号值$x_j$，这使得该操作是非局部的，对于大型图而言计算成本高昂（$O(N^2)$的应用成本）。

## 连接图滤波器和GNN层

GSP框架为理解谱GNN提供了基础。这些模型明确或隐式地定义图滤波器并学习其参数 (parameter)。

### 作为滤波器的谱GNN

早期的谱GNN，例如Bruna等人（2014）提出的模型，直接对滤波器响应$h(\mathbf{\Lambda})$进行参数 (parameter)化。一个谱GNN层可以被视为将可学习的图滤波器$\mathbf{H}_{\theta} = \mathbf{U} h_{\theta}(\mathbf{\Lambda}) \mathbf{U}^T$应用于输入节点特征（信号）$\mathbf{X}^{(k-1)}$：


$$
\mathbf{X}^{(k)} = \sigma \left( \mathbf{U} h_{\theta}(\mathbf{\Lambda}) \mathbf{U}^T \mathbf{X}^{(k-1)} \right)
$$


这里，$h_{\theta}(\mathbf{\Lambda})$是一个对角矩阵，其项取决于可学习的参数$\theta$。非线性函数$\sigma$随后按元素应用。

为了克服通用谱滤波器$\mathbf{H}_{\theta}$的计算成本和非局部性问题，后续工作集中于设计能够高效计算且空间局部化的滤波器$h_{\theta}(\lambda)$。

- **多项式滤波器：** 一种常见方法是使用特征值的多项式来近似滤波器响应$h_{\theta}(\lambda)$，即$h_{\theta}(\lambda) \approx \sum_{p=0}^P \theta_p \lambda^p$。由于$\mathbf{U} \mathbf{\Lambda}^p \mathbf{U}^T = (\mathbf{U} \mathbf{\Lambda} \mathbf{U}^T)^p = L^p$，这导致了涉及拉普拉斯算子幂次的滤波器操作：
  
  $$
  \mathbf{y} \approx \left( \sum_{p=0}^P \theta_p L^p \right) \mathbf{x}
  $$
  
  将$L^p$应用于$\mathbf{x}$只涉及$p$跳内的节点，这使得滤波器是$P$-局部化的。ChebNet（使用切比雪夫多项式）和简化的图卷积网络（GCN）（使用一阶近似，$P=1$）是属于此类别的重要例子。它们学习多项式系数$\theta_p$。

### 滤波器属性与GNN行为

GSP有助于分析滤波器结构如何影响GNN行为：

- **低通滤波器：** 许多标准GNN层，例如GCN，充当低通滤波器。它们通过平均来自邻居的信息来平滑节点特征。这对于节点分类等任务通常是有益的，因为邻近节点往往具有相似的标签。
- **过平滑：** 在许多GNN层中重复应用低通滤波可能导致过平滑。信号中的高频分量（可能包含区分信息）会逐渐衰减。最终，节点表示可能变得过于相似，从而妨碍性能。GSP提供工具，通过分析滤波器的频率响应来量化 (quantization)这种效应。
- **带通或高通滤波器：** 尽管作为主要机制不太常见，但设计能够保留或强调更高频率的滤波器可能与局部变化或邻居之间差异很重要的任务相关。

## 对GNN设计和分析的影响

GSP视角带来多项益处：

1. **原理性理解：** 它为谱GNN架构为何有效以及它们与经典信号处理的关系提供了理论基础。
2. **分析工具：** 它允许通过检查层的频率响应来分析GNN属性，例如过平滑。我们可以研究不同的消息传递方案如何隐式地过滤图信号。
3. **滤波器设计：** GSP原理可以指导设计具有特定所需属性的新GNN层（例如，避免过度平滑或捕获与任务相关的特定频段的层）。
4. **连接谱域与空间域：** 尽管空间GNN（如GraphSAGE、GAT）并非直接通过谱滤波器定义，但其操作通常可以在GSP框架内进行近似分析，有助于理解它们在信号平滑或特征变换方面的行为。

尽管由于计算成本高昂，任意谱滤波器的直接实现对于大型图仍然具有挑战，但图信号处理的框架对于理解许多高级GNN架构背后的机制以及设计下一代图学习模型是不可或缺的。它提供了一个视角，通过该视角，消息传递的聚合和更新步骤可以被解释为对图上信息的复杂滤波操作。

## 参考资料

- [The Emerging Field of Signal Processing on Graphs: Extending High-Dimensional Data Analysis to Networks and Other Irregular Domains](https://ieeexplore.ieee.org/document/6494793) — David I. Shuman, Benjamin Ricaud, Pierre Vandergheynst (2013)
  Journal: IEEE Signal Processing Magazine; Volume: 30; Pages: 83-98; DOI: [10.1109/MSP.2012.2235192](https://doi.org/10.1109/MSP.2012.2235192)
  介绍图信号处理核心概念（包括图拉普拉斯算子、图傅里叶变换和谱图滤波）的开创性综述。
- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](http://geometricdeeplearning.com/) — Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković (2021)
  Publisher: MIT Press
  一本全面和现代的教科书，为几何深度学习提供了统一的理论框架，广泛涵盖图信号处理作为图神经网络的基础。
- [Spectral Networks and Deep Locally Connected Networks on Graphs](https://arxiv.org/abs/1312.6203) — Joan Bruna, Wojciech Zaremba, Arthur Szlam, Yann LeCun (2014)
  Journal: ICLR 2014 Workshop; DOI: [10.48550/arXiv.1312.6203](https://doi.org/10.48550/arXiv.1312.6203)
  一项开创性工作，引入了谱图卷积，将图信号处理概念与图上的深度学习架构连接起来。
- [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://neurips.cc/Conferences/2016/Schedule?showEvent=6094) — Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst (2016)
  Journal: Advances in Neural Information Processing Systems 29; Publisher: NeurIPS; Pages: 3844-3852
  引入切比雪夫多项式来有效近似图卷积滤波器，实现局部化和可扩展的谱图神经网络（ChebNet）。
