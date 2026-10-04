# 图卷积网络 (GCN)

来源：[原文](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-3-foundational-gnn-architectures/graph-convolutional-networks-gcn)

[返回章节目录](README.md) · [返回课程目录](../README.md)

图卷积网络 (GCN) 是一种基础架构，它为消息传递框架提供了一种具体且高效的实现方式。它借鉴了计算机视觉中的卷积操作，并将其改进以适应图的非规则、非欧几里得结构。CNN 卷积核在固定的像素网格上滑动，而 GCN 层则处理来自节点局部图邻域的信息。

GCN 层的主要操作可以用一个简洁的传播规则来表示，该规则能够同时更新图中所有节点的特征。

### GCN 传播规则

对于 GCN 层，从当前层输入特征 $H^{(l)}$ 生成下一层输出特征 $H^{(l+1)}$ 的过程由以下公式定义：


$$
H^{(l+1)} = \sigma(\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})
$$


这个方程起初看起来可能很复杂，但每个部分都有其明确且易于理解的作用。我们逐个变量进行分析：

- $H^{(l)}$ 是第 $l$ 层的节点嵌入 (embedding)矩阵。每一行对应一个节点，每一列代表一个特征。对于第一层 ($l=0$)，它就是初始节点特征矩阵 $X$。
- $W^{(l)}$ 是该层的可训练权重 (weight)矩阵。这是传统神经网络 (neural network)中的标准组件。它的任务是对节点特征进行线性变换，使模型能够学习并提取相关的模式。其形状为 `[输入特征数, 输出特征数]`。
- $\sigma$ 是非线性激活函数 (activation function)（如 `ReLU`），应用于每个元素。与其他神经网络一样，它引入了非线性，使模型能够学习更复杂的联系。

GCN 公式中最特别的部分是 $\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}$。这一部分负责核心的图卷积操作，是根据图的结构构建的：

1. **$\hat{A} = A + I_N$**：这是图的邻接矩阵 $A$ 加上单位矩阵 $I_N$ 后的结果，即添加了自环。添加自环是一个简单但重要的修改。它确保了当节点聚合邻居特征时，也会包含自己上一层的特征信息。如果没有这一步，节点自身的表征在更新时会被忽略。
2. **$\hat{D}$**：这是 $\hat{A}$ 的对角度矩阵。每个对角线元素 $\hat{D}_{ii}$ 包含节点 $i$ 的度（包括其自环）。所有非对角线元素均为零。
3. **对称归一化 (normalization)**：完整项 $\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}$ 对邻接矩阵进行了对称归一化。这一步对于稳定训练非常重要。如果只乘以 $\hat{A}$，则会简单地累加邻居节点的特征向量 (vector)。然而，这会导致度数非常高或非常低的节点出现问题。高度数节点的嵌入可能会呈指数级增长，而低度数节点的嵌入可能会缩小，从而导致梯度不稳定。通过度矩阵 $\hat{D}^{-\frac{1}{2}}$ 进行归一化，可以有效地对邻居消息取平均，防止节点嵌入的规模受节点度数的影响而产生偏差。

下图说明了单个节点 $v$ 的 GCN 更新过程。它的新特征 $h'_v$ 是通过聚合其自身之前的特征 $h_v$ 以及邻居特征 $h_{u1}$、$h_{u2}$ 和 $h_{u3}$ 计算得出的。

> 节点 $v$ 的更新涉及对第 $l$ 层局部邻域特征的归一化求和，随后进行线性变换和非线性激活，从而产生其在第 $l+1$ 层的新特征。

### 将 GCN 映射到消息传递框架

GCN 公式提供了前一章讨论的 `AGGREGATE`（聚合）和 `UPDATE`（更新）步骤的一个具体实例。GCN 层的优点在于它将这些步骤合并为一个高效的矩阵乘法。

- **聚合：** 乘以 $\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}$ 执行了聚合操作。对于每个节点，此操作收集其邻居（及其自身）的特征，计算归一化 (normalization)总和，并生成聚合消息。这是一种加权平均，权重 (weight)由源节点和目标节点的度数决定。
- **更新：** 随后乘以权重矩阵 $W^{(l)}$ 并应用激活函数 (activation function) $\sigma$ 构成了更新步骤。此步骤将聚合的消息转换为节点在下一层的新嵌入 (embedding)。

通用的消息传递方案通常将这些描述为独立的函数，而 GCN 则将它们合并为一次操作。这使其非常高效，尤其是在使用稀疏矩阵乘法库实现时。

### 优势与局限性

由于其简单性和有效性，GCN 被广泛用作许多图学习任务的起点。

**优势：**

- **效率高：** 该公式依赖于稀疏矩阵乘法，使其在计算上非常高效，并可扩展到大规模图。
- **简单：** 该模型易于实现，且与更复杂的架构相比，参数 (parameter)相对较少。
- **强大的基准：** GCN 在许多节点分类基准测试中表现出色，通常作为比较时的强力基准。

**局限性：**

- **转导性限制：** 标准的 GCN 公式在每次前向传播时都需要完整的图邻接矩阵。这意味着整个图，包括验证集和测试集中的节点，在训练期间都必须存在。这使得它本质上是转导性的，不容易推广到训练期间未见过的节点。
- **各向同性聚合：** GCN 聚合机制（归一化 (normalization)后）对所有邻居一视同仁。它无法为不同的邻居分配不同的权重 (weight)，这一局限性已被图注意力网络 (GAT) 解决。
- **过度平滑：** 当堆叠多个 GCN 层时，节点表征可能会变得过于相似，趋向于同一个值。这种“过度平滑”问题使得构建非常深的 GCN 模型变得困难。

尽管存在这些局限性，图卷积网络仍是 GNN 领域的常用模型。它的公式为从抽象的消息传递想法到用于图上学习的实用算法提供了清晰的桥梁。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf, Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍基础的图卷积网络 (GCN) 架构及其用于图半监督学习的传播规则。
- [Graph Neural Networks: A Comprehensive Introduction](https://www.morganclaypool.com/doi/abs/10.2200/S01004ED1V01Y202003AIM006) — William L. Hamilton (2020)
  Publisher: Morgan & Claypool Publishers; DOI: [10.2200/S01004ED1V01Y202003AIM006](https://doi.org/10.2200/S01004ED1V01Y202003AIM006)
  对图神经网络提供了易懂且深入的介绍，包含专门章节讲解 GCN 及其在该领域的地位。
- [Relational Inductive Biases, Deep Learning, and Graph Networks](https://arxiv.org/abs/1806.01261) — Peter W. Battaglia, Jessica B. Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, Caglar Gulcehre, Francis Song, Andrew Ballard, Justin Gilmer, George Dahl, Ashish Vaswani, Kelsey Allen, Charles Nash, Victoria Langston, Chris Dyer, Nicolas Heess, Daan Wierstra, Pushmeet Kohli, Matt Botvinick, Oriol Vinyals, Yujia Li, Razvan Pascanu (2018)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1806.01261](https://doi.org/10.48550/arXiv.1806.01261)
  将包括 GCN 在内的多种图神经网络模型统一在信息传递框架下。

---

[上一节](../02-%E6%B6%88%E6%81%AF%E4%BC%A0%E9%80%92%E6%9C%BA%E5%88%B6/07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20NumPy%20%E5%AE%9E%E7%8E%B0%E7%AE%80%E5%8D%95%E7%9A%84%20GNN%20%E5%B1%82.md) · [下一节](02-%E5%9B%BE%E5%8D%B7%E7%A7%AF%E7%9A%84%E7%A9%BA%E9%97%B4%E7%89%A9%E7%90%86%E8%A7%A3%E8%AF%BB.md)
