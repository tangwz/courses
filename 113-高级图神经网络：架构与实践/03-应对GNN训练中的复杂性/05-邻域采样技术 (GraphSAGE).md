# 邻域采样技术 (GraphSAGE)

来源：[原文](https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-3-gnn-training-complexities/neighborhood-sampling-graphsage)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在大型图上训练图神经网络 (neural network)面临不少挑战。在每个训练步骤中处理整个图的邻接矩阵和特征矩阵，这在标准GNN公式中是必需的，但受限于内存和处理时间，在计算上变得过于昂贵。全批量梯度下降 (gradient descent)通常不可行。需要一种训练GNN的方法，使用较小的节点批量，类似于深度学习 (deep learning)模型在大型图像或文本数据集上的训练方式。邻域采样提供了一种有效策略，以实现这种扩展性。

其主要思想简单而有效：在每个GNN层，不是聚合节点*所有*邻居的信息，我们采样固定大小的邻居子集，并仅在这个采样集上执行聚合。这显著减少了每个节点更新所需的计算量。

### GraphSAGE方法

“GraphSAGE（图采样与聚合），由Hamilton、Ying和Leskovec于2017年提出，是一种开创性的、被广泛采用的基于邻域采样的框架。它定义了一种通用的归纳方法，适用于节点特征可用的图。与需要所有节点（包括测试节点）在训练期间都存在的转导方法不同，GraphSAGE在训练后可以为未见过的节点生成嵌入 (embedding)，这使其对动态图非常实用。”

**工作原理：**

GraphSAGE逐层操作。对于一个有$K$层的GNN，计算目标节点$v$的表示涉及以下步骤：

1. **采样：** 在每一层$k$（从$k=1$到$K$），对于需要其表示以计算$k+1$层节点表示的每个节点$u$，采样固定数量（例如$S_k$）的其邻居，表示为$\mathcal{N}_S(u)$。所需节点的集合随着我们从小批量中的目标节点向后遍历而逐层增长。对于第一层（$k=1$），我们为小批量中的目标节点采样邻居。对于第二层（$k=2$），我们需要来自第一层的采样邻居的表示，因此我们采样*它们的*邻居，依此类推。
2. **聚合：** 在每一层$k$，将来自上一层（$k-1$）的采样邻居$\mathcal{N}_S(u)$的表示聚合为一个聚合邻域向量 (vector)$\mathbf{a}^{(k)}_{\mathcal{N}_S(u)}$。GraphSAGE试用了多种聚合函数：

   - **均值聚合器：** 简单地取邻居$(k-1)$层表示的逐元素均值。
     
     $$
     \mathbf{a}^{(k)}_{\mathcal{N}_S(u)} = \text{均值}(\{ \mathbf{h}^{(k-1)}_{v} \mid v \in \mathcal{N}_S(u) \})
     $$
     
   - **池化聚合器：** 将逐元素对称函数（如最大池化或均值池化）应用于变换后的邻居表示。
     
     $$
     \mathbf{a}^{(k)}_{\mathcal{N}_S(u)} = \text{池化}(\{ \sigma(\mathbf{W}_{\text{pool}} \mathbf{h}^{(k-1)}_{v} + \mathbf{b}_{\text{pool}}) \mid v \in \mathcal{N}_S(u) \})
     $$
     
     这里，$\sigma$是非线性函数，$\mathbf{W}_{\text{pool}}$和$\mathbf{b}_{\text{pool}}$是可学习参数 (parameter)。
   - **LSTM聚合器：** 将LSTM网络应用于邻居表示的随机排列，以实现可能更高的表达能力，尽管这会牺牲排列不变性。
3. **更新：** 将节点自身在上一层的表示$\mathbf{h}^{(k-1)}_u$与聚合邻域向量$\mathbf{a}^{(k)}_{\mathcal{N}_S(u)}$结合，以生成节点在当前层$k$的表示。通常，这包括拼接，然后进行线性变换和非线性处理：

   
   $$
   \mathbf{h}^{(k)}_{u} = \sigma \left( \mathbf{W}^{(k)} \cdot \text{拼接}(\mathbf{h}^{(k-1)}_{u}, \mathbf{a}^{(k)}_{\mathcal{N}_S(u)}) \right)
   $$
   

   其中$\mathbf{W}^{(k)}$是层$k$的可学习权重 (weight)矩阵。初始表示$\mathbf{h}^{(0)}_u$通常是节点的输入特征$\mathbf{x}_u$。

**采样过程的可视化：**

考虑在一个2层GNN中计算节点'A'的表示，每层采样2个邻居。

> 使用2层邻域采样（采样大小=2）计算节点'A'的图。节点'A'需要第1层的'B'和'C'。节点'B'和'C'进而需要第2层（表示第0层特征）的采样邻居（'E'、'F'和'G'、'H'）。未采样的邻居（'D'、'I'、'J'）被忽略。

### 实现小批量训练

固定大小的邻域采样是实现小批量训练的重要方法。我们不是处理整个图，而是：

1. 选择一个目标节点的小批量，我们希望为其计算最终嵌入 (embedding)（例如，在节点分类任务中有标签的节点）。
2. 找出所有必需的节点，以计算所有$K$层中小批量节点的嵌入。这涉及逐层递归地获取采样邻居。
3. 仅在这个子图上运行GNN的前向传播，这个子图由小批量节点及其递归采样的邻居构成。
4. 仅计算初始小批量中节点的损失。
5. 执行反向传播 (backpropagation)并更新GNN参数 (parameter)。

这个过程在训练更新时解除了对整个图结构的依赖，使其能够扩展到具有数十亿条边的图。

### 优点与考量

- **可扩展性：** 邻域采样是主要优点，它允许GNN在无法完全载入内存的大规模图上进行训练。
- **归纳能力：** 通过学习在采样邻域上操作的聚合函数，GraphSAGE可以泛化到训练期间未见过的节点。
- **效率：** 每批次的计算量由采样大小（$S_k$）和层数（$K$）控制，而非总图大小$N$。

然而，也存在一些权衡：

- **方差：** 采样会给梯度估计引入方差。较小的采样大小会导致更高的方差但计算更快；较大的采样大小会降低方差但增加成本。
- **信息丢失：** 由于不考虑所有邻居，每次聚合步骤中可能会遗漏一些潜在相关的信息。期望是经过多次训练迭代和批量处理后，模型能学习到有效的聚合函数。
- **采样成本：** 虽然减少了聚合成本，但采样步骤本身会引入额外开销，特别是对于度数非常高的节点。

GraphSAGE和邻域采样原则是用于将GNN应用于大型数据集的一项基本技术。尽管后续方法如GraphSAINT（接下来会讲到）旨在改进采样策略以提高效率或减少方差，GraphSAGE为可扩展的GNN训练奠定了基础。理解其工作方式对于处理大型图问题非常重要。

## 参考资料

- [Inductive Representation Learning on Large Graphs](https://proceedings.neurips.cc/paper/2017/hash/5dd9dbba6f89d36868843901452479d7-Abstract.html) — William L. Hamilton, Zhitao Ying, and Jure Leskovec (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; Pages: 1024-1034
  介绍GraphSAGE的奠基性论文，这是一种通过邻域采样在大规模图上进行归纳表示学习的开创性方法。
- [CS224W: Machine Learning with Graphs](https://cs224w.stanford.edu/) — Jure Leskovec (2023)
  Publisher: Stanford University
  斯坦福大学官方课程材料，提供GNNs的教育内容，包括对GraphSAGE的详细解释。

---

[上一节](04-%E5%A4%84%E7%90%86%E5%A4%A7%E5%9E%8B%E5%9B%BE%EF%BC%9A%E5%8F%AF%E6%89%A9%E5%B1%95%E6%80%A7%E6%8C%91%E6%88%98.md) · [下一节](06-%E5%9B%BE%E9%87%87%E6%A0%B7%E6%8A%80%E6%9C%AF%20%28GraphSAINT%2C%20ShaDow-GNN%29.md)
