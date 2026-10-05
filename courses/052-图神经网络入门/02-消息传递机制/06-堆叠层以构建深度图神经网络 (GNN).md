# 堆叠层以构建深度图神经网络 (GNN)

来源：[原文](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-2-the-message-passing-mechanism/stacking-gnn-layers)

[返回章节目录](README.md) · [返回课程目录](../README.md)

单个图神经网络 (neural network) (GNN) 层通过考虑节点的直接邻居来处理节点特征。虽然该层是一个强有力的构建模块，但其视角局限于 1 跳邻域。为了获取图中更远部分的信息，我们可以将这些 GNN 层进行堆叠，从而形成深度图神经网络。这种分层技术模拟了标准深度神经网络创建特征层次结构的方式。

### 扩大感受野

堆叠 GNN 层的主要原因是为了扩大每个节点的**感受野**。节点的感受野是指图中能够影响其最终表示的节点集合。

- **1 层之后：** 节点的嵌入 (embedding) $\mathbf{h}_v^{(1)}$ 是其自身初始特征和 1 跳邻居初始特征的函数。它的感受野就是它的直接邻域。
- **2 层之后：** 为了计算 $\mathbf{h}_v^{(2)}$，GNN 使用来自前一层的嵌入，即所有邻居 $u \in \mathcal{N}(v)$ 的 $\mathbf{h}_u^{(1)}$。但每个 $\mathbf{h}_u^{(1)}$ 本身又是根据 $u$ 的邻居的初始特征计算得出的。因此，经过两层之后，来自节点 2 跳邻居的信息已经传播到了该节点。

随着每增加一个 GNN 层，每个节点的感受野就会扩大一跳。因此，具有 $K$ 层的 GNN 可以在相距最多 $K$ 跳的节点之间传播信息。这使得模型能够根据图中更大的子结构来学习特征。

下图说明了这一过程。为了计算节点 A 在两层之后的最终表示，模型首先在第一层聚合其 1 跳邻居（B 和 C）的信息。在第二层中，它聚合 B 和 C 更新后的表示。由于 B 和 C 的表示已经包含了来自它们各自邻居（分别是 D 和 E）的信息，因此节点 A 的最终表示受到了其 2 跳邻居的影响。

> 经过两个 GNN 层后流向节点 A 的信息流。在第二层之后，节点 A 的表示融合了来自节点 D 和 E 的信息，而这两个节点与其相距两跳。

### 深度 GNN 的数学表述

从形式上讲，多层 GNN 通过组合多个消息传递层来工作。第 $l$ 层的输出嵌入 (embedding)（记为 $\mathbf{H}^{(l)}$）成为第 $l+1$ 层的输入嵌入。该过程从初始节点特征 $\mathbf{H}^{(0)} = \mathbf{X}$ 开始。

对于节点 $v$，前两层的计算过程如下：

1. **输入层：** 该过程从原始节点特征开始。
   
   $$
   \mathbf{h}_v^{(0)} = \mathbf{x}_v
   $$
   
2. **第一层 GNN：** 该层通过聚合 1 跳邻居的信息来计算第一组隐藏表示。
   
   $$
   \mathbf{h}_v^{(1)} = \text{更新}^{(1)} \left( \mathbf{h}_v^{(0)}, \text{聚合}^{(1)} \left( \{ \mathbf{h}_u^{(0)} : u \in \mathcal{N}(v) \} \right) \right)
   $$
   
3. **第二层 GNN：** 该层将第一层的嵌入 $\mathbf{h}^{(1)}$ 作为输入，并进行另一轮消息传递。
   
   $$
   \mathbf{h}_v^{(2)} = \text{更新}^{(2)} \left( \mathbf{h}_v^{(1)}, \text{聚合}^{(2)} \left( \{ \mathbf{h}_u^{(1)} : u \in \mathcal{N}(u) \} \right) \right)
   $$
   

此过程重复进行 $K$ 层。每一层的 `聚合` 和 `更新` 函数通常拥有自己的一组可训练参数 (parameter)（例如权重 (weight)矩阵），使得网络能够在不同深度学习 (deep learning)不同的特征变换。第 $K$ 层的最终输出 $\mathbf{h}_v^{(K)}$ 就是用于下游任务的节点嵌入。

### 警示：过度平滑问题

虽然增加深度可以让 GNN 访问更广的图语境，但层数过深也有一个显著的弊端：**过度平滑 (Over-smoothing)**。过度平滑是指在多次消息传递迭代后，连通图中所有节点的表示都趋向于一个相似值的现象。

可以把它想象成向水池中滴入一点彩色染料。搅拌一次（一个 GNN 层）后，颜色会扩散到其附近。搅拌多次后，染料会均匀地扩散到整个水池中，从而无法分辨染料最初的来源。类似地，随着节点特征不断与其邻居进行平均化，它们会丢失初始的、具有辨识性的信息。

当节点嵌入 (embedding)变得难以区分时，模型在节点分类等任务上的表现就会显著下降，因为它不再能区分不同的节点。由于过度平滑，实际中使用的大多数 GNN 架构都相对较浅，通常只包含 2 到 4 层。缓解这一问题是 GNN 研究的一个活跃方向，目前已经产生了通过残差连接或其他机制来保留初始节点信息的复杂架构。

理解这种分层结构是十分基础的。在下一章中，我们将研究 GCN 和 GraphSAGE 等具体且有影响力的架构，它们在多层框架中定义了 `聚合` 和 `更新` 函数的具体形式。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍了图卷积网络（GCN）架构，展示了如何堆叠消息传递层以获取图中节点表示。
- [Graph Representation Learning](https://doi.org/10.2200/S01045ED1V01Y202009AIM046) — William L. Hamilton (2020)
  Journal: Synthesis Lectures on Artificial Intelligence and Machine Learning; Publisher: Morgan & Claypool Publishers; Volume: 14, No. 3; Pages: 1-159; DOI: [10.2200/S01045ED1V01Y202009AIM046](https://doi.org/10.2200/S01045ED1V01Y202009AIM046)
  系统讨论了图表示学习，包括消息传递、通过堆叠层扩展感受野以及过平滑。
- [CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/) — Jure Leskovec (2025)
  一门重要的大学课程，清晰解释了图神经网络、消息传递、节点感受野以及过平滑等实践问题。

---

[上一节](05-%E7%BD%AE%E6%8D%A2%E4%B8%8D%E5%8F%98%E6%80%A7%E4%B8%8E%E7%BD%AE%E6%8D%A2%E7%AD%89%E5%8F%98%E6%80%A7.md) · [下一节](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20NumPy%20%E5%AE%9E%E7%8E%B0%E7%AE%80%E5%8D%95%E7%9A%84%20GNN%20%E5%B1%82.md)
