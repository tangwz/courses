# GraphSAGE 的归纳学习

来源：[原文](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-3-foundational-gnn-architectures/inductive-learning-graphsage)

[返回章节目录](README.md) · [返回课程目录](../README.md)

GraphSAGE 架构的一个显著优势是它具有执行归纳学习（Inductive Learning）的能力。这种能力解决了许多图神经网络 (neural network)模型（包括标准图卷积网络 GCN）中存在的局限性，使得 GraphSAGE 能够泛化到训练阶段未曾出现的节点上。

### 转导式学习与归纳式学习的设置

为了理解 GraphSAGE 的作用，首先需要区分图机器学习 (machine learning)中的两种设置：

- **转导式学习 (Transductive Learning)：** 在这种设置下，模型在训练期间可以访问整个图，包括所有的节点和边。任务是推断该单一固定图中未标记 (token)节点的标签或属性。转导式模型为训练时所针对的具体节点学习嵌入 (embedding)。如果以后在图中添加了新节点，模型在不进行完全重新训练的情况下，无法为该新节点生成嵌入。
- **归纳式学习 (Inductive Learning)：** 在这种设置下，模型在一组图或一个子图上进行训练，然后预期能够对新的、以前未见过的节点甚至完全陌生的图做出预测。归纳式模型学习的是一个通用函数，通过将节点的局部邻域结构和特征映射到向量 (vector)表示来生成嵌入。该函数可以应用于任何节点，无论它是旧的还是新的。

### GCN 的转导式局限性

标准 GCN 本质上是转导式的。GCN 的传播规则是使用图的归一化 (normalization)邻接矩阵 $\hat{A}$ 来定义的。


$$
H^{(l+1)} = \sigma(\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})
$$


这种形式依赖于 $\hat{A}$ 中捕捉到的图的全局结构。如果出现新节点，邻接矩阵的维度和数值就会发生变化，从而使学到的模型失效。GCN 学习的是特定于训练图中节点的嵌入 (embedding)，而不是生成嵌入的通用函数。

### GraphSAGE 如何实现归纳学习

GraphSAGE 专门为归纳学习而设计。它通过将重点从学习特定节点的嵌入 (embedding)转移到学习一组**聚合函数**来实现这一点。这些函数学习如何从节点的局部邻域收集并处理信息。

GraphSAGE 归纳能力的核心在于两个设计选择：

1. **学习通用聚合函数：** GraphSAGE 不使用像 GCN 那样的固定传播规则，而是学习带参数 (parameter)的聚合函数，例如 `均值 (mean)`、`最大池化 (max-pooling)` 或 `LSTM` 聚合器。模型学习这些函数的权重 (weight)，这些权重定义了一套可重用的方案，规定了任何节点应该如何聚合来自其邻居的信息，而不管该节点的身份或其在大图中的位置如何。
2. **仅依赖局部邻域：** 节点 `v` 的嵌入仅使用其自身的特征及其直接邻居的特征来计算。计算过程不依赖于整个图矩阵。当引入新节点时，GraphSAGE 可以通过采样其新的局部邻域并应用已训练好的聚合函数，即时生成其嵌入。

为任何节点 `v`（无论是旧节点还是新节点）生成嵌入的过程如下：

1. 对节点 `v` 的固定大小的邻域进行采样。
2. 将学到的聚合函数应用于采样邻居的特征，以产生聚合后的邻域向量 (vector)。
3. 将节点 `v` 自身的特征向量与聚合后的邻域向量进行拼接。
4. 通过带有非线性激活函数 (activation function)的全连接层处理拼接后的向量，生成节点 `v` 的最终嵌入。

由于聚合函数是可迁移的，模型可以在一个图上训练，然后用于为完全不同的图中的节点生成嵌入，只要这些节点共享相同的特征空间即可。

> 转导式方法学习的是与特定图绑定的嵌入，如果图发生变化，则需要重新训练。归纳式方法学习的是一个通用函数，无需重新训练即可为新节点或全新的图生成嵌入。

### 归纳学习的实际应用

GraphSAGE 的归纳特性对于图结构动态变化或规模巨大的系统非常有用。

- **动态社交网络：** 在 Twitter 或 LinkedIn 等平台上，每秒都会出现新用户和新连接。归纳式模型可以立即为这些新用户生成表示，使他们能够毫无延迟地整合到推荐和内容过滤系统中。
- **不断演化的知识图谱：** 随着新研究的发表，实体（如蛋白质或药物）及其关系会被添加到生物医学知识图谱中。GraphSAGE 可以更新或生成这些新实体的嵌入 (embedding)，以辅助药物重定向等任务。
- **大规模推荐系统：** 对于像亚马逊这样拥有数百万产品的平台，频繁地在完整的用户-物品交互图上重新训练 GCN 是不切实际的。归纳式模型可以在子图上进行训练，然后在添加新产品或新用户时用于生成其嵌入。

通过学习一种通用的特征聚合函数，GraphSAGE 为在动态和不断变化的环境中应用图表示学习提供了一种可扩展且灵活的方案。这使其成为 GNN 许多大规模工业应用的基础架构。

## 参考资料

- [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) — William L. Hamilton, Rex Ying, Jure Leskovec (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Pages: 1024-1034; DOI: [10.48550/arXiv.1706.02216](https://doi.org/10.48550/arXiv.1706.02216)
  这篇开创性论文提出了GraphSAGE，强调了其通过学习聚合函数实现的归纳学习能力。
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf, Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  这篇原始论文介绍了图卷积网络（GCN），其是图表示学习中基础的转导模型。
- [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) — William L. Hamilton (2020)
  Journal: Synthesis Lectures on Artificial Intelligence and Machine Learning; Publisher: Morgan and Claypool; Volume: 14; Pages: 1-159
  由GraphSAGE的创建者之一撰写，本书全面介绍了图表示学习，涵盖了归纳方法及其理论基础。
- [Relational Inductive Biases, Deep Learning, and Graph Networks](https://arxiv.org/abs/1806.01261) — Peter W. Battaglia, Jessica B. Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, Caglar Gulcehre, Francis Song, Andrew Ballard, Justin Gilmer, George Dahl, Ashish Vaswani, Kelsey Allen, Charles Nash, Victoria Langston, Chris Dyer, Nicolas Heess, Daan Wierstra, Pushmeet Kohli, Matt Botvinick, Oriol Vinyals, Yujia Li, Razvan Pascanu (2018)
  Journal: arXiv preprint arXiv:1806.01261; DOI: [10.48550/arXiv.1806.01261](https://doi.org/10.48550/arXiv.1806.01261)
  这篇有影响力的论文为图网络提供了概念框架，讨论了关系偏差和跨图结构泛化，有助于理解归纳学习原理。

---

[上一节](03-GraphSAGE%EF%BC%9A%E9%82%BB%E5%9F%9F%E9%87%87%E6%A0%B7%E4%B8%8E%E8%81%9A%E5%90%88.md) · [下一节](05-%E5%9B%BE%E6%B3%A8%E6%84%8F%E5%8A%9B%E7%BD%91%E7%BB%9C%20%28GAT%29.md)
