---
course: "generative-adversarial-networks-gans"
chapter: "gans-beyond-image-generation"
lesson: "gans-graph-generation"
sourceId: 2764
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-6-gans-beyond-image-generation/gans-graph-generation"
title: "使用GANs生成图"
description: "利用对抗训练生成图结构的方式。"
order: 7
plots: []
sourceHash: "a46d6adca1fc4fa9a7b6603d08e5b994a7284ca6c98ebc4eefeb973393bc0c58"
sourceCorrections: []
---

使用对抗训练生成图，带来了一系列特别的挑战，这与生成图像或音频波形等连续数据有所不同。图本质上是由节点和边定义的离散结构，通常伴随节点或边属性。此外，它们的结构是非欧几里得的，这表示图像GANs中有效使用的标准卷积操作不能直接应用。这些是为图生成任务调整GAN架构所发展出的方法。

### 图的离散性和结构特性

主要难点在于生成离散图结构，特别是邻接矩阵（表示连接）以及可能的节点/边属性。与图像生成中生成器输出固定网格的像素值不同，图生成器必须生成可变大小、置换不变的结构。

1. **离散性:** 生成邻接矩阵涉及二元决定（边是否存在）。将判别器的梯度直接反向传播 (backpropagation)通过这种离散采样过程是有问题的，类似于文本生成中出现的问题。
2. **非欧几里得结构:** 图没有图像的网格状结构。捕捉局部邻域模式和全局图属性需要专门的架构，主要是图神经网络 (neural network)（GNNs）。
3. **置换不变性:** 图的身份不依赖于其节点的顺序。如果生成器输出邻接矩阵，不同的顺序可能表示同一个图。判别器需要对节点置换保持不变，或者生成器必须学习一种标准表示。
   "4. **可变大小:** 图在节点和边的数量上差异很大。标准GAN生成器通常生成固定大小的输出，需要处理可变图大小的机制，例如填充、生成最大尺寸的图，或采用序列生成过程。"

### 图GAN的架构与实现方式

已提出多种方法来应对这些挑战。一种常见模式是在判别器中使用GNNs，以有效处理图结构并向生成器提供有意义的反馈。

**判别器中的图神经网络 (neural network):**
大多数图GAN的核心部分是基于GNN的判别器。判别器将训练数据集中的真实图或生成的图作为输入。它使用GNN层学习捕获局部结构的节点嵌入 (embedding)，并可能聚合这些嵌入（使用图池化或读出函数）以生成一个分数，表示输入图是真实还是伪造。这使得判别器能够评估生成图的结构属性。

> 典型的图GAN设置。生成器从潜在向量 (vector)`Z`生成图表示（例如，邻接矩阵`Adj`和节点特征`Feat`）。基于GNN的判别器学习区分这些生成的图和真实图。

**处理离散输出:**

- **概率生成:** 生成器可能输出边概率矩阵而非离散邻接矩阵。然后可以从这个概率分布中采样边。类似于文本生成中讨论的Gumbel-Softmax松弛等方法有时可以调整使用，尽管它们会增加复杂性。
- **强化学习 (reinforcement learning)（RL）:** 将图生成视为一个序列决策过程（例如，逐步添加节点或边）允许使用强化学习。判别器的分数可以作为奖励信号，用于使用策略梯度等算法训练生成器策略。例如，MolGAN通常使用强化学习来优化特定的化学属性，同时进行对抗训练。
- **中间表示:** 一些模型避免直接离散生成。例如，NetGAN通过学习模仿目标图上的随机游走来生成图。生成器产生合理的随机游走，判别器试图区分来自真实图的游走和生成的游走。然后从生成的游走集合中构建图。

**模型示例:**

- **MolGAN:** 专为生成分子图设计。它通常将对抗训练与强化学习和特定领域的约束结合起来，以确保化学有效性并优化所需属性。生成器通常输出概率邻接张量和节点特征矩阵。
- **NetGAN:** 侧重于生成保留随机游走所捕获结构属性的图。它使用基于LSTMs的生成器生成游走，以及一个判别器（也基于LSTM）对游走进行分类。这最初绕过了邻接矩阵的直接生成。
- **GraphRNN:** 虽然不严格是GAN，但这种自回归 (autoregressive)方法按顺序（逐节点、逐边）生成图，为处理可变大小和依赖性提供了另一种视角。它展现了图生成中经常采用的序列特性。

### 生成图的评估

评估生成图的质量并不简单。评估标准通常涉及将生成图的统计属性与训练集中真实图的统计属性进行比较。常用评估标准包括:

- **度分布:** 比较节点度的分布。
- **聚类系数分布:** 评估局部聚类（三角形）的普遍性。
- **轨道计数/模式分析:** 比较小图模式（基序）的频率。
- **特定领域评估标准:** 对于分子生成等应用，评估标准包括与训练集相比的化学有效性（Val%）、独特性（Unique%）和新颖性（Novelty%），以及对所需属性（例如，药物相似性）的优化分数。

没有单一指标能完美衡量图的质量；通常需要结合多种指标来评估真实性、多样性和结构完整性。

使用GANs生成图仍然是一个活跃的研究方向。离散结构、复杂依赖性和对GNNs等专门架构的需求的结合，使其成为高级生成建模一个具有挑战性但有价值的应用方向。

## 参考资料

- [Molecular Generation with Reinforcement Learning and Graph Convolutional Networks](http://proceedings.mlr.press/v80/jin18a/jin18a.pdf) — Wengong Jin, Regina Barzilay, and Tommi Jaakkola (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning (ICML); Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 80; Pages: 2244-2253; DOI: [10.5555/3326922.3327092](https://doi.org/10.5555/3326922.3327092)
  介绍了一种使用强化学习和图卷积网络生成分子图的方法，旨在优化化学特性。
- [NetGAN: Generating Graphs via Random Walks](http://proceedings.mlr.press/v80/bojchevski18a/bojchevski18a.pdf) — Aleksandar Bojchevski, Gregor Bachmann, Benjamin Rozemberczki, and Stephan Günnemann (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning (ICML); Volume: 80; Pages: 596-605
  提出了一种生成对抗网络模型，通过学习模拟目标图上的随机游走来生成图。
- [GraphRNN: Generating Realistic Graphs with Deep Learning](http://proceedings.mlr.press/v80/you18a/you18a.pdf) — Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, and Jure Leskovec (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning (ICML); Volume: 80; Pages: 5590-5599
  引入了一种自回归模型，用于逐节点和逐边生成真实的图，解决了可变大小和复杂依赖的问题。
