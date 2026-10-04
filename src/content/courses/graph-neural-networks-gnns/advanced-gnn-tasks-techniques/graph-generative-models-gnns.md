---
course: "graph-neural-networks-gnns"
chapter: "advanced-gnn-tasks-techniques"
lesson: "graph-generative-models-gnns"
sourceId: 2732
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-4-advanced-gnn-tasks-techniques/graph-generative-models-gnns"
title: "图生成模型"
description: "学习使用基于GNN的方法（例如自编码器、GANs）生成新图结构的方法。"
order: 4
plots: []
sourceHash: "76d2483689d74c9598d9433c0c2a7dc697f733627debb6a077c96cec3cfb022c"
sourceCorrections: []
---

虽然前面章节主要关注现有图数据的分析和学习，但高级图机器学习 (machine learning)的一个重要方向是*生成*类似图的新图结构。图生成模型旨在从一组已观测的图中学习一个潜在的概率分布，然后从这个学到的分布中采样新的图。这项能力对于发现新颖的分子结构、模拟社交网络演变、生成合成图数据用于基准测试或数据增强，以及理解图形成的基本规律等任务都很重要。

与生成图像或文本相比，生成图具有独特的挑战。图是离散结构，通常具有可变数量的节点和边。此外，图的表示并非唯一；不同的邻接矩阵可以表示同一个图（排列不变性），这使得学习变得复杂。图神经网络 (neural network)在许多现代图生成模型中扮演核心角色，通过有效地捕捉图中复杂的结构依赖关系。

### 图生成方法

构建图生成模型存在几种模式，通常采用GNN作为核心组成部分：

1. **序列式（自回归 (autoregressive)）生成：** 这类模型逐步生成图，通常是逐节点或逐边进行。在每个步骤中，模型决定是添加新节点、在现有节点间添加边，还是停止生成过程。

   - **GNN的运用方式：** GNN通常用于处理每个步骤中部分生成的图。GNN的输出提供了一种状态表示，用于条件化预测下一个动作（例如，连接哪些节点）。这使得模型能够基于当前的全局和局部结构做出决定。
   - **示例模型：** GraphRNN及其变体生成节点和边添加的序列，通常使用RNN结合GNN式的中间图状态处理。它们将图$G$的概率建模为生成步骤条件概率的乘积：$p(G) = \prod_i p(\text{动作}_i | \text{图}_{i-1})$。
   - **挑战：** 管理生成序列的复杂性和潜在的错误传播。在生成过程中捕获长距离依赖关系也可能很困难。

digraph G {
rankdir=LR;
node [shape=box, style=rounded, fontname="sans-serif", fillcolor="#e9ecef", style=filled];
edge [fontname="sans-serif"];

```
Start [label="开始生成"];
Step1 [label="生成节点1"];
Step2 [label="生成节点2\n(决定边1-2)"];
Step3 [label="生成节点3\n(决定边1-3, 2-3)"];
Stop [label="满足停止条件"];

GNN [shape=ellipse, label="GNN状态\n更新", fillcolor="#a5d8ff", style=filled];
Decision [shape=diamond, label="下一步动作?", fillcolor="#ffec99", style=filled];

Start -> GNN;
GNN -> Decision;
Decision -> Step1 [label="添加节点"];
Step1 -> GNN;
Decision -> Step2 [label="添加节点/边"];
Step2 -> GNN;
Decision -> Step3 [label="添加节点/边"];
Step3 -> GNN;
Decision -> Stop [label="停止"];
```

}
\`\`\`
> 自回归图生成过程的高级流程，其中GNN在每次结构添加后更新状态。

2. **图的变分自编码器（VAEs）：** VAEs提供了一个概率框架来学习潜在表示。对于图，VAEs包括：

   - **编码器：** 一个GNN（例如GCN或GAT）将输入图$G$映射到连续潜在空间中某个分布的参数 (parameter)（均值$\mu$和方差$\sigma^2$），即$q(z|G) = \mathcal{N}(z | \mu, \sigma^2 I)$。
   - **解码器：** 一个函数（也可能包含GNN原理或简单的多层感知机）将从潜在分布中抽取的样本$z$映射回图结构，定义$p(G|z)$。这通常涉及生成邻接矩阵概率或边列表概率。
   - **训练：** 模型通过最大化证据下界（ELBO）进行训练：
     
     $$
     \mathcal{L} = \mathbb{E}_{q(z|G)}[\log p(G|z)] - KL(q(z|G) || p(z))
     $$
     
     其中$p(z)$是先验分布（通常是标准高斯分布$\mathcal{N}(0, I)$），$KL$表示Kullback-Leibler散度。第一项鼓励准确重建，而第二项则作为正则化 (regularization)项。
   - **示例模型：** GraphVAE, VGAE（变分图自编码器）。
   - **挑战：** 解码器必须从连续潜在变量$z$生成离散的图结构（通常是邻接矩阵）。这通常需要对概率进行阈值处理或使用Gumbel-Softmax技巧等方法，以允许梯度流经采样过程。确保排列不变性也需要细致的设计。
3. **图的生成对抗网络 (GAN)（GANs）：** GANs采用生成器和判别器之间的双人博弈。

   - **生成器（G）：** 接收随机噪声$z$（从先验分布中采样）作为输入，并输出图结构$\hat{G}$。生成器可能使用MLP或适用于图的转置卷积，有时会加入GNN层。
   - **判别器（D）：** 一个基于GNN的分类器，训练用于区分训练数据集中的真实图$G$和由$G$生成的虚假图$\hat{G}$。判别器输出输入图是真实的概率。
   - **训练：** $G$和$D$进行对抗训练。$D$学习最大化其分类准确性，而$G$学习最小化$D$区分虚假图的能力，有效地试图欺骗$D$。
   - **示例模型：** MolGAN（用于分子图）, NetGAN。
   - **挑战：** 图生成的离散性质使得梯度难以从判别器回流到生成器。强化学习 (reinforcement learning)（将图生成步骤视为动作）或放宽离散选择等近似方法通常是必需的。为判别器定义对细微结构差异敏感的有效GNN架构也十分重要。

digraph GAN {
rankdir=LR;
node [shape=box, style=rounded, fontname="sans-serif", fillcolor="#e9ecef", style=filled];
edge [fontname="sans-serif"];

```
Noise [label="随机噪声 (z)", shape=ellipse, fillcolor="#bac8ff", style=filled];
Generator [label="生成器 (G)\n(输出图)", fillcolor="#b2f2bb", style=filled];
FakeGraph [label="生成的图 (Ĝ)", shape=Mdiamond, fillcolor="#fcc2d7", style=filled];
RealGraph [label="真实图 (G)", shape=Mdiamond, fillcolor="#96f2d7", style=filled];
Discriminator [label="判别器 (D)\n(GNN分类器)", fillcolor="#ffd8a8", style=filled];
Decision [label="真实 / 虚假?", shape=circle, fillcolor="#ffec99", style=filled];

Noise -> Generator;
Generator -> FakeGraph;
RealGraph -> Discriminator;
FakeGraph -> Discriminator [style=dashed];
Discriminator -> Decision;
```

}
\`\`\`
> 使用GAN进行图生成的对抗性设置。生成器创建图，判别器（一个GNN）尝试区分真实图和生成的图。

### 生成图的评估

评估生成图的质量并非易事。常见的评估方法包括：

- **比较图统计量：** 测量生成图和真实图之间节点度数、聚类系数、基元计数或谱性质（例如拉普拉斯算子的特征值）等属性的分布。
- **任务特定评估：** 如果图是为特定目的（例如分子生成）而生成的，则评估生成分子的性质（例如有效性、类药性、可合成性）或它们在下游预测任务上的表现。
- **目视检查：** 对于较小的图有用，但主观且不可扩展。

### 应用场景

由GNN驱动的图生成模型已在以下方面得到应用：

- **药物发现和材料科学：** 生成具有所需化学或物理性质的新颖分子结构。
- **社交网络建模：** 模拟网络增长并理解涌现 (emergence)性质。
- **数据增强：** 创建合成图数据以扩充小型数据集，用于训练其他GNN模型。
- **异常检测：** 在正常图上训练的模型可以对异常图结构赋予较低的似然性。
- **物理模拟：** 模拟物理系统中的相互作用和结构。

图生成模型代表了GNN的一个高级应用，扩展了图数据可以实现的功能。尽管仍存在挑战，特别是在可扩展性、离散生成和评估方面，但通过计算生成真实且新颖图结构的能力，为各个科学和工业范畴带来了重要机会。理解这些技术使您不仅能够分析，还能创建复杂的关联数据。

## 参考资料

- [GraphRNN: Generating Realistic Graphs with Deep Generative Models](https://proceedings.mlr.press/v80/you18a.html) — Jiaxuan You, Rex Ying, Xiang Ren, William Hamilton, Jure Leskovec (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning; Publisher: PMLR; Volume: 80; Pages: 5708-5717; DOI: [10.5555/3326922.3326932](https://doi.org/10.5555/3326922.3326932)
  本文介绍了GraphRNN，一种用于生成真实图的开创性自回归模型。
- [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) — Thomas N. Kipf, Max Welling (2016)
  Journal: arXiv preprint arXiv:1611.07308; DOI: [10.48550/arXiv.1611.07308](https://doi.org/10.48550/arXiv.1611.07308)
  这篇基础论文提出了变分图自动编码器（VGAE），一个早期且有影响力的学习图潜在表示的模型。
- [MolGAN: An Implicit Generative Model for Small Molecular Graphs](https://arxiv.org/abs/1805.11973) — Nicola De Cao, Thomas Kipf (2018)
  Journal: ICML 2018 workshop on Theoretical Foundations and Applications of Deep Generative Models; DOI: [10.48550/arXiv.1805.11973](https://doi.org/10.48550/arXiv.1805.11973)
  本文介绍了MolGAN，一种使用GANs生成分子图的隐式生成模型，解决了离散图结构的挑战。
- [A Survey on Graph Generative Models](https://jmlr.org/tmlr/papers/v21/21-03756.html) — Lin Cai, Junran Wu, Carl Yang (2021)
  Journal: Transactions on Machine Learning Research (TMLR); Publisher: Transactions on Machine Learning Research; DOI: [10.48550/TMLR.2021.03756](https://doi.org/10.48550/TMLR.2021.03756)
  这篇综述全面回顾了各种图生成模型，对不同方法进行分类，并讨论了挑战和应用。
- [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) — William L. Hamilton (2020)
  Journal: Synthesis Lectures on Artificial Intelligence and Machine Learning; Publisher: Morgan & Claypool Publishers; Volume: 14; Pages: Chapter 9; DOI: [10.2200/S01045ED1V01Y202009AIM046](https://doi.org/10.2200/S01045ED1V01Y202009AIM046)
  本书第9章详细介绍了图生成模型，包括序列、VAE和GAN方法。
