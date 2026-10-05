# 图VAEs用于结构化数据表示学习

来源：[原文](https://apxml.com/zh/courses/vae-representation-learning/chapter-6-vaes-sequential-structured-data/graph-vaes)

[返回章节目录](README.md) · [返回课程目录](../README.md)

许多数据集本身就具有结构，这与图像等数据（通常将每个样本独立看待）形成对比。例如社交网络、分子结构或知识库，这些数据很自然地表现为图，由节点（实体）和边（关系）构成。将标准变分自编码器（VAEs）直接应用于这类图数据并不简单，因为它们的大小不一、拓扑结构复杂，且置换不变性很重要（图的含义不会因节点重新排序而改变）。图变分自编码器（Graph VAEs）是一类旨在从图结构数据中学习有意义的表示并生成这类数据的模型。

主要思路是借助图神经网络 (neural network)（GNNs）作为VAE架构中的主要组成部分。GNNs是学习图数据的有效工具，因为它们通过迭代汇集节点局部邻域的信息来运作。这种消息传递机制使GNNs能够学习节点表示，这些表示既能体现节点自身的特征，也能反映其在图中的结构作用。

### 图自编码器 (GAEs) 和变分图自编码器 (VGAEs)

VAEs在图上的一个重要应用是学习节点嵌入 (embedding)和预测链接（即缺失的边）。图自编码器（GAEs）及其变分对应模型，即变分图自编码器（VGAEs），正是为此目的而设计。

典型的VGAE设置如下：

1. **编码器**：一个GNN（例如图卷积网络GCN）接收图的邻接矩阵$A$和节点特征$X$作为输入。对于每个节点$i$，它会为其潜在表示输出参数 (parameter)，一般是高斯分布的均值$\mu_i$和对数方差$\log \sigma_i^2$。因此，节点$i$的潜在向量 (vector)$Z_i$的近似后验为$q(Z_i | X, A) = \mathcal{N}(Z_i | \mu_i, \text{diag}(\sigma_i^2))$。整体近似后验常被假定为在节点上分解：$q(Z | X, A) = \prod_i q(Z_i | X, A)$。
2. **采样**：潜在节点表示$Z_i$通过重参数化技巧进行采样：$Z_i = \mu_i + \sigma_i \odot \epsilon_i$，其中$\epsilon_i \sim \mathcal{N}(0, I)$。
3. **解码器**：解码器旨在从潜在节点嵌入$Z$重建邻接矩阵$A$。一种常用方法是在节点嵌入对之间使用内积（或一个简单的MLP）来预测边的概率：
   
   $$
   p(\hat{A}_{ij}=1 | Z_i, Z_j) = \sigma(Z_i^T Z_j)
   $$
   
   其中$\hat{A}_{ij}$是重建的边概率，$\sigma(\cdot)$是sigmoid函数。

VGAE的ELBO公式如下：


$$
\mathcal{L}_{VGAE} = \sum_{(i,j) \in A \cup A^-} \log p(\hat{A}_{ij} | Z_i, Z_j) - \sum_i KL(q(Z_i|X,A) || p(Z_i))
$$


第一项是邻接矩阵的重建对数似然（经常考虑观测到的边和等量的采样非边$A^-$以保持平衡）。第二项是每个节点潜在表示的近似后验$q(Z_i|X,A)$与先验$p(Z_i)$（一般是标准正态分布$\mathcal{N}(0,I)$）之间的KL散度之和。

> 变分图自编码器（VGAE）的示意图。GNN编码器为每个节点生成潜在参数。采样后，解码器重建图的邻接矩阵。

VGAEs特别适用于社群检测（嵌入相似的节点可能属于同一社群）和不完整图中的链接预测等任务。

### 用于生成完整图的VAEs

除了学习节点嵌入 (embedding)和预测链接之外，一个更具雄心的目标是生成与训练图数据集具有相似特征的全新图。这需要VAEs的潜在变量$z$代表整个图结构，而不仅仅是单个节点。

1. **编码器**：编码器将输入图$G = (X, A)$映射到图级潜在空间$q(z|G)$上的分布参数 (parameter)。这一般涉及：
   - 使用GNN计算所有节点$v \in V$的节点嵌入$H_v$。
   - 应用图池化或读出函数（例如求和、平均，或更复杂的基于注意力机制 (attention mechanism)的方法）将节点嵌入汇集为单个图表示$h_G = \text{READOUT}(\{H_v\}_{v \in V})$。
   - 将$h_G$输入MLP以生成图级潜在变量$z$的均值$\mu_z$和对数方差$\log \sigma_z^2$。
2. **采样**：使用重参数化从$z \sim \mathcal{N}(\mu_z, \text{diag}(\sigma_z^2))$中采样。
3. **解码器**：解码器$p(G'|z)$接收图潜在变量$z$并生成新图$G'$。这是最具难度的一部分。方法包括：
   - **邻接矩阵生成**：MLP可以直接输出扁平化的邻接矩阵，但这对于大型图在置换不变性和可扩展性方面存在困难。
   - **自回归 (autoregressive)生成**：顺序生成节点和边。例如，可以先决定节点数量，然后迭代添加节点并在它们之间预测边，这都以$z$和部分构建的图为条件。GraphRNN或GRAN（图循环注意力网络）等模型就采用此类策略。
   - **一次性生成与细化**：一些模型生成一个原型图结构，然后进行细化。

用于图生成VAEs的ELBO如下：


$$
\mathcal{L}_{GraphVAE} = \mathbb{E}_{q(z|G)}[\log p(G'|z)] - KL(q(z|G) || p(z))
$$


重建项$\log p(G'|z)$可能很复杂，具体取决于图生成方法的定义（例如，边概率的乘积，或顺序生成过程下的似然）。

> 旨在生成全新图结构的VAE架构。编码器为整个图生成一个潜在向量 (vector)，解码器尝试从该向量合成新图。

### 目标函数与架构考量

编码器选择GNN架构（例如GCN、图注意力网络(GAT)、GraphSAGE）会对性能产生很大影响。GATs凭借其注意力机制 (attention mechanism)，在学习节点邻域哪些部分最相关时能发挥特殊作用。

对于图生成中的解码器，确保置换不变性是一个重要考量。如果解码器以特定顺序生成节点或边，模型可能会学习到依赖于顺序的偏差。自回归 (autoregressive)模型经常尝试通过以规范排序为条件或使用置换等变组件来缓解此问题。

图的重建损失需要审慎定义。

- **邻接矩阵重建（VGAE风格）**：对于无权重 (weight)图，常用二元交叉熵；对于有权重图，则常用均方误差。
- **节点特征重建**：如果节点具有特征$X$，可以在ELBO中添加一个额外项以从$Z$重建$X$。
- **完整图生成**：损失可能涉及单个边的概率、节点存在性，甚至更复杂的图相似性度量。

### 图VAEs的应用

图VAEs已在各种方面得到应用：

- **药物发现与分子生成**：图是分子的自然表现形式。图VAEs可以学习生成具有所需化学属性的新颖分子结构。
- **社交网络分析**：建模社群结构、预测未来交互或生成逼真的合成社交网络。
- **推荐系统**：用户-物品交互可以建模为二分图。图VAEs可帮助学习用户和物品嵌入 (embedding)以用于推荐。
- **知识图补全**：预测大型知识图中的缺失关系（边）。
- **生成逼真网络拓扑**：用于物理学、生物学或基础设施规划中的模拟。

### 进阶考量

尽管图VAEs功能强大，但它们也面临自身的一系列难题：

- **可扩展性**：对于非常大的图，GNN计算可能成本很高。图的采样技术或小批量策略是活跃的研究方向。
- **离散结构**：图本质上是离散的。从连续潜在空间生成它们可能很棘手。一些方法使用离散潜在变量（如为图调整的VQ-VAEs）或采用强化学习 (reinforcement learning)技术来处理解码器中的离散决策。
- **动态图**：许多图随时间演变。扩展图VAEs以建模时序图动态是一个持续研究方向。
- **评估**：量化 (quantization)生成图的质量很困难。度量标准经常包括图统计量（度分布、聚类系数）、视觉相似性，或者如果生成的图用于下游任务时的任务特定性能。

图VAEs代表了将生成建模扩展到复杂结构化数据方面的重要进展。通过结合VAEs的概率框架与GNNs的表示能力，它们提供了一个多功能工具集，用于理解、重建和生成各种科学和工业应用中的图数据。随着研究的推进，我们可以期待更高级和可扩展的图VAE模型，能够处理日益复杂的基于图的任务。

## 参考资料

- [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) — Thomas N. Kipf, Max Welling (2016)
  Journal: Bayesian Deep Learning Workshop (NIPS 2016); DOI: [10.48550/arXiv.1611.07308](https://doi.org/10.48550/arXiv.1611.07308)
  介绍了变分图自动编码器（VGAE）模型，这是学习图中节点潜在表示和链接预测任务的核心方法，如本节所述。
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf, Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  介绍了图卷积网络（GCN）架构，这是图 VAEs 中编码器的常见选择，也是许多图神经网络模型的构建模块。
- [GraphRNN: Generating Realistic Graphs with Deep Generative Models](https://arxiv.org/abs/1802.09459) — Jiaxuan You, Zhitao Ying, Xiang Ren, William Hamilton, Jure Leskovec (2018)
  Journal: International Conference on Machine Learning (ICML); Pages: 5571-5580; DOI: [10.48550/arXiv.1802.09459](https://doi.org/10.48550/arXiv.1802.09459)
  一项使用自回归循环神经网络生成完整图的重要工作，展示了如何调整 VAEs 来合成复杂的图结构。
- [Graph Neural Networks: A Review of Methods and Applications](https://doi.org/10.1016/j.aiopen.2021.01.001) — Jie Zhou, Ganqu Cui, Zhengyu Chen, Ming Ding, Shuai Sun, Tianyu Li, Jie Tang (2021)
  Journal: AI Open; Publisher: Elsevier; Volume: 1; Pages: 57-71; DOI: [10.1016/j.aiopen.2021.01.001](https://doi.org/10.1016/j.aiopen.2021.01.001)
  对各种图神经网络（GNN）架构及其应用进行了广泛综述，为图VAE框架中使用的GNN组件提供了额外背景。

---

[上一节](02-%E5%BA%8F%E5%88%97VAE%E7%9A%84%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md) · [下一节](04-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E4%B8%AD%E7%9A%84%20VAE.md)
