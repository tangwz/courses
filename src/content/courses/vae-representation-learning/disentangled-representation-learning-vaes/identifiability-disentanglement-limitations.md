---
course: "vae-representation-learning"
chapter: "disentangled-representation-learning-vaes"
lesson: "identifiability-disentanglement-limitations"
sourceId: 6369
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-5-disentangled-representation-learning-vaes/identifiability-disentanglement-limitations"
title: "解耦表示学习中的识别问题与局限"
description: "讨论解耦表示学习中的理论限制和识别问题。"
order: 7
plots: []
sourceHash: "c4fedbac3faacc9f30803f1709a0f36bc6db4b7e8183ba8e44452f7bd493830c"
sourceCorrections: []
---

尽管研究解耦表示（其中单个潜在维度与数据中不同的生成因素对应）是一个有吸引力的目标，但它面临着根本性的难题。实现真正的无监督解耦，不仅仅是找到更好的VAE架构或更巧妙的损失函数 (loss function)项。存在着研究人员和实践者必须理解和承认的固有理论限制和识别问题。

### 识别问题：我们能否唯一地恢复真实因素？

解耦学习中的识别问题核心是：仅给定观测数据$X$，模型能否唯一地识别生成$X$的真实潜在生成因素$S = (s_1, s_2, \ldots, s_K)$？或者，更现实地说，在没有$S$监督的情况下，它能否学习到一个潜在表示$Z = (z_1, z_2, \ldots, z_M)$，使得每个$z_i$在置换和缩放的范围内对应于某个$s_j$（或其简单变换）？

在完全无监督的设定中，令人清醒的答案通常是否定的。Locatello 等人（2019）发表的一篇重要论文《Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations》表明，无论模型还是数据，在没有任何归纳偏置 (bias)的情况下，无监督解耦表示学习在理论上是不可能的。本质上，对于任何数据集，可以存在无限多个生成模型（以及相应的潜在表示），它们能够同样好地解释数据，但在其纠缠属性上明显不同。

考虑一个简单情况，数据$X$由两个独立因素生成，例如$s_1$（例如，物体位置）和$s_2$（例如，物体颜色）。一个VAE可能会学习到潜在变量$z_1$和$z_2$。
一个理想的解耦模型将具有$z_1 \approx f_1(s_1)$和$z_2 \approx f_2(s_2)$。
然而，另一个模型可能会以高度纠缠的方式学习到$z'_1 = g_1(s_1, s_2)$和$z'_2 = g_2(s_1, s_2)$，但却可能达到相似的重建质量并满足VAE目标（例如，匹配一个简单的先验分布，如$\mathcal{N}(0, I)$）。

> 上图说明了识别难题。真实的潜在因素（A, B）生成观测数据（X）。编码器将X映射到潜在空间Z。Z的几种配置，包括理想的解耦配置、纠缠配置和线性混合（旋转/缩放）配置，都可能使解码器以相似的保真度重建X并满足先验约束。这种模糊性使得在没有额外假设的情况下，难以保证所学习到的Z与真实因素A和B有意义地对应。

这种非唯一性也延伸到对称性。如果先验$p(Z)$（例如，各向同性高斯分布$\mathcal{N}(0, I)$）和似然$p(X|Z)$对Z的某些变换（如旋转）是不变的，那么模型就没有动力去偏好潜在轴的某种对齐 (alignment)方式而不是另一种，即使因素本身是分离的。这意味着即使VAE学习分离了因素，这些因素也可能在潜在空间中任意旋转，未能满足轴对齐解耦的普遍期望。

### 实际应用中的局限

几种实际局限阻碍了解耦表示的可靠学习。

#### 1. 归纳偏置 (bias)的必要性与不足

由于纯粹的无监督解耦是病态的，所有成功的方法都隐式或显式地依赖于归纳偏置。这些偏置是对数据结构或潜在空间所需属性的假设。

- **基于VAE的偏置**:
  - $\beta$-VAE通过增加KL散度项的权重 (weight)，鼓励因子化的后验$q(Z|X)$与因子化的先验$p(Z)$（通常是各向同性高斯分布）紧密匹配。其隐式偏置是解耦的因素在聚合后验中应统计独立，并且单个潜在维度$z_i$应边际独立。
  - FactorVAE和TCVAE（总相关性VAE）通过针对$q(Z)$的总相关性，更直接地惩罚潜在编码$Z$各维度之间的统计依赖性。
  - 常见的偏置通常倾向于*轴对齐 (alignment)*，即每个生成因素预期映射到单个潜在维度。这是一个可能不适用于所有数据集或因素的强假设。

尽管这些偏置可以促成在某些指标上得分高的表示，但它们并非普遍适用，也无法保证恢复“真实”因素。偏置的选择本身就是一种弱监督形式。

#### 2. 对数据、模型架构和超参数 (parameter) (hyperparameter)的敏感性

所达到的解耦程度对以下方面高度敏感：

- **数据特点**: 如果数据中的真实生成因素高度相关，任何模型都极难在没有关于这种相关结构明确信息的情况下解耦它们。真实因素的数量及其复杂性也起作用。
- **模型架构**: 编码器和解码器网络（例如，图像的卷积层、序列的循环层）的容量和架构可以明显影响所学习到的表示。
- **超参数**: $\beta$-VAE中的$\beta$系数、TCVAE中总相关性惩罚的权重、学习率、批次大小和潜在维度都以复杂的方式相互影响。寻找合适的超参数集通常需要大量的经验调整，并且在不同数据集之间差异很大。例如，一个适用于某个数据集的$\beta$值可能会导致在另一个数据集上重建质量差或过度正则化 (regularization)。

#### 3. 定义和衡量“好”的解耦

如前所述，诸如互信息间隙（MIG）、分离属性可预测性（SAP）、解耦性、完备性和信息性（DCI）等指标，提供了定量评估解耦的方法。然而：

- **指标不一致**: 不同的指标可以产生不同的模型排名，因为它们捕捉了解耦的细微不同方面。
- **依赖真实因素**: 大多数指标需要访问真实的生成因素进行评估，这在无监督情景中通常不可用。
- **主观性**: 什么构成“有意义”或“可解释”的因素可能是主观的且依赖于任务。指标可能不总是与人类对解耦的感知一致。
- **忽略下游任务的信息性**: 一个高度解耦的表示可能不是特定下游任务最有用的，如果在解耦过程中丢失了重要信息。

#### 4. 解耦-重建权衡

许多促成解耦的方法，尤其那些重度惩罚KL散度或总相关性的方法（例如，$\beta$较大的$\beta$-VAE），可能导致一种权衡：

- **解耦程度提升**: 潜在维度变得更加独立，并且可能更具可解释性。
- **重建质量下降**: 模型可能会牺牲其准确重建输入数据的能力，因为潜在空间受到的约束过多。生成的样本可能显得模糊或缺乏细节。
  这种权衡意味着实践者通常必须平衡对可解释潜在变量的追求与需要捕获足够数据信息的表示。

#### 5. 可伸缩性与复杂性

- **高维因素**: 在高维数据（如自然图像或视频）中解耦大量复杂、相互影响的因素，仍然是一个重要难题。
- **潜在维度的选择**: 为$Z$选择正确的维度是困难的。维度过少可能导致纠缠，因为多个真实因素被压缩。维度过多可能导致某些维度被解码器忽略（这些维度出现后验坍缩）或学习到虚假相关性。

#### 6. （隐式）监督的要求

Locatello等人的工作也表明，模型选择、超参数甚至随机种子都可以充当隐式监督者，影响着找到哪种解耦方案（如果有的话）。这表明目前无监督解耦方法的成功可能部分归因于这些隐式选择与基准测试中使用的特定数据集和指标良好匹配。真正稳定且能在不同数据集上泛化而无需如此仔细调整的无监督解耦仍然是一个未解决的问题。

### 实际考量与未来方向

理解这些局限性对于在进行解耦表示学习时设定实际预期非常重要。

1. **认识到没有“一刀切”的方案**: 没有单一的VAE变体或一套超参数 (parameter) (hyperparameter)能够保证在所有数据集和任务上实现完美解耦。
2. **归纳偏置 (bias)很重要**: 仔细考量你的模型（例如，$\beta$-VAE，FactorVAE）施加了哪些归纳偏置，以及它们是否与数据中生成因素的假定结构一致。
3. **结合定量和定性评估**: 不要仅仅依赖指标。视觉检查潜在变量遍历。评估学习到的因素是否真正可解释并对你的特定应用有用。
4. **考量下游任务**: 表示的最终检验通常是其在下游任务上的表现。一个“完美”解耦但分类或生成表现不佳的表示，可能不如一个部分纠缠但表现出色的表示有价值。
5. **弱监督作为务实方法**: 如果某种形式的弱监督可用（例如，部分标签、不变性知识），结合它可以明显缓解识别问题，并引导模型得到更有意义的表示。

该领域正在积极研究克服这些局限的方法，例如，通过融入因果关系的思想，考量更复杂的先验以捕获已知对称性，或开发较少依赖脆弱假设的新学习目标。尽管完美的无监督解耦仍然难以实现，但相关工作持续产出有价值的见解，有助于构建更结构化和可解释的生成模型。

## 参考资料

- [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://proceedings.mlr.press/v97/locatello19a.html) — Alessandro Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schoelkopf, and Olivier Bachem (2019)
  Journal: Proceedings of the 36th International Conference on Machine Learning; Publisher: ML Research Press; Volume: 97; Pages: 4114–4124; DOI: [10.48550/arXiv.1811.00007](https://doi.org/10.48550/arXiv.1811.00007)
  这篇里程碑式的论文从理论上证明了在没有归纳偏置的情况下，无监督解缠学习的不可能性，是本节讨论的可识别性问题的重要参考资料。
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2o-xH9g) — Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner (2017)
  Journal: ICLR; DOI: [10.48550/arXiv.1606.05579](https://doi.org/10.48550/arXiv.1606.05579)
  该论文介绍了 β-VAE，这是一种通过修改 VAE 目标以强调 KL 散度项来学习解缠表示的开创性方法，是归纳偏置的一个重要例子。
- [Isolating Sources of Variation in Disentangled Representation Learning](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFXnL84IWEIn2cZ6wv7DOijNi1Q16gPXkCZJTI2wdb_wI08xvfe0SLzFPxIk1WfmiCVJ8zS0YLxIlroAqTN7iK8zcxwL3MjXdII0088yR5AH4-mymO31QXzljOJkrdloh2zT8lyNIdl6CBHU_gORXB64I9TjMqsa0t1H2taTiRPU2FxSmYUzbhkQUk3QWKGgPbh9p1nUlRedRiGGtJqM4a-STs=) — Ricky T. Q. Chen, Xuechen Li, Roger Grosse, David Duvenaud (2018)
  Journal: Advances in Neural Information Processing Systems; Volume: 31; Pages: 2610–2620; DOI: [10.5555/3295222.3295328](https://doi.org/10.5555/3295222.3295328)
  提出了总相关性变分自编码器（TCVAE），这是一种通过显式惩罚潜在表示的总相关性来促进解缠的高级方法，提供了实现统计独立性的更直接途径。
