---
course: "vae-representation-learning"
chapter: "vaes-mathematical-deep-dive"
lesson: "kl-divergence-vaes"
sourceId: 6293
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-2-vaes-mathematical-deep-dive/kl-divergence-vaes"
title: "VAE中的KL散度：作用与解释"
description: "VAE目标函数中KL散度项的功能、特点及其解读。"
order: 4
plots: []
sourceHash: "622ba5da6f27ae2d8f76a78a6a3f88df6c8a5f3f58eeddb1744ffe25d8cb75e8"
sourceCorrections: []
---

证据下界（ELBO）是训练变分自编码器时最大化的目标函数。它由两部分组成：重构似然和Kullback-Leibler（KL）散度项。
$L_{ELBO}(x) = \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - D_{KL}(q_{\phi}(z|x) || p(z))$
第一项 $\mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)]$ 促使解码器 $p_{\theta}(x|z)$ 从潜在表示 $z$（从编码器输出 $q_{\phi}(z|x)$ 中采样得到）准确重构输入 $x$；而第二项 $D_{KL}(q_{\phi}(z|x) || p(z))$ 则扮演着截然不同且同样重要的角色。下面我们来详细考察这个KL散度项。

### 理解Kullback-Leibler散度

Kullback-Leibler散度，常缩写为KL散度，是衡量一个概率分布与第二个参考概率分布差异的指标。如果我们有两个概率分布 $Q(Z)$ 和 $P(Z)$，从 $Q$ 到 $P$ 的KL散度，记作 $D_{KL}(Q || P)$，量化 (quantization)了使用针对 $P$ 优化的编码时，编码来自 $Q$ 的样本所需的“信息损失”或“额外比特”。需要注意的是，KL散度通常不对称，即 $D_{KL}(Q || P) \neq D_{KL}(P || Q)$；它总是非负的，即 $D_{KL}(Q || P) \ge 0$，当且仅当 $Q = P$ 时等号成立。

对于连续分布，它定义为：


$$
D_{KL}(Q(Z) || P(Z)) = \int Q(Z) \log \frac{Q(Z)}{P(Z)} dZ
$$


在VAE中，该项涉及：

- $q_{\phi}(z|x)$: 这是潜在变量 $z$ 的近似后验分布，由编码器网络（以 $\phi$ 为参数 (parameter)）根据输入 $x$ 产生。它代表我们对对应于 $x$ 的潜在编码 $z$ 的“看法”。通常， $q_{\phi}(z|x)$ 被选为高斯分布，即 $N(\mu_{\phi}(x), \text{diag}(\sigma^2_{\phi}(x)))$，其中均值 $\mu_{\phi}(x)$ 和方差 $\sigma^2_{\phi}(x)$ 是编码器的输出。
- $p(z)$: 这是潜在变量的先验分布。它是我们预先选择的分布，反映了我们在观察任何数据之前对潜在空间结构的假设。一个常见且方便的选择是标准多元高斯分布 $N(0, I)$，其中 $I$ 是单位矩阵。

### $D_{KL}(q_{\phi}(z|x) || p(z))$ 作为正则化 (regularization)项的作用

KL散度项 $D_{KL}(q_{\phi}(z|x) || p(z))$ 对编码器起到正则化作用。通过最小化该项（因为在ELBO中它是减去的，所以最大化ELBO包含最小化KL散度），我们促使编码器为不同输入 $x$ 产生的分布 $q_{\phi}(z|x)$ 平均而言接近先验分布 $p(z)$。

为何这种正则化有益？

1. **有结构的潜在空间：** 它促使编码器学习一个潜在空间，其中编码 $q_{\phi}(z|x)$ 不会最终位于任意的、孤立的区域。相反，它们被促使占据一个“看起来像”先验 $p(z)$ 的区域。如果 $p(z)$ 是 $N(0, I)$，这意味着编码被促使围绕原点分布并具有一定方差。这有助于使潜在空间更加连续和有序。
2. **避免“后验坍塌”（在某个方向上）：** 如果没有这一项，编码器可能会学习使每个 $x$ 的 $q_{\phi}(z|x)$ 非常窄（类似狄拉克函数），从而有效地将输入记忆到潜在空间中的某个特定点。这会使重构完美，但会导致一个高度碎片化的潜在空间，其中从 $p(z)$ 采样的点在解码时可能不对应任何有意义的数据。KL项惩罚 $q_{\phi}(z|x)$ 变得过窄（低方差）或其均值偏离先验均值太远。
3. **实现有意义的生成：** VAE的目标之一是生成新数据。我们通过从先验 $p(z)$ 中采样一个潜在向量 (vector) $z_{new}$，然后将其通过解码器 $p_{\theta}(x|z_{new})$ 来实现。为了使其良好运行，解码器需要使用与从 $p(z)$ 中采样的向量有些相似的潜在向量进行训练。KL散度项确保 $q_{\phi}(z|x)$（用于训练解码器）保持接近 $p(z)$，从而使解码器熟悉我们在生成时将采样的区域。

> VAE的ELBO目标函数中的KL散度项衡量编码器的输出分布 $q_{\phi}(z|x)$ 与预设先验 $p(z)$ 之间的差异，并起到正则化作用。

### 高斯分布的解析形式

当近似后验 $q_{\phi}(z|x)$ 和先验 $p(z)$ 都是高斯分布时，KL散度可以解析计算。这是VAE中的一种常见设置。
设：

- $q_{\phi}(z|x) = \mathcal{N}(z | \mu_{\phi}(x), \text{diag}(\sigma^2_{\phi}(x)))$，其中 $\mu_{\phi}(x)$ 是编码器针对输入 $x$ 输出的均值向量 (vector)，$\sigma^2_{\phi}(x)$ 是方差向量。
- $p(z) = \mathcal{N}(z | 0, I)$，一个零均值、单位协方差矩阵 $I$ 的标准多元高斯先验。

对于 $D_z$ 维潜在空间，这两个分布之间的KL散度为：


$$
D_{KL}(q_{\phi}(z|x) || p(z)) = \frac{1}{2} \sum_{j=1}^{D_z} \left( \mu_j(x)^2 + \sigma_j(x)^2 - \log(\sigma_j(x)^2) - 1 \right)
$$


这里， $\mu_j(x)$ 是均值向量 $\mu_{\phi}(x)$ 的第 $j$ 个分量，$\sigma_j(x)^2$ 是方差向量 $\sigma^2_{\phi}(x)$ 的第 $j$ 个分量。

让我们针对单个潜在维度 $j$ 分解求和项：

- $\mu_j(x)^2$: 该项惩罚近似后验的均值偏离先验均值（即0）。当 $\mu_j(x) = 0$ 时，此项最小化。
- $\sigma_j(x)^2$: 该项惩罚大方差。但它被下一项所平衡。
- $-\log(\sigma_j(x)^2)$: 该项惩罚极小方差（因为当 $\sigma^2 < 1$ 时，$\log(\sigma^2)$ 会变得非常负）。它促使编码器保持一定的不确定性。
- $-1$: 一个常数项。

综合来看，这些项促使 $q_{\phi}(z|x)$ 的每个维度都具有接近0的均值和接近1的方差，从而使其与标准高斯先验 $p(z)$ 对齐 (alignment)。

### 平衡之举：重构与正则化 (regularization)

VAE训练过程涉及一个精妙的平衡。ELBO试图最大化重构精度（第一项），同时最小化KL散度（第二项，它被减去）。

- 如果KL散度项权重 (weight)过大（或者模型发现很容易最小化它），则 $q_{\phi}(z|x)$ 对于所有 $x$ 可能会变得过于接近 $p(z)$。在这种情况下，$q_{\phi}(z|x)$ 可能会失去关于特定输入 $x$ 的信息，导致所谓的“后验坍塌”。潜在编码变得不包含信息，解码器本质上学习生成平均输出，导致重构效果差或样本过于模糊。
- 如果KL散度项太弱或被忽略，编码器可能会学习针对每个 $x$ 的非常特定、高置信度的编码（即，每个 $x$ 的 $q_{\phi}(z|x)$ 可能非常不同且远离 $p(z)$）。这可能会导致训练数据重构良好，但潜在空间可能缺乏良好泛化和通过从 $p(z)$ 采样生成新样本所需的平滑、规整结构。解码器在训练期间未被任何 $q_{\phi}(z|x)$ “覆盖”的 $z$ 区域表现可能不佳。

这种权衡是VAE的一个核心特点。KL散度确保潜在空间保持一定程度的“受控”并遵循先验，这有益于生成和创建连续有意义的潜在空间。然而，它也限制了潜在编码存储关于 $x$ 的信息的能力，可能以牺牲重构质量为代价。找到适当的平衡，有时通过调整KL项的权重因子（如在第3章中介绍的 $\beta$-VAE模型所示），对于VAE的成功训练和应用非常重要。

总结来说，KL散度项 $D_{KL}(q_{\phi}(z|x) || p(z))$ 不仅仅是一个数学产物；它是塑造VAE潜在空间的枢纽组件。它对编码器施加结构约束，将学习到的潜在编码分布推向选定的先验，通常是一个简单的高斯分布。这种正则化对于使VAE能够生成新的、连贯的样本以及学习平滑、表现良好的潜在表示不可或缺。理解它的作用以及与重构项的关联，是了解VAE如何学习和工作的基本要点。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍变分自编码器的原始论文，详细阐述了ELBO目标函数和KL散度的作用。
- [Deep Learning](http://www.deeplearningbook.org/contents/generative-models.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 20: Deep Generative Models
  权威教科书，其中专门一章深入解释了变分自编码器、ELBO和KL正则化。
- [Elements of Information Theory](https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959) — Thomas M. Cover, Joy A. Thomas (2006)
  Publisher: Wiley-Interscience; Pages: Chapter 2; DOI: [10.1002/0471742716](https://doi.org/10.1002/0471742716)
  提供信息理论基础的经典教科书，涵盖了Kullback-Leibler散度的定义和属性。
- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://arxiv.org/abs/1804.03599) — Irina Higgins, Loïc Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Alexander Pritzel, Matthew M. Botvinick, Daan Wierstra, Karen Ullrich, David Rezende (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1804.03599](https://doi.org/10.48550/arXiv.1804.03599)
  引入了VAE变体的研究论文，明确控制KL散度项的强度，处理重建与正则化之间的平衡问题。
