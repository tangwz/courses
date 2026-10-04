# VAE目标函数分析

来源：[原文](https://apxml.com/zh/courses/vae-representation-learning/chapter-2-vaes-mathematical-deep-dive/analysis-vae-objectives)

[返回章节目录](README.md) · [返回课程目录](../README.md)

变分自编码器（VAE）使用证据下界（ELBO）作为其目标函数。对ELBO的详细审视包括分解其组成部分，以及研究不同的观点和修正。理解这些方面对于诊断训练问题、调整VAE以及体会更先进VAE变体背后的设计选择很有帮助。

我们旨在最大化的标准VAE目标函数通常写为：


$$
L_{ELBO}(x, \theta, \phi) = \mathbb{E}_{z \sim q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - D_{KL}(q_{\phi}(z|x) || p(z))
$$


这里，$q_{\phi}(z|x)$ 是由 $\phi$ 参数 (parameter)化的编码器（或推断网络），$p_{\theta}(x|z)$ 是由 $\theta$ 参数化的解码器（或生成网络），而 $p(z)$ 是潜在变量的先验分布，通常是一个标准高斯分布 $\mathcal{N}(0, I)$。

让我们来分解一下。

### ELBO的两个主要部分：重建与正则化 (regularization)

ELBO包含两个主要项，它们将模型引导向不同、有时相互竞争的方向。

#### 1. 重建项：$\mathbb{E}_{z \sim q_{\phi}(z|x)}[\log p_{\theta}(x|z)]$

此项是对从编码器输出 $q_{\phi}(z|x)$ 中采样的潜在编码 $z$ 的期望。它衡量了解码器 $p_{\theta}(x|z)$ 在给定代表 $x$ 的潜在编码 $z$ 时，能够多好地重建原始输入 $x$。最大化此项会促使VAE学习有意义的潜在表示，以便能够忠实地重建输入数据。

$\log p_{\theta}(x|z)$ 的具体形式取决于你的数据 $x$ 的性质：

- 对于二元数据（例如，黑白图像），$p_{\theta}(x|z)$ 通常是伯努利分布。此时，重建损失变为二元交叉熵。
- 对于实值数据（例如，自然图像中归一化 (normalization)到 [0,1] 的像素强度），$p_{\theta}(x|z)$ 通常是高斯分布，$\mathcal{N}(x | \mu_{\theta}(z), \sigma^2I)$，其中 $\mu_{\theta}(z)$ 是解码器网络的输出。如果 $\sigma^2$ 固定，最小化负对数似然等同于最小化输入 $x$ 与其重建 $\mu_{\theta}(z)$ 之间的均方误差（MSE）。

此项的值越高，通常意味着重建质量越好。

#### 2. 正则化项：$D_{KL}(q_{\phi}(z|x) || p(z))$

此项是近似后验分布 $q_{\phi}(z|x)$（由编码器针对给定输入 $x$ 生成）与潜在变量的先验分布 $p(z)$ 之间的 Kullback-Leibler (KL) 散度。ELBO 会减去此项，因此最大化ELBO意味着最小化此KL散度。

此KL项的作用很主要：

- **它起到正则化作用。** 它迫使所有输入 $x$ 的分布 $q_{\phi}(z|x)$ 平均而言接近所选的先验 $p(z)$。通常，$p(z)$ 是一个简单的单峰分布，如标准正态分布 $\mathcal{N}(0, I)$。
- **它促进了潜在空间的结构化。** 通过将所有编码分布推向共同的先验，KL项有助于确保潜在空间不会变得过于碎片化或出现“空洞”。这种连续性对于VAE的生成能力很主要：如果我们从 $p(z)$ 中采样 $z$，我们希望它能解码出有意义的内容。
- **它防止了后验坍缩。** 如果没有此项（或其影响太弱），编码器可能会学习将每个 $x$ 映射到一个非常特定、类似 Dirac Delta 函数的 $q_{\phi}(z|x)$，该函数对于重建 $x$ 是最优的，但会使得 $q_{\phi}(z|x)$ 与 $p(z)$ 差异很大，并且对于相似的 $x_i$ 和 $x_j$ 也可能差异很大。KL项可以缓解这种情况。

### 固有的权衡

ELBO中的两项在VAE训练中产生了根本性的权衡：

- **高保真重建：** 如果我们优先考虑重建项，VAE可能会学习将输入编码到潜在空间中远离先验 $p(z)$ 的特定区域。这可能导致出色的重建，但潜在空间会变得“有空洞”或不平滑。在这种情况下，KL散度项会很大。
- **强正则化 (regularization)：** 如果我们严厉惩罚偏离先验（即，KL项占据主导），编码器会将所有输入 $x$ 映射到非常接近 $p(z)$ 的潜在分布 $q_{\phi}(z|x)$。这会产生一个平滑、规则的潜在空间，适用于生成。然而，如果 $q_{\phi}(z|x)$ 对所有 $x$ 都变得与 $p(z)$ 过分相似，它可能会丢失关于 $x$ 的特定信息，导致重建效果较差或“模糊”。这种极端情况被称为“后验坍缩”，其中潜在变量变得与输入无关。

下图说明了这种平衡：

> VAE目标函数平衡了准确数据重建和正则化潜在空间的需求。不平衡可能导致次优结果。

找到正确的平衡对于训练有效的VAE很主要。这通常涉及仔细的模型架构设计、超参数 (parameter) (hyperparameter)选择，有时还需要修改目标函数本身。

### ELBO的其他视角

ELBO可以用几种不同的方式书写和理解，提供额外的见解。回顾推导过程可知，数据的对数边际似然 $\log p(x)$ 可以分解为：


$$
\log p(x) = L_{ELBO} + D_{KL}(q_{\phi}(z|x) || p(z|x))
$$


由于KL散度总是非负的（$D_{KL}(\cdot || \cdot) \ge 0$），这个等式告诉我们 $L_{ELBO} \le \log p(x)$。这就是为什么它被称为证据*下界*：最大化ELBO等同于最大化数据对数边际似然的一个下界。

这也意味着最大化ELBO等同于最小化 $D_{KL}(q_{\phi}(z|x) || p(z|x))$，即我们的近似后验 $q_{\phi}(z|x)$ 与真实（但难以处理）后验 $p(z|x)$ 之间的KL散度。因此，一个训练有素的VAE不仅学习了一个良好的生成模型 $p_{\theta}(x|z)$，还学习了一个推断网络 $q_{\phi}(z|x)$，该网络近似于给定数据下的潜在变量的真实后验分布。

### $\beta$-VAE目标函数

对VAE目标函数最常见且影响很大的修改之一是 $\beta$-VAE 公式，其主要目的是为了促进学习解耦表示（我们将在第5章详细介绍）。$\beta$-VAE目标函数为：


$$
L_{\beta-VAE} = \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - \beta D_{KL}(q_{\phi}(z|x) || p(z))
$$


唯一的改变是引入了超参数 (parameter) (hyperparameter) $\beta$。

- **当 $\beta = 1$ 时**，我们得到标准的VAE ELBO。
- **当 $\beta > 1$ 时**，KL散度项的权重 (weight)增加。这会给编码器带来更大的压力，使其将 $q_{\phi}(z|x)$ 与先验 $p(z)$ 匹配。直观上，这迫使潜在维度更加独立（如果 $p(z)$ 是像 $\mathcal{N}(0,I)$ 这样的因子分布），并可能导致更解耦的表示，其中每个潜在维度对应数据中不同、可解释的变异因子。然而，增加 $\beta$ 通常会以牺牲重建质量为代价。模型可能会为了满足更严格的正则化 (regularization)而牺牲保真度。
- **当 $0 < \beta < 1$ 时**，KL项的权重减小，优先考虑重建。当重建质量很重要且标准VAE产生过于模糊的结果时，这会很有用。

$\beta$ 的选择为权衡讨论带来了另一个维度，允许你明确地控制重建与潜在通道 $q_{\phi}(z|x)$ 相对于先验的“信息量”或“复杂性”之间的平衡。

### 先验 $p(z)$ 的影响

先验 $p(z)$ 的标准选择是各向同性高斯分布 $\mathcal{N}(0, I)$。这种选择意味着：

1. 每个潜在维度都是独立的。
2. 每个潜在维度都以零为中心，具有单位方差。

KL散度项 $D_{KL}(q_{\phi}(z|x) || \mathcal{N}(0, I))$ 促使学习到的编码符合这些属性。这很简单，并且通常效果良好，尤其是在目标是解耦，希望潜在因子之间独立时。

然而，数据潜在流形的真实底层结构可能无法通过简单高斯分布很好地近似。可以使用更复杂的先验，例如：

- **高斯混合模型（GMM）：** 如果你认为数据在潜在空间中存在不同的簇。
- **可学习先验：** 使用像归一化 (normalization)流（在第3章中讨论）这样的技术来定义一种更灵活的先验分布，可以从数据中学习。

改变先验 $p(z)$ 会改变KL散度项的目标分布，这进而会影响学习到的潜在空间的结构以及VAE发现的表示类型。例如，使用GMM先验可能会促使 $q_{\phi}(z|x)$ 将输入映射到潜在空间中不同的簇。

### 目标分析与训练难题的关联

理解VAE目标函数有助于诊断常见的训练问题：

- **模糊重建：** 如果重建项 $\mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)]$ 的权重 (weight)不足，或者解码器 $p_{\theta}(x|z)$ 过于简单（例如，具有固定方差的高斯分布对于复杂数据可能限制过多），就可能发生这种情况。如果KL项过强，迫使 $q_{\phi}(z|x)$ 过于接近 $p(z)$，从而丢失过多关于 $x$ 的信息，也可能出现此问题。
- **后验坍缩：** 当 $q_{\phi}(z|x)$ 对于所有 $x$ 都变得与 $p(z)$ 非常相似时，就会发生这种情况，这意味着潜在编码 $z$ 几乎不包含关于输入 $x$ 的信息。解码器基本上学会忽略 $z$，只生成一个平均样本。这通常由KL散度项变得非常接近零来表示。如果解码器相对于编码器过于强大（例如，一个强大的自回归 (autoregressive)解码器），或者KL项的权重（如 $\beta$）过高，或者潜在编码的“信息容量”对于数据的复杂性来说太小，都可能发生这种情况。
- **潜在空间中的“空洞”：** 如果KL正则化 (regularization)太弱，编码器可能会将不同的输入映射到潜在空间中不相连的区域，而这些区域之间不对应任何有效数据。从这些“空洞”中采样会产生无意义的输出。KL项通过将所有 $q_{\phi}(z|x)$ 拉向连续的先验 $p(z)$ 来帮助“填补”这些空隙。

### 目标项的加权

在实践中，除了 $\beta$-VAE 之外，你可能会遇到重建损失（例如MSE）的数值范围与KL散度的范围不同的情况。例如，如果图像像素值在 [0, 255] 之间，MSE可能会非常大，而KL散度可能只有几十或几百。这种不平衡会使KL项在训练过程中几乎可以忽略不计。

一些实践者会为重建项引入一个明确的加权因子，或者对各项进行归一化 (normalization)，以确保两个组成部分在优化过程中都能对梯度做出有效贡献。这通常是一个经验性的选择，并且可能取决于特定的数据集和模型架构。例如，可以将损失写为：


$$
L = \lambda_{rec} \cdot (\text{Negative Reconstruction Term}) + \lambda_{KL} \cdot D_{KL}(q_{\phi}(z|x) || p(z))
$$


其中 $\lambda_{rec}$ 和 $\lambda_{KL}$ 是加权系数。$\beta$-VAE 是其中的一个特定例子，其中 $\lambda_{rec}=1$（通过最大化负损失隐含地实现）且 $\lambda_{KL}=\beta$。

对VAE目标函数的细致分析表明，有效训练这些模型需要精巧的平衡。重建保真度与潜在空间正则化 (regularization)之间的互相影响是其行为的核心。通过理解这些组成部分及其变体，你将更有能力设计、训练和排查各种VAE应用的故障，从生成到表示学习。接下来的章节将在这些目标函数的基础上，讨论更高级的架构和推断技术。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114v10) — Diederik P Kingma, Max Welling (2014)
  Journal: 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings; DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍变分自编码器（VAE）及其目标函数，即证据下界（ELBO），奠定了其基础理论框架。
- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://api.semanticscholar.org/CorpusID:46798026) — Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner (2017)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.1606.05579](https://doi.org/10.48550/arXiv.1606.05579)
  提出了 \(\beta\)-VAE，通过调整 KL 散度项的权重来鼓励学习解耦的潜在表示，是对 VAE 目标函数的一种改进。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  第20章全面概述了变分自编码器、其目标函数及其在深度学习背景下的理论基础。

---

[上一节](06-%E5%B8%B8%E8%A7%81%20VAE%20%E8%AE%AD%E7%BB%83%E9%97%AE%E9%A2%98.md) · [下一节](08-VAE%20%E5%AE%9E%E7%8E%B0%E4%B8%8E%E8%AF%8A%E6%96%AD%EF%BC%9A%E5%8A%A8%E6%89%8B%E6%93%8D%E4%BD%9C.md)
