---
course: "vae-representation-learning"
chapter: "vae-inference-techniques"
lesson: "variational-inference-implicit-models"
sourceId: 6346
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-4-vae-inference-techniques/variational-inference-implicit-models"
title: "使用隐式模型的变分推断"
description: "考察当后验或先验是隐式模型时的变分推断。"
order: 6
plots: []
sourceHash: "6a7768f1d0395119a5a8fa2ea5122b745ad8e55c7b07cf69212a651d49ec1a42"
sourceCorrections: []
---

近似后验 $q_\phi(z|x)$ 的表达能力是变分自编码器 (VAE) 性能的主要因素。诸如对角高斯分布等常见选择，通常难以充分捕捉 $p_\theta(z|x)$ 的真实复杂性。为变分后验甚至先验使用*隐式模型*，提供了一种强有力的方法。

### 什么是隐式模型？

隐式模型是指我们可以轻松地从分布中*采样*，但难以（或根本无法）计算给定点的概率密度函数（PDF）或概率质量函数（PMF）的分布。以生成式神经网络 (neural network)为例：你输入随机噪声，它会产生复杂的输出。你可以获得样本，但对于特定 $z$，其 $\log q(z)$ 是多少呢？这通常是难以计算的。

形式上，如果我们想要一个隐式后验 $q_\phi(z|x)$，我们通过简单噪声变量 $\epsilon$（例如，$\epsilon \sim \mathcal{N}(0, I)$）和输入 $x$ 的确定性变换 $g_\phi$ 来定义它：


$$
z = g_\phi(\epsilon, x)
$$


尽管我们可以通过先采样 $\epsilon$ 然后应用 $g_\phi$ 来从 $q_\phi(z|x)$ 生成样本 $z$，但 $q_\phi(z|x)$ 的密度本身是无法直接获得的。这与高斯后验（例如 $q_\phi(z|x) = \mathcal{N}(z | \mu_\phi(x), \Sigma_\phi(x))$）形成鲜明对比，在高斯后验中，我们可以直接计算 $\log q_\phi(z|x)$。

### 挑战：ELBO与隐式密度

VAEs的证据下界（ELBO）是：


$$
L(\theta, \phi; x) = E_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - KL(q_\phi(z|x) || p(z))
$$


我们来展开KL散度项：


$$
KL(q_\phi(z|x) || p(z)) = E_{z \sim q_\phi(z|x)}[\log q_\phi(z|x) - \log p(z)]
$$


如果 $q_\phi(z|x)$ 是隐式的，则项 $E_{z \sim q_\phi(z|x)}[\log q_\phi(z|x)]$（即 $q_\phi$ 的负熵）变得难以处理，因为 $\log q_\phi(z|x)$ 未知。这是将隐式模型用于变分后验时的主要难题。

如果先验 $p(z)$ 也被选为隐式模型（也许是为了表示潜在空间中复杂的标的结构），那么 $E_{z \sim q_\phi(z|x)}[\log p(z)]$ 项也无法通过直接评估得到。然而，如果 $p(z)$ 简单（例如，$\mathcal{N}(0, I)$），这一项仍然可以通过采样 $z \sim q_\phi(z|x)$ 并评估 $\log p(z)$ 来估计。

### 处理难以计算的对数密度

那么，如果无法计算 $\log q_\phi(z|x)$，我们如何优化ELBO呢？已经出现了几种策略，其中对抗训练尤为突出。

1. **密度比估计 / 对抗方法：**
   KL散度或其部分，通常可以使用类似于生成对抗网络 (GAN)（GANs）的技术进行重写或近似。核心思想是训练一个判别器（或评论器）网络 $D_\psi(z)$（如果以 $x$ 为条件，则为 $D_\psi(z, x)$），以区分来自 $q_\phi(z|x)$ 的样本和来自 $p(z)$ 的样本。
   然后， $q_\phi$ 的目标（具体来说，是 $g_\phi$ 的参数 (parameter)）是生成能够“欺骗”这个判别器的样本 $z$。

   例如，项 $KL(q_\phi(z|x) || p(z))$ 可以间接估计或优化。如果 $p(z)$ 是一个简单分布，主要问题是熵项 $H(q_\phi(z|x)) = -E_{z \sim q_\phi(z|x)}[\log q_\phi(z|x)]$。
   各种f-散度，包括KL散度，可以使用涉及判别器的变分表示来表达。例如，詹森-香农散度 $JS(q_\phi || p)$，由原始GAN目标函数最小化，是这样一种情况。

   对抗变分贝叶斯（AVB）框架，我们将在下一节讨论，它提供了一种具体机制。它通常涉及训练一个独立的网络 $T(z)$ 以近似 $\log q_\phi(z|x)$ 或直接估计KL项。重点是 $q_\phi(z|x)$ 通过 $z = g_\phi(\epsilon,x)$ 隐式定义，其参数根据来自重构项和近似KL散度的对抗项的梯度流进行更新。
2. **核密度估计（KDE）：**
   原则上，可以从 $q_\phi(z|x)$ 中抽取大量样本，并使用KDE来估计任何点 $z$ 的密度 $q_\phi(z|x)$。然后，$\log q_\phi(z|x)$ 可以被近似。然而，KDE严重受维度灾难影响，并需要非常大量的样本，这使其在VAEs典型的潜在空间维度中不切实际。
3. **无似然推断方法：**
   更广泛的无似然推断领域（也称为近似贝叶斯计算，ABC）处理似然（或在本例中，后验密度）难以计算但可以采样的情况。该领域的一些技术可以为带有隐式后验的VAEs的方法提供启发，通常涉及比较从不同模型生成的数据的概括统计量。

以下图表说明了如何将隐式后验集成到VAE中，并使用对抗方法处理KL散度项。

> 带有隐式后验 $q_\phi(z|x)$ 和用于KL散度的对抗机制的VAE中的信息流。采样器 $g_\phi$ 生成潜在编码 $z$，这些编码用于重构。这些样本连同来自先验 $p(z)$ 的样本被输入到判别器 $D_\psi$ 中，以计算对抗损失，从而辅助塑形 $q_\phi(z|x)$。

### 隐式先验 $p(z)$

同样的思路可以应用于先验分布 $p(z)$。替代固定、简单的先验（如 $\mathcal{N}(0, I)$），人们可能希望学习一个更复杂的先验，也许它本身就是一个隐式模型。对抗式自编码器（AAEs），我们将在第7章中提及，通常使用对抗损失来将*聚合后验* $q(z) = E_{p_{data}(x)}[q_\phi(z|x)]$ 匹配到一个选定的先验 $p(z)$，并且这个 $p(z)$ 可以通过来自另一个生成器的样本隐式定义。如果 $q_\phi(z|x)$ 和 $p(z)$ 都是隐式的，那么 $KL(q_\phi(z|x) || p(z))$ 项通常完全依赖于对抗或密度比估计技术。

### 优点与缺点

**优点：**

- **表达能力强的后验/先验：** 隐式模型能够表示任意复杂的分布，潜在地捕捉真实后验 $p_\theta(z|x)$ 或期望先验 $p(z)$ 的多模态 (multimodal)或非高斯特性。这可以导致更紧密的ELBO（如果可以估计）和更好的生成性能。
- **改进的样本质量：** 更准确的后验可以带来更高质量的重构和生成样本，因为潜在空间可能结构更好。
- **灵活性：** 避免了对 $q_\phi(z|x)$ 形式的限制性参数 (parameter)假设。

**缺点：**

- **训练不稳定：** 对抗训练组件出了名的难以稳定和调整，需要仔细平衡生成器和判别器更新、损失函数 (loss function)选择和网络架构。
- **评估难题：** 评估真正的ELBO变得困难。尽管我们可以优化一个替代目标，但了解界限的实际紧密程度更难。模型比较通常依赖于样本质量、使用如重要性采样（之前已涵盖）的方法进行留存似然估计，或下游任务性能。
- **复杂性增加：** 由于额外的网络（例如，判别器或评论器）和更复杂的训练循环，整体模型复杂性增加。

### 展望

使用隐式模型的变分推断为增强VAEs提供了丰富途径。通过采样过程 $z = g_\phi(\epsilon, x)$ 来定义 $q_\phi(z|x)$（以及潜在的 $p(z)$），我们可以建模远比标准高斯分布更复杂的分布。主要障碍，即 $\log q_\phi(z|x)$ 的不可计算性，通常使用对抗训练方案来解决。关于对抗变分贝叶斯（AVB）的下一节将提供一个更具体的例子，说明这些理念如何付诸实践以创建更强大的VAEs。这些方法代表着在弥合基本VAEs的可计算但有限的后验与像GANs这类模型的高度灵活但通常无结构的潜在空间之间差距方面迈出的重要一步。

## 参考资料

- [Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks](https://proceedings.mlr.press/v70/mescheder17a.html) — Lars Mescheder, Sebastian Nowozin, Andreas Geiger (2017)
  Journal: International Conference on Machine Learning; Volume: 70; Pages: 2467-2476; DOI: [10.5591/978-1-57758-703-9-33](https://doi.org/10.5591/978-1-57758-703-9-33)
  提出了对抗变分贝叶斯，一种利用对抗训练来处理VAE中隐式变分后验的方法。
- [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644) — Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Frey (2015)
  Journal: ICLR; DOI: [10.48550/arXiv.1511.05644](https://doi.org/10.48550/arXiv.1511.05644)
  介绍了对抗自编码器，它使用对抗网络将聚合后验分布与先验分布进行匹配。
- [InfoVAE: Information Maximizing Variational Autoencoders](https://arxiv.org/abs/1706.02262) — Shengjia Zhao, Jiaming Song, Stefano Ermon (2017)
  Journal: arXiv; Volume: 32
  提出了InfoVAE，它使用最大均值差异（MMD）来解决VAE中KL散度的局限性，这与隐式模型带来的挑战有关。
