---
course: "bayesian-machine-learning"
chapter: "bayesian-deep-learning"
lesson: "vaes-probabilistic-models"
sourceId: 3629
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-6-bayesian-deep-learning/vaes-probabilistic-models"
title: "变分自编码器 (VAEs) 作为概率模型"
description: "理解 VAEs 的贝叶斯解释及其与变分推断的关联。"
order: 7
plots: []
sourceHash: "52a59abd0baf054d2bc7fc74ea82cd15659aa4a2bb8d084859b4cebc0b2d991a"
sourceCorrections: []
---

变分自编码器 (VAEs) 是深度学习 (deep learning)与贝叶斯推断的结合点，提供了一种无监督生成建模的方法。与学习确定性映射到和从低维空间的标准自编码器不同，VAEs 采用概率视角，根植于潜在变量建模和变分推断原理。

可以将 VAE 看作由两个主要组成部分构成，通过概率潜在空间连接：

1. **编码器（识别模型）：** 将输入数据点 $x$ 映射到潜在空间中的一个分布。它不是输出潜在空间中的一个单点 $z$，而是输出表示 $x$ 的可能潜在表示的概率分布的参数 (parameter)（如均值 $\mu$ 和方差 $\sigma^2$）。这个分布，表示为 $q_{\phi}(z | x)$，是我们的*近似后验*。它由 $\phi$ 参数化，通常是神经网络 (neural network)的权重 (weight)。
2. **解码器（生成模型）：** 将从潜在分布中采样的点 $z$ 映射回原始数据空间。它定义了一个条件分布 $p_{\theta}(x | z)$，表示给定潜在表示 $z$ 时观测到 $x$ 的可能性。这由 $\theta$ 参数化，通常是另一个神经网络的权重。

其基本思想是，我们观测到的数据 $x$ 是由一些未观测到的、低维潜在变量 $z$ 生成的。我们对这些潜在变量假设一个先验分布 $p(z)$，通常选择一个简单分布，例如标准多变量高斯分布 $\mathcal{N}(0, I)$。解码器 $p_{\theta}(x | z)$ 定义了这些潜在变量如何生成我们所见的复杂数据。

主要困难在于计算真实的后验分布 $p_{\theta}(z | x) = p_{\theta}(x | z) p(z) / p(x)$。分母 $p(x) = \int p_{\theta}(x | z) p(z) dz$ 是边际似然或证据，通常难以计算，因为它需要对所有可能的潜在变量 $z$ 进行积分。

这正是变分推断发挥作用的地方。我们引入编码器网络 $q_{\phi}(z | x)$ 作为真实、难以处理的后验 $p_{\theta}(z | x)$ 的近似。正如我们在第三章中所学，变分推断将推断重构为一个优化问题。我们的目标是通过最小化 KL 散度 $D_{KL}(q_{\phi}(z | x) || p_{\theta}(z | x))$，使 $q_{\phi}(z | x)$ 尽可能接近 $p_{\theta}(z | x)$。最小化这个散度等同于最大化证据下界 (ELBO)，即 $\mathcal{L}(\theta, \phi; x)$：


$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x | z)] - D_{KL}(q_{\phi}(z | x) || p(z))
$$


最大化 ELBO $\mathcal{L}(\theta, \phi; x)$ 会同时训练编码器（参数 $\phi$）和解码器（参数 $\theta$）。让我们分析 ELBO 中的两个项：

- **重构项：** $\mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x | z)]$。这一项衡量了解码器在将原始输入 $x$ 编码为潜在分布 $q_{\phi}(z | x)$ 并从中采样一个 $z$ 后，能够多好地重构原始输入 $x$。对于高斯或伯努利似然 $p_{\theta}(x | z)$，这一项通常简化为输入 $x$ 与解码器输出 $\hat{x}$ 之间的均方误差或二元交叉熵损失。它促使 VAE 学习有意义的潜在表示，这些表示捕获了重构所需的重要信息。
- **正则化 (regularization)项（KL 散度）：** $D_{KL}(q_{\phi}(z | x) || p(z))$。这一项起到正则化器的作用。它衡量了编码器产生的近似后验分布 $q_{\phi}(z | x)$ 与潜在变量的先验分布 $p(z)$ 之间的散度。通过鼓励编码分布接近先验（例如，标准高斯分布），它确保了潜在空间结构良好，并防止编码器将不同输入折叠到不同、孤立的区域（后验坍塌）。这种规律性对于 VAE 的生成能力很重要。

### 重参数 (parameter)化技巧

在尝试使用梯度上升优化 ELBO 时出现了一个重要挑战：如何通过期望 $\mathbb{E}_{q_{\phi}(z|x)}[\cdot]$ 中涉及的采样步骤反向传播 (backpropagation)梯度？采样操作本身是随机且不可微分的。

VAE 通过*重参数化技巧*解决了这个问题。我们不是直接从 $q_{\phi}(z | x)$ 中采样 $z$，而是将 $z$ 表示为参数 $\phi$、输入 $x$ 以及一个具有固定分布（例如 $\epsilon \sim \mathcal{N}(0, I)$）的独立随机变量 $\epsilon$ 的确定性函数。

例如，如果我们的编码器 $q_{\phi}(z | x)$ 输出高斯分布的均值 $\mu_{\phi}(x)$ 和标准差 $\sigma_{\phi}(x)$，我们可以这样采样 $z$：


$$
z = \mu_{\phi}(x) + \sigma_{\phi}(x) \odot \epsilon, \quad \text{其中 } \epsilon \sim \mathcal{N}(0, I)
$$


在这里，$\odot$ 表示按元素乘法。现在，随机性来自 $\epsilon$，它不依赖于模型参数 $\phi$。从 $\phi$ 和 $\theta$ 到损失 $\mathcal{L}$ 的路径是确定性的，这使得梯度可以反向传播通过 $\mu_{\phi}(x)$ 和 $\sigma_{\phi}(x)$ 来更新编码器网络 $\phi$，并通过解码器来更新 $\theta$。

> VAE 架构的流程图。输入 `x` 经编码器网络处理，生成定义近似后验 `q_phi(z|x)` 的参数 `mu(x)` 和 `sigma(x)`。潜在样本 `z` 使用重参数化技巧和噪声 `eps` 生成。KL 散度项比较 `q_phi(z|x)` 和先验 `p(z)`。样本 `z` 经过解码器网络生成重构 `x_hat`。重构损失衡量 `x` 和 `x_hat` 之间的差异。两个损失项都对 ELBO 目标做出贡献。

### 贝叶斯深度学习 (deep learning)背景下的 VAEs

VAEs 是概率深度学习的核心。虽然它们通常不像 BNNs 那样直接对神经网络 (neural network)的*权重 (weight)*设置先验，但它们对*潜在变量* $z$ 进行贝叶斯推断。它们使用变分推断学习一个深度生成模型 $p_{\theta}(x|z)p(z)$，以实现高效训练。

与 BDL 的重要关联包括：

- **生成建模：** VAEs 通过潜在变量框架 $p(x) = \int p_{\theta}(x | z) p(z) dz$ 来学习数据的潜在分布 $p(x)$。一旦训练完成，你可以通过从 $p(z)$ 中抽取 $z$ 并将其通过解码器 $p_{\theta}(x | z)$ 来生成新的数据样本。
- **不确定性表示：** 潜在空间 $z$ 本身就捕获了数据中变异因素的不确定性。编码器为给定 $x$ 输出的方差 $\sigma^2_{\phi}(x)$ 反映了该特定输入在潜在表示中的不确定性。
- **推断技术：** VAEs 完全依赖于变分推断，这是我们在复杂贝叶斯模型（包括 BNNs）中进行近似推断时研究的主要技术之一。理解 VAEs 有助于巩固变分推断和 ELBO 的实际运用。

总之，VAEs 提供了一种可扩展且有效的方法，用于学习具有坚实概率根基的深度生成模型，该方法基于变分推断。它们连接了复杂的深度学习架构与对数据中潜在结构的严谨贝叶斯推理 (inference)。

## 参考资料

- [Variational Inference: A Review for Statisticians](http://dx.doi.org/10.1080/01621459.2017.1285773) — David M. Blei, Alp Kucukelbir, Jon D. McAuliffe (2017)
  Journal: Journal of the American Statistical Association; Publisher: Taylor & Francis; Volume: 112; Pages: 859-877; DOI: [10.1080/01621459.2017.1285773](https://doi.org/10.1080/01621459.2017.1285773)
  全面回顾了变分推断方法，提供了详细的理论背景和在各种统计模型中的应用，对理解VAE的推断机制至关重要。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 20
  一本关于深度学习的权威教科书，其中专门有一章介绍深度生成模型，清晰地解释了变分自编码器及其与更广泛的生成模型原则的联系。
- [CS236: Deep Generative Models, Lecture 4: Variational Autoencoders](http://cs236.stanford.edu/) — Stefano Ermon and CS236 Course Staff (2023)
  Publisher: Stanford University
  来自一门深度生成模型大学课程的讲义，对变分自编码器、其理论基础和实践方面进行了易于理解而又不失严谨的讲解。
