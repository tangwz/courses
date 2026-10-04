---
course: "autoencoders-representation-learning"
chapter: "variational-autoencoders-generative"
lesson: "latent-variable-model-perspective"
sourceId: 2850
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-4-variational-autoencoders-generative/latent-variable-model-perspective"
title: "隐变量模型视角"
description: "将 VAE 置于概率性隐变量模型的背景下。"
order: 3
plots: []
sourceHash: "837db183622ca1e2116178d7e17ed4d00785acaab8e48b314276e731a2e24b08"
sourceCorrections: []
---

要全面掌握变分自编码器 (VAE)，通过概率性\*\*隐变量模型 (LVM)\*\*的视角来审视它们是有效的方法。这种方法将重心从标准自编码器主要执行的简单数据压缩，转移到对数据自身内在概率分布的显式建模。

### 什么是隐变量模型？

其核心在于，LVMs 假设我们观察到的高维数据（我们称之为 $x$）是由一些潜在的、未观测到的（隐）变量 $z$ 生成的。这些隐变量通常存在于一个较低维度的空间中，并捕获数据中变动的主要因素。

基本思路是定义观测变量和隐变量上的联合概率分布 $p(x, z)$。我们可以以生成式的方式分解这个联合分布：


$$
p(x, z) = p(x | z) p(z)
$$


这里：

- $p(z)$ 是隐变量的**先验分布**。它定义了我们在观测任何数据之前，对隐空间结构最初的设想。在 VAEs 中，这通常被选择为一个简单的分布，例如标准多变量高斯分布 $\mathcal{N}(0, I)$。
- $p(x | z)$ 是**似然**（或生成分布）。它描述了在给定特定隐编码 $z$ 的情况下，如何生成一个观测数据点 $x$。这一部分通常由神经网络 (neural network)建模（在 VAE 环境中为解码器）。

生成式建模的最终目的是对观测数据 $p(x)$ 的分布进行建模。根据概率规则，我们可以通过对隐变量进行边缘化来获得这一点：


$$
p(x) = \int p(x, z) dz = \int p(x | z) p(z) dz
$$


如果我们能够完美地建模 $p(x|z)$ 并轻松计算这个积分，我们将拥有一个强大的生成模型，能够评估新数据点的似然性，并通过首先从 $p(z)$ 中采样 $z$，然后从 $p(x|z)$ 中采样 $x$ 来生成样本。

### 推断难题

尽管上述的生成过程 ($z \rightarrow x$) 相对直接，但与 LVMs 打交道时会遇到一个重要问题：**推断**。给定一个观测数据点 $x$，我们如何确定最有可能生成它的对应隐表示 $z$？这需要计算**后验分布** $p(z|x)$。使用贝叶斯定理：


$$
p(z|x) = \frac{p(x|z) p(z)}{p(x)} = \frac{p(x|z) p(z)}{\int p(x | z') p(z') dz'}
$$


问题在于：计算分母 $p(x)$（边缘似然或证据）需要对所有可能的隐变量 $z'$ 进行积分。对于像那些涉及到深度神经网络 (neural network)建模 $p(x|z)$ 的复杂模型以及连续隐变量 $z$，这个积分通常是**难以计算的**。因此，计算真实的后验 $p(z|x)$ 也是难以计算的。

### 变分推断与 VAE

这是 VAEs 中“变分”一词变得重要的原因。由于我们无法计算真实的后验 $p(z|x)$，我们转而使用一个更简单、可计算的分布 $q(z|x)$ 来对其进行近似。这个近似后验 $q(z|x)$ 也由一个神经网络 (neural network)建模 – 即 VAE 中的**编码器**网络。编码器接收数据点 $x$ 作为输入，并输出分布 $q(z|x)$ 的参数 (parameter)（例如，如果 $q$ 被选择为高斯分布，则输出均值和方差）。我们用 $\phi$ 表示编码器网络的参数，因此我们写为 $q_\phi(z|x)$。类似地，我们用 $\theta$ 表示解码器网络的参数（建模 $p(x|z)$），写为 $p_\theta(x|z)$。

现在的目标有两个方面：

1. 训练解码器网络 $p_\theta(x|z)$，使其能够从隐编码 $z$ 中准确地重建数据 $x$。
2. 训练编码器网络 $q_\phi(z|x)$，使其输出分布能够很好地近似真实的（但难以计算的）后验 $p_\theta(z|x)$。

我们需要一个目标函数，使我们能够共同优化参数 $\phi$ 和 $\theta$。变分推断通过最大化一个称为\*\*证据下界 (ELBO)\*\*的量来提供这个目标。正如我们将在“推导证据下界 (ELBO)”一节中详细阐述的，最大化 ELBO 等同于最小化近似后验 $q_\phi(z|x)$ 与真实后验 $p_\theta(z|x)$ 之间的 KL 散度，同时最大化数据的似然性。

简而言之，VAE 框架巧妙地结合了：

- 一个**生成模型**（先验 $p(z)$ 和解码器 $p_\theta(x|z)$），它定义了数据是如何假设性地生成的。
- 一个**推断模型**（编码器 $q_\phi(z|x)$），它学习在给定观测数据的情况下近似隐变量的后验分布。

通过将 VAE 构造成一个使用变分推断训练的 LVM，我们理解了它*为什么*能学习到适合生成任务的结构化隐空间。目标函数明确地促使编码器生成平均上接近先验 $p(z)$ 的分布 $q_\phi(z|x)$（通过 ELBO 中的 KL 散度项），同时确保这些分布中的样本能够被解码回真实的数据点（通过 ELBO 中的重建项）。这种概率方法是 VAE 生成能力得以实现的根本所在，使其与标准自编码器有明显不同。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR 2014); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍了变分自编码器（VAE）框架的开创性论文。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 738
  一本关于概率机器学习的奠基性著作，涵盖了隐变量模型和变分推断。
- [Variational Inference: A Review for Statisticians](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1285773) — David M. Blei, Alp Kucukelbir, Jon D. McAuliffe (2017)
  Journal: Journal of the American Statistical Association; Publisher: Taylor & Francis; Volume: 112; Pages: 859-877; DOI: [10.1080/01621459.2017.1285773](https://doi.org/10.1080/01621459.2017.1285773)
  全面概述了变分推断方法及其应用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  深度学习的标准参考书，包含生成模型（包括VAE）的相关章节。
