---
course: "synthetic-data-gans-diffusion"
chapter: "generative-modeling-foundations-revisited"
lesson: "probabilistic-modeling-generation"
sourceId: 5628
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-1-generative-modeling-foundations-revisited/probabilistic-modeling-generation"
title: "用于生成的概率建模"
description: "回顾生成模型中概率论的基本理论。"
order: 1
plots: []
sourceHash: "38b2b9cfabd85793664cf813ded82a0d26aded89cc16566876f6219c6570c94b"
sourceCorrections: []
---

生成建模的核心目标在于学习给定数据集的内在结构和概率分布。假设你有一组图像，例如手写数字。生成模型的目标是理解这些数字是*如何*形成的，而不仅仅是分类它们。更正式地说，如果我们将数据点（图像、音频信号、文本序列）表示为 $x$，那么目标就是学习或近似真实数据分布，通常记作 $p_{data}(x)$。这个函数告诉我们观察到任意特定数据点 $x$ 的概率（对于连续数据而言是概率密度）。

学习 $p_{data}(x)$ 有什么用处？

1. **生成（采样）：** 如果我们有一个 $p_{data}(x)$ 的良好模型，就可以从中提取新的样本。这意味着可以生成与模型训练数据相似的新图像、音乐或文本。这是合成数据生成的核心任务。
2. **密度估计：** 模型可以为任意给定点 $x$ 分配一个概率（或似然），表示其根据所学分布的典型性或异常性。这对异常检测等任务很有价值。
3. **表示学习：** 在学习 $p_{data}(x)$ 的过程中，模型通常会学习到数据有意义的、压缩的表示（特征），这对于后续任务会有帮助。

然而，$p_{data}(x)$ 几乎总是未知且极其复杂的，特别是对于像自然图像这样的高维数据。一张 256x256 像素的彩色图像存在于一个包含 $256 \times 256 \times 3 = 196,608$ 维度的空间中。直接在这种高维空间 (high-dimensional space)中建模概率分布在计算上具有挑战性，并且需要大量数据。

因此，我们不直接精确地找出 $p_{data}(x)$，而是使用一个*模型*分布 $p_{model}(\theta; x)$，它由一组可学习参数 (parameter) $\theta$ 定义。这些参数通常是深度神经网络 (neural network)的权重 (weight)和偏差。训练生成模型的核心任务是调整 $\theta$，使得 $p_{model}(\theta; x)$ 尽可能接近真实的（但未知的）$p_{data}(x)$。

> 此图展现了真实数据分布（$p_{data}$）、观测数据样本、生成模型的分布（$p_{model}$）、其参数（$\theta$）以及生成样本之间的关系。训练过程旨在调整 $\theta$，使 $p_{model}$ 尽可能接近 $p_{data}$。

我们如何衡量 $p_{model}$ 与 $p_{data}$ 之间的“接近程度”并优化 $\theta$？不同类型的生成模型采用不同的策略：

- **显式密度模型：** 这些模型为 $p_{model}(\theta; x)$ 定义了显式数学公式，并常使用\*\*最大似然估计（MLE）\*\*进行训练。目标是找到使训练数据观测（对数）概率最大化的参数 $\theta$：
  $\theta^* = \arg \max_\theta \sum_{i=1}^N \log p_{model}(\theta; x^{(i)})$
  这里，$x^{(i)}$ 表示训练集中的数据点。虽然理论上具有吸引力，但对于许多灵活的模型（如深度神经网络）来说，由于复杂的依赖关系或归一化 (normalization)常数，计算或优化此似然可能难以处理。变分自编码器（VAEs）、基于流的模型和自回归 (autoregressive)模型等方法均属于此类，每种方法都采用不同方式使似然变得可计算或对其进行近似。扩散模型，正如我们将要看到的，也常与似然估计相关联，尽管它们的训练目标可能采用不同形式（例如，分数匹配或去噪目标）。
- **隐式密度模型：** 这些模型不定义显式的 $p_{model}(\theta; x)$。相反，它们提供了一种从其隐式表示的分布中*采样*的机制。生成对抗网络 (GAN)（GANs）是主要示例。GAN 的生成器网络 $G$ 学习从简单先验分布 $p_z(z)$（例如高斯噪声）到复杂数据分布的转换。它学习生成与真实数据 $x \sim p_{data}(x)$ 难以区分的样本 $G(z)$，由判别器 $D$ 指导。你之前看到的最小-最大目标函数推动了这一过程，隐式地调整 $G(z)$ 的分布以匹配 $p_{data}(x)$，而无需写下或计算生成样本的概率密度。

了解这一概率理论很重要。无论是显式地最大化似然，还是通过对抗性博弈隐式地匹配分布，核心目标保持不变：创建一个能够忠实反映原始数据集中特征和变化的模型。随着学习的展开，我们将看到 GANs 和扩散模型如何以独特而有力的方式运用这些概率原理，在合成数据生成中取得领先成果。

## 参考资料

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Volume: 27; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  介绍生成对抗网络（GANs）的开创性论文，这是本节讨论的隐式密度建模的基石。
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P. Kingma and Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR 2014); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  关于变分自编码器（VAEs）的原始论文，它是一个显式密度模型的例子，用于近似复杂数据分布的似然性。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems; Volume: 33; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  本文介绍了去噪扩散概率模型（DDPMs），这是一种在生成建模中影响深远的方法，通过去噪目标与似然估计相关联。
- [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) — Kevin P. Murphy (2022)
  Publisher: MIT Press; Pages: Chapter 18
  为概率建模和机器学习提供了全面的理论基础，其中有一章专门讨论生成模型，强化了数据分布和模型近似的核心概念。
