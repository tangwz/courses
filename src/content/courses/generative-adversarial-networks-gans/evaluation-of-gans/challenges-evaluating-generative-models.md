---
course: "generative-adversarial-networks-gans"
chapter: "evaluation-of-gans"
lesson: "challenges-evaluating-generative-models"
sourceId: 2729
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-5-evaluation-of-gans/challenges-evaluating-generative-models"
title: "评估生成模型的挑战"
description: "为何评估生成样本的质量与多样性存在难题。"
order: 1
plots: []
sourceHash: "4bf9223264d33778480f90e6322bd4d3605271a15ce0bd6afd6c51bdfb0bad9d"
sourceCorrections: []
---

评估生成对抗网络 (GAN)的成效与评估监督学习 (supervised learning)模型显著不同。在分类或回归中，我们通常有明确的衡量标准，如准确率、精确率、召回率或均方误差，它们根据真实标签或数值计算得出。然而，GANs 学习生成*类似*真实数据分布 $P_{数据}$ 的数据。对于给定的输入潜在向量 (vector) $z$，并没有单一的“正确”输出。我们关心的是生成样本的分布 $P_{生成}$ 整体上与 $P_{数据}$ 匹配得如何。这一分布目标带来了一些基本的评估难题。

### 训练损失的模糊性

GAN 训练过程中最直接可用的衡量标准是生成器和判别器的损失。虽然这些损失值对于通过梯度下降 (gradient descent)驱动学习过程非常重要，但它们往往是最终样本质量或多样性的不良指示器。请记住 GANs 的对抗性质：它是一个最小最大博弈。判别器损失的目标是随着其区分真实与伪造的能力提高而下降，而生成器损失的目标是随着其欺骗判别器的能力提高而下降。

在某些 GAN 变体的稳定训练情况中（例如我们在第3章讨论的 WGANs），损失可能在一定程度上与质量相关。然而，对于原始 GAN 公式和许多其他变体，损失值振荡或不收敛是很常见的。较低的生成器损失可能表明它成功欺骗了*当前*的判别器，但不能保证生成的样本全局真实或多样。类似地，较低的判别器损失可能意味着它轻松区分真实与伪造，这可能是因为生成器已发生模式坍塌，并产生低质量或重复的样本。仅依赖损失曲线提供了不完整且常具误导性的 GAN 实际生成能力画面。

### 核心问题：比较分布

最终目标是评估两个概率分布之间的相似度：真实数据分布 $P_{数据}$ 和生成数据分布 $P_{生成}$。这本质上是困难的，特别是当这些分布定义在高维空间 (high-dimensional space)中时，如图像、音频或文本。

考虑图像生成。一个典型的图像数据集存在于一个可能具有数百万维（像素 $\times$ 颜色通道）的空间中。直接估计和比较此类高维空间中的概率密度，在计算上难以处理且统计上不可靠，常受“维度灾难”的影响。我们通常只拥有这些分布的样本，而不是它们的显式函数形式。因此，评估方法通常必须依赖于比较从 $P_{数据}$ 和 $P_{生成}$ 中抽取的*样本集*。

> GAN评估的抽象视图。来自真实和生成分布的样本常通过特征提取器（如预训练 (pre-training)网络）处理。评估指标随后比较这些特征的统计数据，以估计原始分布之间的相似度。主要难题在于确保这些指标与人类对质量和多样性的判断一致。

### 测量保真度（质量）

生成的样本有多“真实”？这一特性常被称为保真度或质量，其自动量化 (quantization)难度众所周知。简单的像素级指标，如生成样本与*某些*真实样本之间的均方误差（MSE）或峰值信噪比（PSNR），通常是感知质量的差指示器。一幅图像相对于目标可能具有较低的像素级误差，但看起来模糊或包含伪影；而另一幅图像可能由于轻微的平移或旋转而具有较高的像素误差，但在人类观察者看来却完全真实。人类感知对结构信息、纹理和语义内容敏感，而像素级统计数据无法充分捕捉这些内容。

### 测量多样性（覆盖度）

除了单个样本质量，生成器必须产生多样化输出，覆盖真实数据分布的广度。它不应出现*模式坍塌*，即仅生成少数几种不同类型的样本，忽略数据分布的大部分。评估多样性涉及评估生成样本集中看到的多样性是否与真实数据集中的多样性相符。这也具有挑战性：如何量化 (quantization)样本在高维空间 (high-dimensional space)中的“分布范围”，并将其与真实数据的分布范围进行比较？仅关注质量的指标可能会给一个生成器打高分，即使它从多类别数据集中仅生成一个特定对象类别的完美图像。

### 权衡与缺乏单一完美指标

通常，保真度与多样性之间存在权衡。某些技术可能以降低多样性为代价提高表观样本质量。因此，评估需要考虑这两个方面。不幸的是，没有单一指标能完美捕捉保真度和多样性，同时完美契合人类在所有数据集和模型类型中的感知。不同的指标（如初始分数 Inception Score、FID、精确率/召回率，本章稍后将讨论）捕捉这种比较的不同方面，各自具有其优缺点、偏置 (bias)和计算成本。

理解这些固有的困难是有效使用和解释我们接下来将讨论的定量和定性评估技术的第一步。我们需要超越简单的损失值的方法，并尝试以一种至少部分地与真实且多样化生成预期结果相关的方式来测量分布相似度。

## 参考资料

- [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems (NIPS 2014); Pages: 2672-2680; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  本文介绍了生成对抗网络的基本框架，包括最小最大目标函数。理解其训练动态对于识别其损失行为带来的评估挑战至关重要。
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems (NIPS 2017); Volume: 30; Pages: 6626-6637; DOI: [10.48550/arXiv.1706.08500](https://doi.org/10.48550/arXiv.1706.08500)
  提出了Fréchet Inception Distance (FID)，这是一个被广泛接受且稳健的指标，通过比较特征表示的统计数据来评估生成图像的质量和多样性。
- [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991) — Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, Timo Aila (2019)
  Journal: NeurIPS 2019; Pages: 12030-12040; DOI: [10.48550/arXiv.1904.06991](https://doi.org/10.48550/arXiv.1904.06991)
  提出了一种改进的Precision和Recall指标，对生成模型的保真度（Precision）和多样性（Recall）进行单独评估，有助于明确这两个方面之间的权衡。
