---
course: "intro-diffusion-models"
chapter: "generative-modeling-fundamentals"
lesson: "motivation-diffusion-models"
sourceId: 5362
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-1-generative-modeling-fundamentals/motivation-diffusion-models"
title: "扩散模型的缘由"
description: "讨论扩散模型旨在应对的其他生成式模型的问题。"
order: 2
plots: []
sourceHash: "acef505a4d821e72c2c7ffd62ae69875d49364a51e789807f5201429224b1ef6"
sourceCorrections: []
---

变分自编码器（VAE）和生成对抗网络（GAN）等模型已显著推动生成式模型的发展，但它们各自面临研究人员和实践者常遇到的一系列挑战。分析这些局限性有助于阐明扩散模型为何获得许多关注。

### 变分自编码器（VAE）面临的问题

VAE优化数据似然的下界，即证据下界（ELBO）。此目标函数平衡了输入数据的重构与确保潜在空间分布符合预设先验（通常是高斯分布）的要求。尽管在数学上很优美，但这种方法在实践中常导致一些问题：

1. **样本质量：** VAE生成的样本与原始数据分布相比，常显得有些模糊或过于平滑。这部分原因在于重构损失项（常为均方误差）和ELBO本身带来的约束，它们并非总是与感知质量完美契合。
2. **优化上的复杂性：** 在ELBO中的重构项和Kullback-Leibler (KL) 散度项之间找到恰当平衡可能很敏感，需要仔细调整超参数 (parameter) (hyperparameter)。

### 生成对抗网络（GAN）面临的问题

GAN采用一种独特的对抗训练过程，包含一个生成器和一个判别器网络相互对抗。这种动态学习过程能生成令人印象深刻的清晰、真实样本，但管理起来非常困难：

1. **训练不稳定性：** 主要问题在于平衡生成器和判别器。如果一个网络明显压倒另一方，训练可能发散或震荡，无法收敛到有用的平衡点。这常需要仔细的架构设计、归一化 (normalization)技术和超参数 (parameter) (hyperparameter)调整。
2. **模式崩溃：** 一种常见的失败模式，其中生成器学会只生成可能数据变化的一个有限子集，实际上是“崩溃”到数据分布的少数几种模式上。它成功地用这几个例子欺骗了判别器，但未能捕捉到训练数据的全部多样性。
3. **评估上的困难：** 评估GAN的性能并非易事。不同于具有明确似然目标的模型，没有一个单一的、普遍接受的指标能准确衡量样本质量和多样性。Fréchet Inception Distance (FID) 等指标虽常用，但提供的是间接衡量。

### 扩散模型为何受青睐？

扩散模型提供了一种替代方法，解决了这些困难中的一部分，并具备若干吸引人的特点：

1. **高样本质量：** 先进的扩散模型以生成高保真样本而闻名，这些样本的质量常与GAN生成的结果不相上下或更优，尤其在图像生成方面，且通常不会出现VAE中的模糊问题。
2. **训练稳定性：** 扩散模型的训练过程，通常涉及预测添加到数据的噪声，倾向于更稳定，且不易出现GAN中出现的对抗性动态问题。目标函数（常为噪声的简单均方误差）通常易于优化。
3. **可处理的似然（理论上）：** 尽管在实践中出于计算原因常被简化，但扩散模型的基础数学框架允许计算数据似然，类似于VAE。这为更严格的模型评估提供了可能的途径，尽管最大化似然并不总是与最佳感知质量完美相关。
4. **条件设置的灵活性：** 扩散模型生成过程的迭代特性非常适合加入条件信息（如类别标签或文本描述）以引导样本生成，这将在后续章节中进一步讨论。

这些优势使得扩散模型成为生成式模型工具箱中一个强大且日益受欢迎的技术。它们通过一种根本不同的机制实现这些优势：一个逐步添加噪声并学习如何逆转它的过程，这也构成了后续章节的核心内容。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P. Kingma, Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍变分自编码器及其基于证据下界（ELBO）的理论框架的原始论文。
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Pages: 2672-2680; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  提出生成对抗网络（GANs）的原始论文，建立了对抗训练框架。
