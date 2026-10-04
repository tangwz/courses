---
course: "cnns-for-computer-vision"
chapter: "gans-image-synthesis"
lesson: "gan-evaluation-metrics"
sourceId: 2625
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-7-gans-image-synthesis/gan-evaluation-metrics"
title: "GAN 的评估指标"
description: "了解用于评估 GAN 生成图像质量和多样性的量化指标，例如 FID 和 IS。"
order: 6
plots: ["plots/2625-0.json"]
sourceHash: "e8abf653cfde4aa50a0ba86d065ba2adae294c118956832754df80d68d42cb64"
sourceCorrections: []
---

评估生成对抗网络（GAN）的输出不像监督学习 (supervised learning)中计算准确度或损失那样直接。由于生成器的目标是产生逼真*且*多样的样本，以模拟目标分布，我们需要能评估单个生成图像的质量（保真度）以及整个生成集合的多样性（变化程度）的指标。仅仅查看样本是主观的，且难以大规模衡量，而训练期间生成器和判别器的损失通常与最终输出的感知质量没有强关联。因此，需要专门的量化 (quantization)指标来对不同 GAN 模型或训练检查点进行客观比较。

主要问题在于比较概率分布：真实数据分布 $p_{data}$ 和由生成器隐式定义的分布 $p_g$。我们想衡量 $p_g$ 与 $p_{data}$ 的“接近”程度。此领域中有两个重要指标已成为标准：Inception 分数（IS）和 Fréchet Inception 距离（FID）。

### Inception 分数 (IS)

Inception 分数旨在利用预训练 (pre-training)的图像分类模型（通常是在 ImageNet 上训练的 Inception V3）来衡量保真度和多样性。其原理有两方面：

1. **保真度：** 好的 GAN 生成的图像应该清晰可辨且包含有意义的物体。当通过 Inception 分类器时，条件概率分布 $p(y|x)$（图像 $x$ 属于类别 $y$ 的概率）应具有低熵。这意味着分类器对将图像归类到特定类别充满信心。
2. **多样性：** 生成器应生成涵盖数据集中多种类别的图像。因此，边缘概率分布 $p(y) = \int p(y|x) p_g(x) dx$（所有生成图像中类别的整体分布）应具有高熵。这表明生成器没有只生成少数类别的图像（模式崩溃）。

这两方面结合起来，使用条件分布和边缘分布之间的 Kullback-Leibler (KL) 散度，并对所有生成样本 $x \sim p_g$ 进行平均：


$$
IS = \exp(\mathbb{E}_{x \sim p_g} [ D_{KL}(p(y|x) || p(y)) ])
$$


更高的 Inception 分数通常被认为更好。然而，IS 存在局限性。它主要衡量生成的图像是否像 ImageNet 中的*任何*类别，而不一定衡量如果目标数据集与 ImageNet 不同时，生成的图像是否像目标数据集中的特定类别。它也不直接比较生成图像与目标分布中的真实图像，并且易受类别内对抗性样本影响。此外，研究表明 IS 并非总能与人类对图像质量的感知良好关联，特别是在类别内部的多样性方面。

### Fréchet Inception 距离 (FID)

Fréchet Inception 距离已成为一个更受欢迎和广泛采用的指标，因为它解决了 IS 的一些不足。FID 直接比较生成图像的统计数据与目标数据集中真实图像的统计数据。它在预训练 (pre-training) Inception V3 模型的特征空间中运行。

FID 的计算方法如下：

1. **特征提取：** 从预训练的 Inception V3 网络中选择特定层（通常是分类头之前的最终平均池化层）。将大量真实图像 ($x_r$) 和生成图像 ($x_g$) 通过网络处理到该层，以获取每张图像的特征向量 (vector)。
2. **分布建模：** 假设真实图像和生成图像的提取特征向量服从多元高斯分布。分别计算真实和生成集合的特征向量的均值向量 ($\mu_r$，$\mu_g$) 和协方差矩阵 ($\Sigma_r$，$\Sigma_g$)。
3. **距离计算：** 计算两个建模分布 ($N(\mu_r, \Sigma_r)$ 和 $N(\mu_g, \Sigma_g)$) 之间的 Fréchet 距离（在高斯分布中也称为 Wasserstein-2 距离）。公式如下：

   
   $$
   FID = ||\mu_r - \mu_g||^2_2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
   $$
   

   其中，$||\cdot||^2_2$ 表示均值向量间的平方欧几里得距离，$\text{Tr}$ 是矩阵的迹（对角线元素的和），$(\Sigma_r \Sigma_g)^{1/2}$ 是协方差矩阵乘积的矩阵平方根。

较低的 FID 分数表明生成图像特征的统计数据与真实图像特征的统计数据更相似，这表示生成分布 $p_g$ 更接近真实数据分布 $p_{data}$。较低的 FID 通常对应更好的图像质量和多样性。



![FID：特征分布比较](plots/2625-0.json)



> 使用 Inception 模型从真实和生成图像中提取的特征向量被建模为高斯分布。FID 衡量这些分布之间的距离，同时考虑它们的均值 ($\mu$) 和协方差 ($\Sigma$)。距离越小意味着相似度越高。

FID 对噪声更敏感，对模式崩溃也敏感（因为它会影响均值和协方差），并且与人类对图像质量的判断相关性优于 IS。然而，它需要真实和生成分布中的大量样本（通常为 10,000 到 50,000 个）才能可靠地估计均值和协方差矩阵。其计算也比 IS 更密集。

### 其他指标与考量

尽管 IS 和 FID 最为常见，但还存在其他衡量方法：

- **分布的精确度与召回率：** 这些指标借鉴了信息检索中的想法，用于 GAN 评估。精确度衡量被认为是逼真（保真度）的生成样本比例，而召回率衡量生成器可以生成的真实样本比例（多样性）。
- **感知路径长度 (PPL)：** 主要用于基于风格的生成器（如 StyleGAN），PPL 衡量生成器潜在空间的平滑度。潜在输入向量 (vector)的微小变化，理想情况下应导致输出图像出现微小、感知上平滑的变化。

**实用建议：**

- **标准化：** 在报告 FID 或 IS 时，请使用标准化的实现和相同的预训练 (pre-training) Inception 模型（通常是 V3），以确保不同研究或实验之间的公平比较。
- **样本量：** 请注意 FID 计算所需样本量，以确保稳定结果。样本过少可能导致分数不可靠。
- **互补评估：** 量化 (quantization)指标提供有价值的客观数据，但无法涵盖所有方面。务必通过对生成样本的定性目视检查来补充指标分数，以评估连贯性、细节和指标可能遗漏的潜在伪影等。

总之，评估 GAN 需要超越简单的损失函数 (loss function)。像 IS，特别是 FID 这样的指标，通过比较生成图像（通常在特征空间中）与真实图像的分布，提供了量化方法来评估生成图像的质量和多样性。理解这些指标的工作原理及其局限性，对于有效地开发和比较生成模型来说非常重要。

## 参考资料

- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) — Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen (2016)
  Journal: Advances in Neural Information Processing Systems 29; Volume: 29; DOI: [10.48550/arXiv.1606.03498](https://doi.org/10.48550/arXiv.1606.03498)
  介绍了Inception Score (IS)，这是评估GAN生成样本质量和多样性的基础指标。
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems 30; Volume: 30; DOI: [10.48550/arXiv.1706.08500](https://doi.org/10.48550/arXiv.1706.08500)
  提出了Fréchet Inception Distance (FID)，一种优于IS的指标，通过比较特征统计数据来更可靠地评估GAN。
