---
course: "generative-adversarial-networks-gans"
chapter: "evaluation-of-gans"
lesson: "frechet-inception-distance-fid"
sourceId: 2736
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-5-evaluation-of-gans/frechet-inception-distance-fid"
title: "Fréchet Inception 距离 (FID): 公式"
description: "通过预训练网络的激活来计算FID，以比较分布。"
order: 4
plots: []
sourceHash: "b0850c21b6a140f4d6e035ea22e15b4517f8978462e46d18b92b34f6dde80e39"
sourceCorrections: []
---

尽管 Inception Score (IS) 提供了一个量化 (quantization)评估，但它主要关注生成图像自身的特性（基于 Inception 分类器的清晰度和多样性），而没有直接将它们的分布与真实数据分布进行比较。这有时可能产生误导，特别是当生成器产生高质量但不具多样性的样本（模式崩溃），并且这些样本在有限的类别上恰好能很好地欺骗 Inception 分类器时。

Fréchet Inception 距离 (FID) 通过直接比较从真实图像和生成图像中提取特征的统计数据来弥补这一不足。它提供了一种更科学的方法来衡量高维特征空间中这两种分布之间的距离。

### FID 的工作原理：比较特征分布

FID 的主要思路是使用预训练 (pre-training)的深度卷积网络（通常是 Inception v3 模型）生成的嵌入 (embedding)来表示真实图像集和生成图像集。FID 不使用最终的分类输出，而是使用来自中间层（通常是分类头之前的最终平均池化层）的激活。该层能获取丰富的、高层次的视觉特征。

以下是具体过程：

1. **特征提取：** 大量真实图像 ($N_r$) 和大量生成图像 ($N_g$) 经过预训练的 Inception v3 网络。收集每个图像所选中间层的激活。这会得到两组特征向量 (vector)：一组用于真实图像 ($X_r$)，一组用于生成图像 ($X_g$)。每个特征向量都存在于一个高维空间 (high-dimensional space)中（例如，典型的 Inception v3 层有 2048 维度）。
2. **分布建模：** FID 假定真实图像和生成图像的这些高维特征向量可以被多元高斯分布合理地建模。这是一种简化，但在实际应用中效果良好。
3. **计算统计量：** 为每组特征向量计算平均向量和协方差矩阵：

   - 真实图像：均值 $\mu_r$，协方差 $\Sigma_r$
   - 生成图像：均值 $\mu_g$，协方差 $\Sigma_g$

### FID 公式

Fréchet 距离是衡量两个多元高斯分布之间距离的一种方式。FID 分数是根据 Inception 特征的均值和协方差矩阵计算得出的，具体如下：


$$
FID(X_r, X_g) = ||\mu_r - \mu_g||^2_2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$


我们来分析这个公式：

- $||\mu_r - \mu_g||^2_2$: 这是真实图像 ($\mu_r$) 和生成图像 ($\mu_g$) 平均特征向量 (vector)之间的平方欧几里得距离（或 L2 范数平方）。它反映了两个集合的*平均*特征差异程度。距离越小，表示生成的图像在平均意义上与真实图像具有类似的高层次特征。
- $\text{Tr}(\dots)$: 这一项涉及协方差矩阵 ($\Sigma_r$ 和 $\Sigma_g$) 组合的迹（对角线元素的和）。协方差矩阵反映了不同特征维度之间的分散和相关性。这一项衡量了两种分布协方差结构之间的距离。$(\Sigma_r \Sigma_g)^{1/2}$ 表示协方差矩阵乘积的矩阵平方根。较小的迹值表明，生成图像特征的分散和相关性与真实图像类似。这部分对于体现生成样本相对于真实数据的*多样性*尤其有意义。

### 分数解读

FID 分数是一个非负值，其单位与特征空间中的平方距离有关。

- **值越低越好：** 较低的 FID 分数表示生成图像特征的分布与真实图像特征的分布更为接近。这说明生成的样本在质量和多样性上更优，以 Inception 网络的特征表示来看。
- **FID 为零：** 理论上，分数为 0 表示两个分布完全相同 ($\mu_r = \mu_g$ 和 $\Sigma_r = \Sigma_g$)，但实际中很少能达到。
- **比较应用：** FID 主要用于在特定数据集上比较不同的 GAN 模型或同一模型的不同训练检查点。它提供了一种量化 (quantization)方式来衡量生成性能。

> 该图描述了 FID 如何在 Inception 特征空间中衡量真实和生成图像特征分布（建模为高斯分布）之间的距离。FID 同时考量均值 ($\mu_r, \mu_g$) 和协方差矩阵 ($\Sigma_r, \Sigma_g$) 的差异。

与 Inception Score 相比，FID 通常被认为更值得信赖。它对模式崩溃（影响 $\Sigma_g$）和图像伪影（同时影响 $\mu_g$ 和 $\Sigma_g$）都很敏感。它还使得生成分布与目标真实分布的比较更为直接。需要注意的是，FID 的计算依赖于所选的预训练 (pre-training)模型 (Inception v3) 和使用的具体特征层。此外，为获得可靠的均值和协方差估计，计算一个稳定的 FID 分数需要从真实和生成集合中获取足够多的样本（通常为 10,000 或更多，常推荐 50,000 个）。

## 参考资料

- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.48550/arXiv.1706.08500](https://doi.org/10.48550/arXiv.1706.08500)
  提出了Fréchet Inception Distance (FID) 作为评估生成对抗网络的指标，详细阐述了其公式和相对于先前指标的优势。
- [A Survey of GAN Evaluation Metrics](https://doi.org/10.1016/j.cviu.2018.10.009) — Ali Borji (2019)
  Journal: Computer Vision and Image Understanding; Publisher: Elsevier; Volume: 179; Pages: 41-65; DOI: [10.1016/j.cviu.2018.10.009](https://doi.org/10.1016/j.cviu.2018.10.009)
  回顾了评估生成对抗网络的各种指标，包括对FID及其在其他定量度量中的地位的比较讨论。
- [Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play](https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/) — David Foster (2019)
  Publisher: O'Reilly Media
  提供了生成模型的实用介绍，其中一章专门讨论GAN的评估，涵盖了FID等指标。
