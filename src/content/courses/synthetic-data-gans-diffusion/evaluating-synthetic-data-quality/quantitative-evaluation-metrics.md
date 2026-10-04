---
course: "synthetic-data-gans-diffusion"
chapter: "evaluating-synthetic-data-quality"
lesson: "quantitative-evaluation-metrics"
sourceId: 5717
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-5-evaluating-synthetic-data-quality/quantitative-evaluation-metrics"
title: "定量指标：IS、FID、精确率、召回率"
description: "理解并实现Inception分数、Fréchet Inception距离和相关指标。"
order: 2
plots: ["plots/5717-0.json"]
sourceHash: "1b4073aad52b7696aba4818ec7a5883230ccd9c9f36e73cd48b82bb9393fd7a8"
sourceCorrections: []
---

尽管视觉检查能提供初步判断，但仅依赖主观评估不足以客观地评估生成模型。我们需要客观的定量测量方法来比较不同模型、追踪训练进展，并了解生成器的具体优点和不足。已经开发了一些指标，主要使用Inception等预训练 (pre-training)图像分类网络提取的特征。其中最常用的指标包括：Inception分数 (IS)、Fréchet Inception距离 (FID) 以及为生成建模调整的精确率/召回率。

### Inception 分数 (IS)

Inception分数 (IS) 是首批被广泛采用的用于GAN评估的定量指标之一，特别是针对图像生成。它旨在同时衡量生成样本的两个理想特性：

1. **样本质量：** 生成图像应包含有意义的物体。对于图像 $x$，好的分类器应以高置信度将其归类到特定类别。这意味着条件类别分布 $p(y|x)$ 具有低熵。
2. **样本多样性：** 生成器应产生覆盖不同类别的多样化图像。对于所有生成的图像，边缘分布 $p(y) = \int p(y|x) p_g(x) dx$ ($p_g$ 为生成器的分布) 应具有高熵，表明表示了广泛的类别。

IS 通过计算特定生成图像 $x$ 的条件分布 $p(y|x)$ 与所有生成图像的平均边缘分布 $p(y)$ 之间的Kullback-Leibler (KL) 散度来结合这些特性。然后将分数取指数：


$$
IS(G) = \exp(\mathbb{E}_{x \sim p_g} [D_{KL}(p(y|x) || p(y))])
$$


计算IS时，生成大量样本，经过预训练 (pre-training)的Inception网络（通常是ImageNet上训练的Inception v3），结果类别概率用于估计 $p(y|x)$ 和 $p(y)$。较高的IS通常被认为更好。

然而，IS有明显的局限：

- 它不直接比较生成分布与真实数据分布。
- 生成器可以通过产生与目标数据集完全不同的高质量、多样化图像来获得高IS。
- 它对预训练分类器的属性很敏感。
- 它主要衡量ImageNet类别之间的多样性，这对于内容不同的数据集可能不适用。
- 已经表明，IS可以在没有真正提高样本质量的情况下被人为地提高。

由于这些不足，尽管IS具有历史重要性，但它经常被FID等指标补充或取代。

### Fréchet Inception 距离 (FID)

Fréchet Inception距离 (FID) 已成为评估生成图像质量的标准指标。与IS不同，FID明确比较生成样本与真实样本的统计数据。它衡量真实和生成图像的分布在由预训练 (pre-training)Inception v3网络中间层定义的特征空间中的距离。

计算涉及以下步骤：

1. 使用Inception v3模型的特定层（通常是最终平均池化层的输出，得到2048维向量 (vector)），将一组真实图像 ($X_r$) 和一组生成图像 ($X_g$) 嵌入 (embedding)到特征空间中。
2. 假设真实和生成集的这些特征向量都遵循多元高斯分布。分别计算真实和生成图像特征的均值向量 ($\mu_r, \mu_g$) 和协方差矩阵 ($\Sigma_r, \Sigma_g$)。
3. 计算这两个高斯分布之间的Fréchet距离（也称为Wasserstein-2距离）：


$$
FID(X_r, X_g) = ||\mu_r - \mu_g||^2_2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$


这里，$||\cdot||^2_2$ 表示平方欧几里得范数（均值向量元素平方差之和），$\text{Tr}$ 是矩阵的迹（对角线元素之和），而 $(\Sigma_r \Sigma_g)^{1/2}$ 是协方差矩阵乘积的矩阵平方根。

**解释：**

- 较低的FID表示生成图像特征的分布更接近真实图像特征的分布，表明更好的质量和多样性相似度。FID为0意味着在此特征空间中分布是相同的。
- 第一项 ($||\mu_r - \mu_g||^2_2$) 衡量均值之间的距离（捕捉平均特征的差异）。
- 涉及迹的第二项衡量与协方差矩阵相关的距离（捕捉特征变化和关联方式的差异）。

FID解决了IS的几个不足之处：

- 它直接比较生成数据与真实数据。
- 它比IS对模式坍塌（低多样性）更敏感。
- 它通常比IS更能抵抗噪声。

尽管如此，FID并非没有需要考虑的地方：

- 它依赖于Inception v3特征空间，这对于所有数据集或任务可能不是最优的。
- 计算值取决于用于估计的样本数量；比较FID分数需要使用相同数量的真实和生成样本（通常是10,000或50,000）。
- 计算FID可能是计算密集型任务，特别是高维特征的协方差矩阵及其平方根。本章后面你会亲自动手进行这项计算。

### 精确率和召回率

虽然FID提供一个单一数字来总结分布之间的距离，但它不分别量化 (quantization)保真度（质量）和多样性。借鉴分类方面的术语，精确率和召回率指标为生成模型调整，以提供更细致的分析。

想象真实数据流形（底层结构）和生成数据流形在合适的特征空间中（例如通过VGG或Inception嵌入 (embedding)获得）。

- **精确率：** 衡量落在真实数据流形支撑范围内的生成样本比例。高精确率意味着大多数生成样本是真实的或合理的，符合真实数据分布。它回答了：“生成的样本是否具有高保真度？”
- **召回率：** 衡量被生成数据流形覆盖的真实数据流形比例。高召回率意味着生成器可以生成覆盖真实数据中全部多样性的样本。它回答了：“生成器能否捕获真实数据的多样性？”

精确计算这些指标通常涉及复杂的流形估计和高维空间 (high-dimensional space)中的距离计算。常见方法使用特征空间中的k近邻（k-NN）：

1. 将大量真实 ($N_r$) 和生成 ($N_g$) 样本嵌入到特征空间中。
2. 对于每个生成样本，在真实样本中找到其k个最近邻居。如果生成样本的第k个最近真实邻居在特定距离阈值内（或者样本位于估计的真实流形内），则有助于精确率。
3. 类似地，对于每个真实样本，在生成样本中找到其k个最近邻居。如果真实样本被附近的生成样本“覆盖”，则有助于召回率。

具体定义和计算方法可能有所不同（参见Kynkäänniemi等人，2019的常见方法）。



![特征空间中的精确率与召回率 (示意图)](plots/5717-0.json)



> 精确率和召回率理念的示意，使用简化的二维特征空间表示。真实数据点显示为蓝色圆圈。粉色叉号表示高精确率（真实样本）但低召回率（缺失真实分布的一部分）。橙色菱形表示低精确率（蓝色区域外的一些不真实样本）但可能更高的召回率（覆盖了真实分布的更多范围）。绿色星号表示高精确率和高召回率的理想情况。

精确率和召回率提供有价值的见解：

- 生成器可能获得不错的FID但存在模式坍塌问题（高精确率，低召回率）。
- 另一个生成器可能覆盖模式但产生许多不真实的样本（低精确率，高召回率）。
- 这些指标有助于诊断此类权衡，指导模型改进。

在实践中，FID仍然是最常报告的单一指标，而精确率和召回率提供更详细的诊断视角，特别是在分析保真度-多样性权衡时。结合使用这些定量指标以及定性视觉检查来评估模型，提供了对生成器性能更全面的了解。

## 参考资料

- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) — Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen (2016)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 29; Pages: 2234-2242; DOI: [10.48550/arXiv.1606.03498](https://doi.org/10.48550/arXiv.1606.03498)
  介绍了Inception Score (IS)，作为早期评估GANs的量化指标，用于衡量样本质量和多样性。
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; Pages: 7547-7557; DOI: [10.48550/arXiv.1706.08500](https://doi.org/10.48550/arXiv.1706.08500)
  提出了Fréchet Inception Distance (FID)，作为评估生成图像质量的更稳健指标，通过比较特征分布进行衡量。
