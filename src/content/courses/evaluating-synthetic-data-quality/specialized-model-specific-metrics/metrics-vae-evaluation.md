---
course: "evaluating-synthetic-data-quality"
chapter: "specialized-model-specific-metrics"
lesson: "metrics-vae-evaluation"
sourceId: 5713
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-5-specialized-model-specific-metrics/metrics-vae-evaluation"
title: "VAE评估指标"
description: "讨论与变分自编码器性能及重建相关的指标。"
order: 5
plots: ["plots/5713-0.json"]
sourceHash: "7cfd2bcbd1ff413b2ddc7373b5661b9d9d7e20f37beaf9a8955616905e70f6fb"
sourceCorrections: []
---

变分自编码器（VAE）拥有一种独特的架构，由编码器 $q_\phi(z|x)$ 和解码器 $p_\theta(x|z)$ 组成，其训练目标是最大化数据对数似然的证据下界（ELBO）。这种结构影响了我们评估其性能的方式，要求我们使用能同时衡量自动编码能力和概率潜在空间 $z$ 质量的指标。

### 重建准确度

VAE的一个基本方面是其重建输入数据的能力。解码器 $p_\theta(x|z)$ 学习将由编码器 $q_\phi(z|x)$ 产生的潜在表示 $z$ 映射回原始数据空间。我们通过比较原始输入 $x$ 与通过编码器和解码器处理 $x$ 后生成的重建 $\hat{x}$ 来评估这一点。

常见的重建指标取决于数据类型：

- **均方误差（MSE）：** 适用于连续数据，例如图像中归一化 (normalization)的像素强度。它衡量原始数据点与重建数据点之间的平均平方差。
  
  $$
  MSE = \frac{1}{N} \sum_{i=1}^{N} ||x_i - \hat{x}_i||^2
  $$
  
  其中 $N$ 是数据点数量，$x_i$ 是原始数据点，$\hat{x}_i$ 是其重建。
- **二元交叉熵（BCE）：** 当数据是二元或表示概率时使用，例如二值化图像中的像素或伯努利分布的参数 (parameter)。
  
  $$
  BCE = -\frac{1}{N} \sum_{i=1}^{N} [x_i \log(\hat{x}_i) + (1 - x_i) \log(1 - \hat{x}_i)]
  $$
  
  在此，$\hat{x}_i$ 被解释为输出分布的概率参数（例如，像素的伯努利分布）。

低重建误差表明 VAE 通过编码-解码过程保留了信息。然而，实现良好的重建主要是衡量自编码器组件的性能，并不能保证 VAE 能从先验分布 $p(z)$ 生成高质量的新样本。一个 VAE 即使能完美重建输入，也可能拥有一个结构不良的潜在空间，从而导致生成性能不佳。

### 潜在空间质量

VAE 作为生成模型的有效性取决于其潜在空间的质量。VAE 的目标函数包含一个 Kullback-Leibler（KL）散度项，该项促使每个输入 $x$ 的近似后验分布 $q_\phi(z|x)$ 接近先验分布 $p(z)$，后者通常是一个标准多元高斯分布 $\mathcal{N}(0, I)$。


$$
\mathcal{L}_{ELBO}(x) = \mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
$$


尽管 ELBO 中的 KL 项侧重于单个后验，但我们通常关注的是*聚合后验*分布 $q_\phi(z) = \int q_\phi(z|x) p_{data}(x) dx$。理想情况下，这个聚合分布应与先验 $p(z)$ 匹配。

有几种技术可以帮助评估潜在空间：

1. **KL 散度监测：** 训练期间，监测平均 KL 散度项可以帮助了解潜在空间是否得到有效正则化 (regularization)。过低的 KL 项可能表明“后验坍塌”，即编码器忽略输入 $x$，且 $q_\phi(z|x)$ 坍塌到先验 $p(z)$，导致重建质量差。
2. **聚合后验与先验比较：** 训练后，可以通过编码整个数据集（或一个大样本）来估计聚合后验 $q_\phi(z)$，并查看所得潜在编码 $z$ 的分布。然后可以估计 KL 散度 $D_{KL}(q_\phi(z) || p(z))$。低值表明潜在空间与假定的先验结构良好匹配。
3. **可视化：** 对从编码数据集获得的潜在均值 $\mu_\phi(x)$ 应用 t-SNE 或 UMAP 等降维技术，可以显示潜在空间的结构。对于有标签数据（如 MNIST 数字），与不同类别对应的不同簇表明有意义的表示学习。簇之间的平滑过渡可能表示良好的插值能力。

   
   
   ![VAE潜在空间的UMAP可视化（MNIST数字）](plots/5713-0.json)
   
   

   > 不同的 MNIST 数字的潜在均值的 UMAP 投影示例。明显的分离表明潜在空间捕获了类别信息。（注意：数据点为示意性。）
4. **采样质量：** 生成性能的最终检验是，通过从 $p(z)$（先验）中抽样 $z$ 并使其通过解码器 $p_\theta(x|z)$ 所产生的样本的质量。这些生成的样本 $\hat{x}_{new}$ 可以使用前面讨论过的特定模态指标进行评估（例如，图像的 FID，表格数据的统计测试，文本的困惑度）。高质量的生成样本表明学习到了一个良好的生成分布。

### 解耦度指标

对于某些应用，期望潜在空间 $z = (z_1, z_2, ..., z_d)$ 的各个维度对应数据中独立且可解释的变异因素（例如，物体旋转、颜色、大小）。这个属性被称为解耦。虽然标准 VAE 不会明确地对此进行优化，但像 $\beta$-VAE 这样的变体引入了修改以促成它。评估解耦需要专门的指标，通常依赖于具有已知真实变异因素的数据集：

- **Beta-VAE 分数：** 衡量个体潜在维度捕获单一真实因素的程度。
- **FactorVAE 分数：** 基于多数投票分类量化 (quantization)潜在维度的独立性。
- **互信息间隙（MIG）：** 评估每个潜在维度主要包含关于一个真实因素信息的程度。
- **DCI 解耦度：** 使用简单回归器，基于潜在维度中真实因素的可预测性来评估解耦度。

实现良好的解耦通常以牺牲重建质量为代价，这突显了 VAE 评估中的另一个权衡。

### 对数似然估计

VAE 提供对数似然 $\log p(x)$ 的下界（ELBO），该下界在训练期间进行优化。尽管 ELBO 本身可以在测试集上报告，但它通常低估了真实对数似然。更准确的估计可以使用重要性采样或退火重要性采样（AIS）等技术获得。这些方法涉及对于给定的 $x$，从 $q_\phi(z|x)$ 抽取多个样本，以更好地近似计算 $p(x) = \int p_\theta(x|z) p(z) dz$ 所需的积分。在保留数据上估计的对数似然更高，通常表明生成模型拟合得更好。这种估计似然的能力是 VAE 相对于生成对抗网络（GAN）的优势。

评估一个 VAE 需要一个全面的视角。你需要考虑它重建数据的好坏、其潜在空间的结构和一致性、从先验生成的新样本的质量，以及可能的专门属性，如解耦度或估计的数据对数似然。这些方面的相对重要性在很大程度上取决于 VAE 模型的预期应用。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍了变分自编码器（VAE）框架和证据下界（ELBO）目标。
- [Importance Weighted Autoencoders](https://arxiv.org/abs/1502.05477) — Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1502.05477](https://doi.org/10.48550/arXiv.1502.05477)
  讨论了在VAE中使用重要性采样进行更精确的对数似然估计。
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://arxiv.org/abs/1606.05579) — Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1606.05579](https://doi.org/10.48550/arXiv.1606.05579)
  介绍了用于学习解耦表示的$\beta$-VAE模型和$\beta$-VAE分数。
- [Disentangling by Factorizing](https://proceedings.mlr.press/v80/kim18b.html) — Hyunjik Kim, Andriy Mnih (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning; Publisher: PMLR; Volume: 80; Pages: 2649-2658; DOI: [10.48550/arXiv.1802.05983](https://doi.org/10.48550/arXiv.1802.05983)
  提出了用于无监督解耦表示学习的FactorVAE及其评估指标。
- [Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/pdf/1802.04942) — Xi Chen, Toufiq Parvez, Roger Grosse, Jimmy Ba (2018)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS Foundation; Volume: 31; DOI: [10.48550/arXiv.1802.04942](https://doi.org/10.48550/arXiv.1802.04942)
  引入了互信息间隙（MIG）作为评估解耦表示的指标。
