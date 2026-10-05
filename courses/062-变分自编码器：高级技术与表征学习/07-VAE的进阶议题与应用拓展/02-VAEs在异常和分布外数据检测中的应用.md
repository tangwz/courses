# VAEs在异常和分布外数据检测中的应用

来源：[原文](https://apxml.com/zh/courses/vae-representation-learning/chapter-7-advanced-vae-topics-extensions/vaes-anomaly-detection)

[返回章节目录](README.md) · [返回课程目录](../README.md)

变分自编码器通过学习训练数据的压缩表示和生成模型，天然提供了一个识别偏离常规样本的有效体系。这种能力使它们非常适合异常检测和分布外 (OOD) 数据检测任务。基本假定是，一个在“正常”数据上训练的 VAE，能够熟练地重建已知样本，同时难以重建异常或 OOD 输入。

### 运用重建误差

采用 VAE 进行异常检测最直接的方法是基于重建概率。VAE 的训练目标是最小化与训练分布相似的数据点的重建误差。VAE 最大化的证据下界 (ELBO) 包含一个重建项，通常是一个对数似然，例如：


$$
\mathcal{L}_{重建}(x, \hat{x}) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
$$


其中 $x$ 是输入，$\hat{x}$ 是通过 VAE 得到的重建（即 $z \sim q_\phi(z|x)$，$\hat{x} \sim p_\theta(x|z)$）。

对于新的输入 $x_{new}$，其重建误差（例如，连续数据的均方误差，或从 $q_\phi(z|x_{new})$ 采样的 $z$ 对应的 $p_\theta(x_{new}|z)$ 的负对数似然）可作为异常分数。


$$
\text{异常分数}(x_{new}) = ||x_{new} - \hat{x}_{new}||^2
$$


或者更普遍地，$-\log p_\theta(x_{new}|\hat{z})$，其中 $\hat{z}$ 是 $q_\phi(z|x_{new})$ 的均值。“正常”或分布内样本应具有较低的重建误差，而异常或 OOD 样本预计会产生较高的重建误差，因为 VAE 的解码器 $p_\theta(x|z)$ 尚未优化以从编码器 $q_\phi(z|x)$ 产生的任何潜在编码 $z$ 准确生成这些样本。



[交互图表：重建误差分布](https://apxml.com/zh/courses/vae-representation-learning/chapter-7-advanced-vae-topics-extensions/vaes-anomaly-detection#plot-a6ysd5)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "histogram",
      "x": [
        0.05,
        0.06,
        0.075,
        0.055,
        0.065,
        0.08,
        0.09,
        0.07,
        0.085,
        0.095,
        0.1,
        0.075,
        0.065,
        0.055,
        0.105,
        0.11
      ],
      "name": "正常数据",
      "marker": {
        "color": "#228be6"
      },
      "opacity": 0.75,
      "nbinsx": 10
    },
    {
      "type": "histogram",
      "x": [
        0.3,
        0.35,
        0.4,
        0.28,
        0.42,
        0.45,
        0.5,
        0.38,
        0.48,
        0.52,
        0.33,
        0.39,
        0.41,
        0.46,
        0.55,
        0.6
      ],
      "name": "异常数据",
      "marker": {
        "color": "#fa5252"
      },
      "opacity": 0.75,
      "nbinsx": 10
    }
  ],
  "layout": {
    "barmode": "overlay",
    "title": {
      "text": "重建误差分布"
    },
    "xaxis": {
      "title": {
        "text": "重建误差分数"
      }
    },
    "yaxis": {
      "title": {
        "text": "数量"
      }
    },
    "legend": {
      "x": 0.6,
      "y": 0.95,
      "bgcolor": "rgba(255,255,255,0.5)"
    },
    "autosize": true,
    "paper_bgcolor": "#f8f9fa",
    "plot_bgcolor": "#f8f9fa",
    "font": {
      "color": "#495057"
    }
  }
}
```

</details>



> 正常数据的重建误差分数往往聚集在较低值，而异常数据点通常显示较高的分数，从而能够区分它们。

### 潜在空间特征在异常检测中的应用

除了重建，VAE 学到的潜在空间 $z$ 提供了识别异常的另一个途径。

1. **距先验的距离：** 编码器 $q_\phi(z|x)$ 将输入映射到潜在空间中的分布。对于分布内数据，这些编码分布通过 KL 散度项 $D_{KL}(q_\phi(z|x) || p_\theta(z))$ 进行正则化 (regularization)，使其接近先验 $p_\theta(z)$（通常是标准正态分布 $\mathcal{N}(0, I)$）。异常输入可能被映射到潜在空间中远离 $p_\theta(z)$ 高密度区域的位置。KL 散度值本身，或 $q_\phi(z|x)$ 均值与先验均值之间的马哈拉诺比斯距离，可用作异常分数。
2. **潜在空间中的密度：** 正常数据点预计会映射到学到的潜在流形中更密集的区域。异常可能落入更稀疏、数据较少的区域。可以采用核密度估计 (KDE) 等技术，对正常数据的验证集中的潜在编码在 $z = \mu_\phi(x)$ 附近估计密度，尽管这可能计算量大且对维度敏感。

> 正常数据（蓝色点）通常聚集在潜在先验（虚线椭圆）所偏好的区域内或附近。异常数据（红色点 A1, A2, A3）可能被编码到远离此中心密集区的位置，或进入潜在空间中更稀疏的区域。

### 运用 ELBO 或其组成部分

ELBO 本身，$\mathcal{L}(x; \theta, \phi)$，是对数边际似然 $\log p(x)$ 的一个近似。


$$
\mathcal{L}(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p_\theta(z))
$$


较低的 ELBO 值可能表明是一个 OOD 样本。然而，仅依赖 ELBO 有时可能产生误导。高度灵活的解码器可能会对易于重建的简单 OOD 样本赋予高似然（从而产生高 ELBO），即使这些样本在语义上与训练数据不同。一些研究表明，$p(x)$ 的典型集（VAE 表现良好的区域）和 $p(x)$ 的高密度区域在复杂数据分布下可能无法完美对齐 (alignment)，这会使使用原始似然进行 OOD 检测变得复杂。

结合多种指标，例如负重建误差和 KL 散度的加权和，通常能产生更好的异常分数：


$$
\text{异常分数}(x) = \alpha \cdot (-\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]) + (1-\alpha) \cdot D_{KL}(q_\phi(z|x) || p_\theta(z))
$$


超参数 (parameter) (hyperparameter) $\alpha$ 平衡这两个项，通常需要调整。

### 设置检测阈值

一旦异常评分机制到位（例如，重建误差、潜在 KL 散度或组合分数），就必须确定一个阈值。任何分数超过此阈值的样本都被标记 (token)为异常。阈值选择的常见策略包括：

- **经验百分位数：** 计算正常数据的留出验证集的异常分数。阈值可以设置为这些分数的高百分位数（例如，95 或 99 百分位数）。这种方法假定正常数据上的一小部分误报是可接受的。
- **统计建模：** 将正常验证数据中的异常分数拟合到某种分布（例如，高斯分布或极值理论分布），并根据期望的 p 值或误报率设置阈值。
- **监督式阈值设定（如果可用）：** 如果有一个用于验证的少量正常和异常样本的带标签数据集，可以选择阈值来优化特定指标，如 F1 分数，或在该验证集上达到特定的精度/召回率目标。

### 实际考量与难题

尽管 VAE 为异常检测提供了一种合理的方法，但有几个因素会影响其性能：

- **模型架构：** 编码器和解码器网络的容量和架构很重要。一个表现力过强的 VAE 可能学会过度重建某些类型的异常（过度泛化），而容量不足的 VAE 可能无法准确建模正常数据分布，即使是正常样本也可能导致高重建误差。
- **潜在维度：** 潜在空间 $d_z$ 的大小会影响性能。过小的 $d_z$ 可能过度压缩信息，丢失区分正常和异常的精微细节。过大的 $d_z$ 可能使 KL 散度项难以优化，或导致“后验坍缩”问题，即 $q_\phi(z|x)$ 对所有 $x$ 都变得信息量不足且类似于 $p_\theta(z)$。
- **训练数据的性质：** VAE 对“正常”的定义完全由其训练数据决定。如果训练集无意中包含异常，或未能全面覆盖正常行为的所有变体，VAE 的性能会下降。
- **异常类型：** VAE 通常更擅长检测与训练数据结构不同或位于学到的数据流形之外的异常。对于流形内的精微变体，或与正常数据共享许多低级特征的异常，它们可能难以处理。
- **后验坍缩：** 如果 ELBO 中的 KL 散度项在训练中占据主导地位，导致 $q_\phi(z|x) \approx p_\theta(z)$（后验坍缩），则潜在表示 $z$ 对 $x$ 的信息量会减少。这会削弱依赖潜在空间特征的异常检测方法。前面章节讨论的缓解后验坍缩的技术（例如 $\beta$-VAE、自由位）在此可能有所帮助。

### 扩展与精进

更高级的基于 VAE 的异常检测可能涉及：

- **VAE 集成：** 训练多个具有不同初始化或架构的 VAE，并组合它们的异常分数，可以提高稳定性。
- **迭代精化/检测：** 有些方法采用迭代过程，其中高置信度异常从训练集中移除（或降低权重 (weight)），然后重新训练 VAE。
- **对比方法：** 训练 VAE 以明确区分正常数据和人工生成的“临界”异常。
- **带有归一化 (normalization)流的 VAE：** 运用归一化流以获得更具表达力的先验 $p_\theta(z)$ 或后验 $q_\phi(z|x)$，可以带来更好的密度估计，并可能改进 OOD 检测，尽管这些模型训练起来可能更复杂。

总之，VAE 通过学习正常数据的模型，为异常和 OOD 检测提供了一个多功能工具集。异常分数的选择，无论是基于重建误差、潜在空间属性，还是 ELBO 本身，以及精心设计的模型和阈值选择，对于在实践中取得有效性能都非常重要。理解这些细节使工程师和研究人员能够将 VAE 调整应用于广泛的检测情境。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: arXiv preprint arXiv:1312.6114; DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍变分自编码器的原始论文，为变分自编码器在生成建模和异常检测等应用中奠定基础框架。
- [Variational Autoencoder based Anomaly Detection using Reconstruction Probability](http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf) — Jinwon An, Sungzoon Cho (2015)
  Journal: Special Lecture on IE; Volume: 2; Pages: 1-9
  一篇主要参考文献，详细介绍了利用VAE进行异常检测，将重建误差作为关键异常分数。
- [Do Deep Generative Models Know What They Don't Know?](https://arxiv.org/abs/1802.04013) — Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, and Balaji Lakshminarayanan (2018)
  Journal: International Conference on Learning Representations (ICLR); Publisher: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1802.04013](https://doi.org/10.48550/arXiv.1802.04013)
  这篇论文强调了使用深度生成模型（包括VAE）的原始似然进行可靠的域外检测的局限性，为基于VAE的异常评分提供重要背景。
- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://arxiv.org/abs/1611.00005) — Irina Higgins, Loïc Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1611.00005](https://doi.org/10.48550/arXiv.1611.00005)
  提出一种VAE的变体，通过修改ELBO来鼓励解耦的潜在表示，从而改善潜在空间的质量，有助于异常检测和缓解后验坍塌问题。

---

[上一节](01-%E4%BD%BF%E7%94%A8%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%EF%BC%88VAE%EF%BC%89%E7%9A%84%E5%8D%8A%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0.md) · [下一节](03-%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%20%28GANs%29%20%E5%AF%B9%E6%AF%94%20%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%20%28VAEs%29%EF%BC%9A%E4%B8%80%E9%A1%B9%E6%AF%94%E8%BE%83%E5%88%86%E6%9E%90.md)
