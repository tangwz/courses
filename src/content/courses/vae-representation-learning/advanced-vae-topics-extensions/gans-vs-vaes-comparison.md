---
course: "vae-representation-learning"
chapter: "advanced-vae-topics-extensions"
lesson: "gans-vs-vaes-comparison"
sourceId: 6382
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-7-advanced-vae-topics-extensions/gans-vs-vaes-comparison"
title: "生成对抗网络 (GANs) 对比 变分自编码器 (VAEs)：一项比较分析"
description: "对 VAEs 和 GANs 的详细比较，概述它们各自的优点和缺点。"
order: 3
plots: []
sourceHash: "25645757589c823b0ff9dda0a891666c0a4c2731375c9f0f3dc17e40941c545a"
sourceCorrections: []
---

尽管变分自编码器 (VAEs) 为生成建模提供了一个强大的概率框架，但它们并非唯一的生成模型。生成对抗网络 (GAN) (GANs) 呈现了一种独特且高效的替代方案。了解 VAEs 和 GANs 的根本区别、优势和不足，有助于为特定任务选择合适的模型，并理解结合两者优点的各种混合方法。

### 核心架构与目标差异

VAEs 和 GANs 从根本不同的角度处理学习数据分布的问题。

**变分自编码器 (VAEs):**
如前所述，VAEs 包含一个编码器网络 $q_\phi(z|x)$，它将输入数据 $x$ 映射到一个低维潜在空间的分布，以及一个解码器网络 $p_\theta(x|z)$，它将潜在空间中的点 $z$ 映射回数据空间。训练目标是最大化证据下界 (ELBO)：


$$
\mathcal{L}_{VAE}(\theta, \phi; x) = \mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
$$


此目标平衡了两个项：

1. **重建对数似然：** 促使解码器从其潜在表示准确地重建输入数据。
2. **KL散度正则项：** 促使近似后验分布 $q_\phi(z|x)$ 接近预定义的先验分布 $p(z)$ (通常是标准高斯分布)。这有助于潜在空间更有序，更便于采样。

VAEs 通过这些概率组成部分显式地建模数据生成过程，目标是学习真实数据分布 $p(x)$ 的近似。

**生成对抗网络 (GAN) (GANs):**
GANs，由 Ian Goodfellow 等人于 2014 年提出，采用博弈论方法。一个 GAN 包含两个相互竞争训练的神经网络 (neural network)：

1. **生成器 (G)：** 以随机噪声 $z$ (从简单的先验 $p_z(z)$ 中采样，例如高斯或均匀分布) 作为输入，试图将其转换为类似真实数据的样本 $G(z)$。
2. **判别器 (D)：** 一个二分类器，试图区分来自真实数据分布 $p_{data}(x)$ 的真实数据样本 $x$ 以及生成器产生的虚假样本 $G(z)$。

训练涉及一个最小-最大博弈，其中判别器试图最大化其分类准确性，而生成器试图通过生成越来越真实的样本来最小化判别器的准确性。标准的 GAN 目标函数是：


$$
\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$


与 VAEs 不同，GANs 不显式表示概率密度 $p(x)$。而是学习直接从数据分布中采样。

下图展示了高级架构差异：

> VAEs 和 GANs 中数据流与目标的高级比较。VAEs 侧重于通过 ELBO 进行重建和正则化 (regularization)潜在空间。GANs 涉及生成器和判别器之间的最小-最大博弈。

### 训练特点

不同的目标导致了不同的训练动态：

- **VAEs：** VAEs 的训练通常更稳定。ELBO 提供了一个单一、明确的损失函数 (loss function)，可以使用标准梯度下降 (gradient descent)技术进行优化。进展通常可以通过跟踪重建误差和 KL 散度项来观察。然而，VAEs 可能会遇到诸如“后验坍塌”之类的问题，即编码器将所有输入映射到潜在空间的一个小区域，从而有效地忽略了输入 $x$，或者 KL 项消失，导致 $q_\phi(z|x)$ 变得与先验 $p(z)$ 过于相似。
- **GANs：** GANs 的训练出了名的具有挑战性，并且可能不稳定。最小-最大博弈可能难以收敛。常见的失败模式包括：
  - **模式坍塌：** 生成器只生成有限种类的样本，未能捕获数据分布的完整多样性。
  - **梯度消失：** 判别器变得过于熟练，为生成器的学习提供很少或没有梯度信息。
  - **不收敛：** 生成器和判别器参数 (parameter)可能震荡或发散。
    实现生成器和判别器之间稳定的纳什均衡通常需要仔细的架构设计、超参数 (hyperparameter)调整和专门的训练技术 (例如，Wasserstein GANs，谱范数归一化 (normalization))。生成器和判别器的损失值并不总是样本质量或收敛的直接指标。

### 样本质量与多样性

最显著的差异之一在于生成样本的质量：

- **VAEs：** VAEs 生成的样本，特别是使用标准高斯解码器时，通常显得有些模糊或“平均化”。这可能归因于重建损失中的平方误差部分，它倾向于通过平均来惩罚偏差，以及 ELBO 本身的性质，它可能不总是与感知质量完美匹配。然而，VAEs 通常擅长捕获整体数据流形并生成多样化的样本。更先进的 VAE 解码器，如自回归 (autoregressive)模型，可以显著提高样本的清晰度。
- **GANs：** GANs 以其生成非常清晰和逼真样本的能力而闻名，尤其是在图像方面。对抗训练过程促使生成器匹配真实数据分布的精细细节和纹理，以欺骗判别器。权衡之下，可能是由于模式坍塌导致多样性不足，即生成器集中生成数据分布的少数高质量模式。

### 潜在空间属性与推断

所学潜在空间 $z$ 的特性和执行推断的能力差异很大：

- **VAEs：** VAE 目标中的 KL 散度项促使学习到的潜在空间连续且相对平滑，通常类似于先验 $p(z)$。这使得 VAE 潜在空间非常适合样本间插值、潜在代码的语义操作以及通过从 $p(z)$ 采样和解码来生成新颖变体等任务。重要的是，VAEs 提供了一个显式编码器 $q_\phi(z|x)$，允许对给定数据点进行潜在表示的有效推断。
- **GANs：** 标准 GANs 不像 VAEs 那样天然地学习结构化的潜在空间。尽管生成器将噪声 $z$ 映射到数据，但这种关系可能复杂且纠缠。在 GAN 的噪声空间 $z$ 中进行插值可能不会总是在数据空间中产生语义上有意义的过渡。此外，标准 GANs 缺乏直接的推断机制，这意味着没有直接的方法获取与给定真实数据样本 $x$ 对应的潜在代码 $z$。双向 GANs (BiGANs) 或对抗学习推断 (ALI) 等扩展引入了编码器来解决这一局限。

### 评估方法

评估生成模型是一个开放的研究课题，偏好的指标也有所不同：

- **VAEs：**
  - **ELBO：** ELBO 本身可以用作指标，尽管较高的 ELBO 并不总是与更好的样本质量完美相关。
  - **重建误差：** 衡量 VAE 重建其输入的能力。
  - **对数似然：** 对于一些 VAEs，可以计算边际对数似然 $log p(x)$ 的估计值 (例如，使用重要性加权自编码器, IWAEs)，这是概率模型的常见指标。
  - **定性评估：** 对生成样本进行视觉检查。
  - **解耦指标：** 如果 VAE 旨在进行解耦表示学习 (例如，$\beta$-VAE)，则使用互信息间隙 (MIG) 或分离属性可预测性 (SAP) 等指标。
- **GANs：**
  - **Inception 得分 (IS)：** 使用预训练 (pre-training)的 Inception 网络测量样本质量（清晰度）和多样性。越高越好。
  - **Fréchet Inception 距离 (FID)：** 比较预训练网络（例如 Inception）对真实和生成样本的激活统计数据。较低的 FID 表明生成样本与真实样本更相似。FID 通常被认为比 IS 更具鲁棒性。
  - **定性评估：** 对样本真实性和多样性进行人工评估通常很有必要。
  - 用于多样性、模式覆盖等方面的专门指标。

难以直接比较 VAE 的 ELBO 与 GAN 的 FID 分数，因为它们衡量模型性能的不同方面。

### 差异总结

下表总结了主要区别：

| 特征 | 变分自编码器 (VAEs) | 生成对抗网络 (GAN) (GANs) |
| --- | --- | --- |
| **主要机制** | 编码器-解码器，概率性，显式密度近似 | 生成器-判别器，对抗博弈，隐式采样 |
| **目标** | 最大化证据下界 (ELBO) | G 和 D 之间的最小-最大博弈 |
| **训练稳定性** | 通常更稳定，单一目标函数 | 通常不稳定，易出现模式坍塌、梯度消失 |
| **样本质量** | 可能模糊 (特别是简单解码器)，多样性好 | 通常更清晰、更真实；可能遭遇模式坍塌 |
| **潜在空间** | 有序，因 KL 正则化 (regularization)而平滑，利于插值 | 可能不够有序，插值可能无意义 |
| \*\*推断 \$p(z | x)\$\*\* | 显式学习一个编码器 \$q\_\phi(z |
| **评估** | ELBO，重建损失，估计对数似然，样本质量 | FID，Inception 得分，样本质量，多样性指标 |
| **优势** | 训练稳定，具备概率依据，潜在空间有意义，显式推断 | 高保真样本，在图像生成方面表现出色 |
| **劣势** | 模糊样本 (可缓解)，后验坍塌，ELBO 是下界 | 训练不稳定，模式坍塌，难以评估，无直接推断 |

### VAEs 与 GANs 的选择

VAEs 和 GANs 之间的选择，或者是否考虑混合模型，很大程度上取决于具体的应用和要求：

- **选择 VAEs 的情况：**

  - 您需要一个结构化、平滑的潜在空间，用于语义操作、插值或学习解耦表示等任务。
  - 需要一个显式推断机制 (将数据编码到潜在空间)。
  - 概率可解释性或密度估计是主要考量。
  - 训练稳定性是主要考量。
  - 目标应用是异常检测（基于重建误差或潜在空间密度）或数据压缩。
- **选择 GANs 的情况：**

  - 主要目标是生成高度真实、清晰的样本，特别是在图像合成等应用中。
  - 您已准备好投入精力进行仔细调优和管理训练稳定性。
  - 显式密度模型或高度结构化的潜在空间并非主要需求。

这项比较分析表明，VAEs 和 GANs 具有互补的优势。VAEs 提供稳定的训练和表现良好的潜在空间，而 GANs 在生成清晰样本方面表现出色。这自然引出了一个问题：我们能否结合这些方法以获得两者的优点？以下部分将介绍 VAE-GANs 和对抗自编码器 (AAEs) 等混合模型，这些模型正试图实现这一点。

## 参考资料

- [Generative Adversarial Networks](https://doi.org/10.48550/arXiv.1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: arXiv preprint arXiv:1406.2661; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  这篇论文首次介绍了生成对抗网络（GANs），详细阐述了其架构、对抗性训练目标和初步实验结果。
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: arXiv preprint arXiv:1312.6114; DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  这篇开创性论文介绍了变分自编码器（VAEs），提出了其概率框架、ELBO目标函数和重参数化技巧。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本深度学习的标准教科书，其中第20章全面解释并比较了包括VAEs和GANs在内的多种生成模型。
- [Wasserstein Generative Adversarial Networks](http://proceedings.mlr.press/v70/arjovsky17a.html) — Martin Arjovsky, Soumith Chintala, Léon Bottou (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 214-223
  这篇论文介绍了 Wasserstein GANs（WGANs），这是一项重要的进展，显著提升了GANs训练的稳定性，并解决了模式崩溃等问题。
