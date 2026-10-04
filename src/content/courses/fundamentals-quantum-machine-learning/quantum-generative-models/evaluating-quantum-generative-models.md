---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-generative-models"
lesson: "evaluating-quantum-generative-models"
sourceId: 1075
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-6-quantum-generative-models/evaluating-quantum-generative-models"
title: "评估量子生成模型"
description: "评估量子生成模型表现的指标和方法，如保真度和散度量度。"
order: 5
plots: ["plots/1075-0.json"]
sourceHash: "236d1f02e8f2965b23b7b465f88d088f5040a35937a58070229ada2e6796976b"
sourceCorrections: []
---

训练好量子电路玻尔兹曼机（QCBM）或量子生成对抗网络 (GAN)（QGAN）后，如何判断它是否实际有效地学到了目标数据分布？评估生成模型即使在传统上也是一项复杂任务，而量子方法带来了独特的考量。本节论述了评估量子生成模型表现的指标和方法。

与监督学习 (supervised learning)中准确度提供明确的参考标准不同，评估生成模型涉及评价单个样本的*质量*以及模型生成的整体分布$p_{\text{模型}}(x)$与真实数据分布$p_{\text{数据}}(x)$的*相似性*。直接计算QCBM和QGAN等模型的似然度$p_{\text{模型}}(x)$通常是不可行的，这与GAN等传统隐式生成模型类似。因此，我们主要依赖基于样本的评估技术。

### 比较概率分布

一些统计距离或散度可以量化 (quantization)$p_{\text{模型}}$和$p_{\text{数据}}$之间的差异。这些通常需要获取来自两个分布的样本。

#### 库尔巴克-莱布勒（KL）散度

信息论中的一个基本量度是KL散度：
$D_{KL}(p_{\text{数据}} || p_{\text{模型}}) = \sum_x p_{\text{数据}}(x) \log \frac{p_{\text{数据}}(x)}{p_{\text{模型}}(x)}$
或对于连续变量：
$D_{KL}(p_{\text{数据}} || p_{\text{模型}}) = \int p_{\text{数据}}(x) \log \frac{p_{\text{数据}}(x)}{p_{\text{模型}}(x)} dx$

$D_{KL} \ge 0$，当且仅当 $p_{\text{数据}} = p_{\text{模型}}$ 时，$D_{KL} = 0$。然而，KL散度存在不足：

- 它不对称：$D_{KL}(p_{\text{数据}} || p_{\text{模型}}) \neq D_{KL}(p_{\text{模型}} || p_{\text{数据}})$。
- 如果对于某些 $x$，$p_{\text{数据}}(x) > 0$ 但 $p_{\text{模型}}(x) = 0$，它可能变为无穷大。
- 从样本中准确估计它很困难，尤其是在高维度情况下，这通常需要分箱或密度估计技术，这些技术会引入自身的偏差。评估$p_{\text{模型}}(x)$本身可能就很难。

虽然最小化KL散度与最大化似然度有关，但其对于量子生成模型的直接计算或估计通常不切实际。

#### 詹森-香农（JS）散度

JS散度是KL散度的一个对称且有界的版本：
$D_{JS}(p_{\text{数据}} || p_{\text{模型}}) = \frac{1}{2} D_{KL}(p_{\text{数据}} || p_{\text{平均}}) + \frac{1}{2} D_{KL}(p_{\text{模型}} || p_{\text{平均}})$
$p_{\text{平均}} = \frac{1}{2} (p_{\text{数据}} + p_{\text{模型}})$。

JS散度的范围在0到$\log 2$之间（如果使用自然对数则为1），这使其比KL散度更稳定。当且仅当分布相同时，它为零。虽然JS散度对称且有界，但从样本中估计它在高维度时仍面临挑战，尽管在GAN文献中它常优于KL散度，部分原因是它构成了原始GAN目标函数的基础。

#### 最大均值差异（MMD）

MMD是一种非参数 (parameter)量度，其依据是当且仅当两个分布的所有矩都匹配时，它们是相同的。它测量在由核函数 $k(x, x')$ 定义的再生核希尔伯特空间（RKHS）$\mathcal{H}$ 中，这些分布均值嵌入 (embedding)之间的距离。

$MMD^2(p_{\text{数据}}, p_{\text{模型}}) = || \mathbb{E}_{x \sim p_{\text{数据}}}[\phi(x)] - \mathbb{E}_{x' \sim p_{\text{模型}}}[\phi(x')] ||^2_{\mathcal{H}}$
其中 $\phi(x) = k(x, \cdot)$ 是与核 $k$ 相关的特征映射。

使用核技巧，MMD可以从样本 $\{x_i\}_{i=1}^N \sim p_{\text{数据}}$ 和 $\{x'_j\}_{j=1}^M \sim p_{\text{模型}}$ 中估计：
$MMD^2_u = \frac{1}{N(N-1)} \sum_{i \neq j} k(x_i, x_j) + \frac{1}{M(M-1)} \sum_{i \neq j} k(x'_i, x'_j) - \frac{2}{NM} \sum_{i=1}^N \sum_{j=1}^M k(x_i, x'_j)$

常见的核选择包括高斯（RBF）核。MMD的表现很大程度上取决于所选的核及其参数（例如，高斯核的带宽$\sigma$）。MMD的估计通常比KL或JS散度计算成本低，并且不需要显式密度估计。它常用于评估GAN，甚至可以用作训练目标（例如在MMD-GAN中）。

### 评估样本质量

除了分布相似性之外，我们经常需要评价单个样本的质量或真实性。

- **定性评估：** 对于图像等数据类型，人类的视觉检查仍然是一种常见但主观的方法。生成的样本“看起来像”真实数据吗？
- **下游任务表现：** 一种更客观的方法是评估生成数据的实用性。在生成样本上训练一个独立的模型（例如分类器），并在真实的测试集上测试其表现。将此表现与仅在真实数据上训练的模型进行比较。如果使用合成数据训练的模型表现良好，则表明生成的样本捕捉到了相关特征。
- **领域专用指标：** 根据数据类型（例如金融、化学），可能存在特定的指标来评估生成样本的有效性或特性（例如生成分子的化学有效性）。

### 量子特有评估难题

评估量子生成模型涉及额外的障碍，这些障碍源于量子计算的特性：

1. **采样成本和噪声：** 生成样本 $x \sim p_{\text{模型}}(x)$ 需要执行量子电路（QCBM生成器或QGAN生成器）并进行测量。在噪声中等规模量子（NISQ）硬件上，此过程容易受到噪声（退相干、门误差、读出误差）的影响，这会扭曲生成的分布 $p_{\text{模型}}$。获取大量“干净”样本以准确估计指标可能耗时且资源密集。错误抑制技术可以提供帮助，但会增加额外开销。
2. **指标估计：** 估计KL、JS或MMD等统计散度需要足够的样本数量。量子采样带来的难题使得高维度分布比较的常见困难更为复杂。
3. **基准比较：** 公平地比较量子生成模型与传统模型，需要仔细考量计算资源（量子比特、电路深度、测量次数与传统计算时间、内存）并确保数据集和评估协议一致。



![直方图比较（低维示例）](plots/1075-0.json)



> 比较从真实数据样本和量子模型（例如QCBM或QGAN）生成的样本获得的经验概率分布。散度指标量化 (quantization)了这些直方图之间的差异。

### 评估的最佳实践

鉴于这些难题，评估策略应包含：

- **多项指标：** 结合使用分布相似性指标（例如MMD、估计的JS散度）和样本质量评估（例如视觉检查、下游任务表现）。单一数值无法完全反映情况。
- **传统基线：** 始终与在相同数据集上使用相同指标训练和评估的相关传统生成模型（例如GAN、VAE）进行比较。
- **样本量意识：** 确认可用样本所带来的限制。报告用于评估的样本数量，如果可能，分析指标对样本量的稳定性。
- **资源报告：** 记录所涉及的量子资源（量子比特数量、电路深度、测量次数、使用的错误抑制）和传统计算资源。

评估量子生成模型是一个活跃的研究方向。随着硬件性能提升和理论认识加深，评估技术将持续发展，旨在更可靠、更高效地评价这些强大的新型模型。

## 参考资料

- [A kernel two-sample test](https://www.jmlr.org/papers/v13/gretton12a.html) — Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard SchÃ¶lkopf, Alexander Smola (2012)
  Journal: JMLR; Publisher: JMLR; Volume: 13; Pages: 723-773; DOI: [10.5555/2228795.2228800](https://doi.org/10.5555/2228795.2228800)
  介绍了最大均值差异（MMD）作为一种非参数统计检验方法，用于比较分布，是该度量的核心参考文献。
- [Quantum generative adversarial learning](https://link.aps.org/doi/10.1103/PhysRevLett.121.040502) — Seth Lloyd, Christian Weedbrook (2018)
  Journal: Physical Review Letters; Publisher: American Physical Society; Volume: 121; Pages: 040502; DOI: [10.1103/PhysRevLett.121.040502](https://doi.org/10.1103/PhysRevLett.121.040502)
  介绍了量子生成对抗网络（QGANs）的框架，这是本节讨论的一个主要量子生成模型。
- [Generative modelling using quantum circuits](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.042334) — Marcello Benedetti, John Realpe-Gómez, Michael Rojas, Joseph Leyton-Ortega (2019)
  Journal: Physical Review A; Publisher: American Physical Society; Volume: 99; Pages: 042334; DOI: [10.1103/PhysRevA.99.042334](https://doi.org/10.1103/PhysRevA.99.042334)
  描述了使用量子电路进行生成式建模，侧重于量子电路玻尔兹曼机（QCBM），这是一个重要的量子生成模型。
- [Pros and cons of GAN evaluation measures](https://doi.org/10.1016/j.cviu.2018.10.009) — Ali Borji (2018)
  Journal: Computer Vision and Image Understanding; Publisher: Elsevier; Volume: 179; Pages: 41-65; DOI: [10.1016/j.cviu.2018.10.009](https://doi.org/10.1016/j.cviu.2018.10.009)
  讨论了经典生成对抗网络各种评估方法的优缺点，为理解生成模型度量提供了背景。
