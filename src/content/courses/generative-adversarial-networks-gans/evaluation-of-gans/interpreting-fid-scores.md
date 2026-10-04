---
course: "generative-adversarial-networks-gans"
chapter: "evaluation-of-gans"
lesson: "interpreting-fid-scores"
sourceId: 2739
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-5-evaluation-of-gans/interpreting-fid-scores"
title: "解读 FID 分数"
description: "如何使用 FID 分数比较不同的 GAN 模型和训练进展。"
order: 5
plots: ["plots/2739-0.json"]
sourceHash: "4018f002fec89ed617d6fec5fa94344ed5b202387e386ccfa4931ed93253317e"
sourceCorrections: []
---

Fréchet Inception Distance (FID) 是一种定量衡量生成图像分布与真实图像分布之间相似度的方法，其依据是预训练 (pre-training)的 Inception 网络提取的特征。FID 分数越低，表示这两种分布越接近，表明生成的图像在视觉质量和多样性方面更接近真实图像。然而，要理解某个 FID 具体数值的含义，需要结合具体情况。

### 比较 FID 分数

FID 分数在比较时最有价值。一个独立的 FID 分数并不能说明全部情况。它的主要用途有：

1. **模型比较：** 在评估于*相同数据集*上训练的不同 GAN 架构或变体时，获得较低 FID 的模型通常被认为是更好的。例如，在 FFHQ 数据集上比较 StyleGAN2 和 BigGAN，FID 较低的模型表明其生成分布根据 Inception 特征来看更接近真实人脸分布。
2. **跟踪训练进展：** 监测训练迭代周期中的 FID 变化是一种常见做法。FID 呈下降趋势表明生成器模仿真实数据分布的能力正在提升。分数停滞不前或上升可能预示着训练停滞或不稳定。
3. **消融研究：** 在评估特定技术（例如，新的损失函数 (loss function)组成部分、归一化 (normalization)层）的影响时，比较使用和不使用该技术时的 FID 分数，能为该技术的有效性提供量化 (quantization)依据。
4. **基准测试：** 将你的模型在标准数据集（如 CIFAR-10、CelebA、LSUN）上的 FID 与已发表研究中报告的分数进行比较，有助于衡量其相对于当前最佳表现的水平。

### 什么是“好”的分数？

对于“好”的 FID 分数，没有一个统一的判断标准。它很大程度上取决于：

- **数据集复杂性：** 为 ImageNet 或高分辨率人脸（FFHQ）等复杂数据集生成逼真图像本身就更难，与 MNIST 或 CIFAR-10 等简单数据集相比，通常会导致最低可达到的 FID 分数更高。
- **图像分辨率：** 更高分辨率的生成任务通常与更高的 FID 分数相关。
- **当前最佳水平：** 随着研究进展，“好”的标准也在变化。几年前被认为是优异的分数，今天可能只是一般水平。

作为标准基准测试（如 CelebA 256x256 或 FFHQ 1024x1024）的粗略指导，个位数的分数（例如 2-5）通常代表非常高质量的生成，根据 Inception 网络的感知，与真实数据分布非常接近。10-30 范围内的分数可能表明生成质量良好，而高于 50 的分数通常表示与真实图像相比，质量或多样性存在明显差异。然而，务必始终参考你所使用的数据集和分辨率的具体基准测试结果。



![训练期间的 FID 分数 (示例)](plots/2739-0.json)



> 示例图表，显示了两个模型在相同数据集上训练周期中 FID 分数的降低情况。与模型 B 相比，模型 A 表现出更好的收敛性，FID 更低。

### 影响 FID 计算的因素

在计算或比较 FID 分数时，保持一致性很重要：

- **样本数量：** FID 的稳定性取决于使用足够的样本数量。常见的做法是使用 10,000 或 50,000 个生成样本和真实样本。使用过少的样本可能导致对分布统计数据（Inception 激活的均值和协方差）的估计产生噪声且不可靠。计算涉及估计真实图像和生成图像的 Inception 激活的均值（$\mu_r, \mu_g$）和协方差（$\Sigma_r, \Sigma_g$）矩阵。FID 的计算公式为：
  
  $$
  FID = ||\mu_r - \mu_g||^2_2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
  $$
  
  准确估计这些统计数据需要充足的样本量。
- **Inception 网络：** 确保所有比较都使用相同的预训练 (pre-training) Inception v3 模型实现。实现或权重 (weight)上的微小差异可能会影响提取的特征，从而影响最终得分。参考实现通常使用在 ImageNet 上预训练的权重。
- **图像预处理：** 在输入到 Inception 网络之前，真实图像和生成图像都必须经过相同的预处理步骤（调整大小到 Inception 预期的输入尺寸，通常是 299x299，以及归一化 (normalization)，通常是将像素值缩放到 [-1, 1] 或 [0, 1] 范围）。任何不一致都将使比较无效。

### 需考虑的局限性

尽管 FID 被广泛使用，但它并非一个完美的指标。请记住：

- **感知对齐 (alignment)：** FID 依赖于 Inception v3 的特征，该网络用于图像分类训练。尽管这些特征能捕获相关统计信息，但它们可能无法完全契合人类对图像质量或特定类型伪影的感知。一个模型即使获得了良好的 FID 分数，仍可能显示出不易察觉的重复模式或其他人类可以察觉的缺陷。
- **敏感性：** FID 对特定的失败模式（如模式崩溃，即低多样性）可能敏感，尽管通常不如 Inception Score 敏感。它衡量的是平均分布之间的距离，可能无法完全捕捉异常质量或精细细节。例如，如果生成器生成了完美的图像，但只覆盖了真实数据一半的模式，FID 分数会上升，但这可能无法完全反映与其他潜在问题相比，模式崩溃的严重性。
- **无法了解具体缺陷：** 单个 FID 数值不能说明模型表现不佳的*原因*。它无法区分低质量（模糊图像、伪影）和低多样性（模式崩溃）。因此，仍需要进行定性评估，以了解生成缺陷的性质。

总结来说，FID 是一种用于 GAN 定量评估的有效工具，在比较模型和跟踪训练进展方面尤为有用。解释分数时应采用相对视角，确保计算方法一致，并且始终用定性视觉检查生成样本来补充定量指标，以全面了解 GAN 的表现。

## 参考资料

- [Fréchet Inception Distance for Evaluating Generative Networks](https://arxiv.org/abs/1706.08500) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Günter Klambauer, Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems 30; Volume: 30; Pages: 630-640; DOI: [10.48550/arXiv.1706.08500](https://doi.org/10.48550/arXiv.1706.08500)
  介绍了Fréchet Inception Distance (FID) 指标，其数学公式，并展示了其在评估生成模型方面的有效性。
- [Are We There Yet? A Review of GAN Metrics](https://arxiv.org/abs/1711.10337) — Mario Lucic, Karol Kurach, Marcin Michalski, Sylvain Gelly, Olivier Bousquet (2018)
  Journal: NIPS'18; DOI: [10.48550/arXiv.1711.10337](https://doi.org/10.48550/arXiv.1711.10337)
  全面回顾并实证比较了评估生成对抗网络 (GAN) 的各种指标，包括FID，并讨论了它们的优缺点。
