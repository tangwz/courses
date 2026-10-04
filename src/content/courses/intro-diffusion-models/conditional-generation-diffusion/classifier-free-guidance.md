---
course: "intro-diffusion-models"
chapter: "conditional-generation-diffusion"
lesson: "classifier-free-guidance"
sourceId: 5441
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-6-conditional-generation-diffusion/classifier-free-guidance"
title: "无分类器引导 (CFG)"
description: "通过联合训练条件和无条件模型，实现在无显式分类器的情况下进行引导的技术。"
order: 3
plots: []
sourceHash: "5bf3656f303b7f1cdc1ab313e756d07c4d1980532ac15c59acb6b633ea30d587"
sourceCorrections: []
---

虽然分类器引导提供了一种引导扩散过程的方式，但它需要在扩散模型之外训练和维护一个独立的分类器模型。这增加了复杂性和潜在的故障点。此外，分类器的梯度可能不总能与扩散模型的内部表示完美对齐 (alignment)，有时会导致伪影或次优结果。

无分类器引导 (CFG) 是一种优雅且有效的技术，用于在无需外部分类器的情况下实现条件生成。其主要理念是训练扩散模型本身来处理条件和无条件生成的情况。

### CFG 训练方式

为实现 CFG，噪声预测网络（通常是表示为 $\epsilon_\theta(x_t, t, y)$ 的 U-Net）在条件和无条件输入上进行训练。在训练过程中，条件信息 $y$（例如类别标签或文本嵌入 (embedding)）会在一定比例的时间（例如，10-20% 的训练样本）被随机丢弃或替换为特殊的“空”标记 (token) $\emptyset$。

因此，模型学习预测：

1. 给定条件下的噪声：$\epsilon_\theta(x_t, t, y)$
2. 无任何条件下的噪声：$\epsilon_\theta(x_t, t, \emptyset)$

这种联合训练迫使模型理解生成特定于条件 $y$ 的数据与生成总体分布典型数据之间的区别。两个任务使用相同的网络参数 (parameter) $\theta$。这意味着 U-Net 架构必须调整以接受条件信息 $y$ 作为额外输入，以及噪声图像 $x_t$ 和时间步 $t$。

### CFG 采样：引导去噪步骤

在采样（逆向扩散）过程中，我们运用模型预测条件和无条件噪声的能力。在每个去噪步骤 $t$，我们使用当前的噪声状态 $x_t$ 和时间步 $t$ 对 U-Net 执行两次前向传播：

- 一次传播提供条件噪声预测：$\epsilon_\theta(x_t, t, y)$，使用期望的条件 $y$。
- 另一次传播提供无条件噪声预测：$\epsilon_\theta(x_t, t, \emptyset)$，使用空标记 (token) $\emptyset$ 代替条件。

直觉是，这两个预测之间的差异 $\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset)$ 表示噪声空间中使生成趋向满足条件 $y$ 的方向。CFG 结合这些预测，生成一个调整后的噪声估计 $\hat{\epsilon}_\theta$，该估计在条件方向上进一步外推。

CFG 调整后的噪声预测公式为：


$$
\hat{\epsilon}_\theta(x_t, t, y, w) = \epsilon_\theta(x_t, t, \emptyset) + w (\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset))
$$


此处，$w$ 是*引导强度*。

- 如果 $w = 0$，公式简化为 $\hat{\epsilon}_\theta = \epsilon_\theta(x_t, t, \emptyset)$。这对应于纯粹的无条件生成，忽略 $y$。
- 如果 $w = 1$，我们得到 $\hat{\epsilon}_\theta = \epsilon_\theta(x_t, t, y)$。这等同于仅使用条件预测的标准条件生成，没有明确的引导增益。
- 如果 $w > 1$，过程将进行外推。它从无条件预测 $\epsilon_\theta(x_t, t, \emptyset)$ 开始，并沿着条件 $y$ 指示的方向（由差异向量 (vector)表示）进一步移动，按 $w$ 进行缩放。这有效地放大了条件 $y$ 对生成过程的影响。

然后，这个调整后的噪声估计 $\hat{\epsilon}_\theta(x_t, t, y, w)$ 被用于去噪更新规则（例如在 DDPM 或 DDIM 采样中），以计算噪声较小状态 $x_{t-1}$ 的估计值。

### 引导强度 (w)

引导强度 $w$ 是一个超参数 (parameter) (hyperparameter)，您可以通过它控制条件的影响力。它就像一个旋钮，用于平衡生成结果对条件的符合度与样本多样性和质量。

- **低 $w$ (例如，1-3)：** 弱引导。生成的样本倾向于更多样和富有创意，但可能仅松散地遵循所提供的条件 $y$。
- **中等 $w$ (例如，5-10)：** 通常提供一个良好平衡。样本通常能很好地符合条件 $y$，同时保持合理的视觉质量和一定程度的多样性。这个范围在文本到图像模型实践中常用。
- **高 $w$ (例如，15+)：** 强引导。样本非常紧密地遵循条件 $y$。然而，这有时会导致多样性降低（样本看起来相似）、颜色或特征潜在的过饱和，或由于外推将生成过程推向训练中较少涉足的区域而产生其他伪影。

为 $w$ 选择合适的值通常需要根据特定的扩散模型、训练数据集以及期望的输出特点进行一些实验。

> 一个图示，演示无分类器引导如何结合无条件噪声预测 $\epsilon_\theta(x_t, \emptyset)$ 和条件预测 $\epsilon_\theta(x_t, y)$，通过沿差异向量 (vector)进行外推，并由引导强度 $w$ 控制，以生成调整后的预测 $\hat{\epsilon}_\theta(x_t, y, w)$。此示例展示了 $w=2$ 时的外推。

### CFG 的优点

无分类器引导已成为条件扩散模型的一种标准技术，这归因于它的一些重要优点：

1. **无需额外分类器：** 它消除了训练、管理并确保独立分类模型兼容性的需要。这简化了整体流程。
2. **简便性：** 其实现主要涉及修改训练数据（通过随机丢弃条件）和采样循环（通过每步计算 U-Net 的两次前向传播，并使用 CFG 公式结合结果）。
3. **有效性：** CFG 通常能产生高质量的条件样本，这些样本严格遵循提供的引导。它在各种任务中，特别是在文本到图像生成方面，通常表现与分类器引导一样好或更好。
4. **灵活性：** 引导强度 $w$ 提供了一种简单直观的方式，可在推理 (inference)时控制条件符合度与样本多样性之间的权衡，而无需重新训练模型。

通过将条件机制直接整合到扩散模型的训练目标中，CFG 提供了一种强大且广泛采用的方法来控制生成过程。我们将在本章后续部分研究如何在实践中实现这项技术。

## 参考资料

- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — Jonathan Ho, Tim Salimans (2021)
  Journal: NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications; DOI: [10.48550/arXiv.2207.12598](https://doi.org/10.48550/arXiv.2207.12598)
  介绍了无分类器引导的概念，详细阐述了其训练方法、采样算法以及引导尺度的作用。
