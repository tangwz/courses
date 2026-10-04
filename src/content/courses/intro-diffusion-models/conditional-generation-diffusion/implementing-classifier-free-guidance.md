---
course: "intro-diffusion-models"
chapter: "conditional-generation-diffusion"
lesson: "implementing-classifier-free-guidance"
sourceId: 5444
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-6-conditional-generation-diffusion/implementing-classifier-free-guidance"
title: "实现分类器无关引导"
description: "修改用于CFG的网络输入和采样过程。"
order: 4
plots: []
sourceHash: "80644fe41af0e85a932ede488f790d5837d9b857c85ca2eb1bad4235909f4694"
sourceCorrections: []
---

分类器无关引导 (CFG) 提供了一种有效方法来引导生成过程，而无需依赖单独的预训练 (pre-training)分类器模型。这种技术避免了与分类器准确性或数据域不匹配相关的问题，并已成为现代扩散模型中的一种标准方法。

主要思路是训练一个单一的扩散模型 $\epsilon_\theta$，使其能够*同时*有条件地和无条件地运行。这是通过在训练阶段进行修改来实现的。

### 条件Dropout训练

在训练期间，对于每个数据样本 $x_0$ 及其对应的条件信息 $y$ (例如类别标签或文本嵌入 (embedding))，我们执行以下步骤：

1. 采样一个时间步长 $t \sim \mathcal{U}(1, T)$。
2. 采样噪声 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$。
3. 使用前向过程方程计算噪声样本 $x_t$：$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$。
4. 以一定的概率 $p_{uncond}$ (例如，10-20%)，将实际的条件信息 $y$ 替换为一个特殊的*空*或*无条件*标记 (token)，表示为 $\emptyset$。这种替换就像一种“条件Dropout”形式。
5. 将噪声样本 $x_t$、时间步长 $t$ 和（可能已替换的）条件信息输入到U-Net模型中：$\epsilon_{pred} = \epsilon_\theta(x_t, t, y_{eff})$，其中 $y_{eff}$ 为 $y$ 或 $\emptyset$。
6. 计算损失，通常是均方误差 (MSE)，在预测噪声 $\epsilon_{pred}$ 和真实噪声 $\epsilon$ 之间：$L = ||\epsilon - \epsilon_{pred}||^2$。
7. 使用此损失的梯度下降 (gradient descent)法更新模型参数 (parameter) $\theta$。

通过在训练期间随机省略条件信息，模型同时学习两件事：

- 如何在给定条件下预测噪声：$\epsilon_\theta(x_t, t, y)$。
- 如何无条件地预测噪声：$\epsilon_\theta(x_t, t, \emptyset)$。

空标记 $\emptyset$ 需要一个特定的表示。对于类别标签，它可能是一个专用的“无条件”类别索引。对于文本嵌入（如CLIP嵌入），它通常是一个固定的、可学习的嵌入向量 (vector)，与模型一同训练，以表示文本条件的缺失。

### 带有引导的采样

在生成（采样）过程中，我们借助模型的这种能力，在每一步进行有条件和无条件的预测。对于给定的时间步长 $t$ 和当前的噪声样本 $x_t$，我们计算*两个*噪声预测：

1. **无条件预测：** $\epsilon_{uncond} = \epsilon_\theta(x_t, t, \emptyset)$
2. **有条件预测：** $\epsilon_{cond} = \epsilon_\theta(x_t, t, y)$ (其中 $y$ 是输出所需的条件)

我们不只是使用 $\epsilon_{cond}$ 来执行去噪步骤，而是使用一个引导比例参数 (parameter) $w$（通常称为引导强度或比例，有时表示为 $s$ 或 $\gamma$）来结合这两个预测。组合噪声预测 $\tilde{\epsilon}_t$ 计算如下：


$$
\tilde{\epsilon}_t = \epsilon_{uncond} + w \cdot (\epsilon_{cond} - \epsilon_{uncond})
$$


这个公式有清晰的解释：

- 从无条件噪声预测 $\epsilon_{uncond}$ 开始。
- 计算“引导方向”：有条件预测和无条件预测之间的差值 $(\epsilon_{cond} - \epsilon_{uncond})$。这个向量 (vector)从无条件生成路径指向有条件生成路径。
- 按引导比例 $w$ 缩放此方向，并将其加到无条件预测中。

另一种等价的写法是：


$$
\tilde{\epsilon}_t = (1-w)\epsilon_{uncond} + w \cdot \epsilon_{cond}
$$


从这种形式中，我们可以看出：

- 如果 $w = 0$，我们得到 $\tilde{\epsilon}_t = \epsilon_{uncond}$，导致纯粹的无条件生成。
- 如果 $w = 1$，我们得到 $\tilde{\epsilon}_t = \epsilon_{cond}$，这对应于没有额外引导增强的标准有条件生成。
- 如果 $w > 1$，我们沿引导方向进行外推，更强烈地将生成推向条件 $y$。

这个组合噪声估计 $\tilde{\epsilon}_t$ 然后用于标准的去噪步骤（例如，DDPM 或 DDIM 更新规则）来计算噪声更少的样本 $x_{t-1}$。该过程从 $t=T$ 重复到 $t=1$。

下图说明了在单个采样步骤中的计算：

> 流程图用于计算在单个去噪步骤 $t$ 中使用分类器无关引导的引导噪声预测 $\tilde{\epsilon}_t$。

### 实际考虑

- **引导比例 ($w$)：** 这是一个重要的超参数 (parameter) (hyperparameter)。典型值范围为 $1.5$ 到 $15$。较高的值会更强地强制执行条件，这可以提升对齐 (alignment)效果（例如，使图像看起来更像文本提示），但可能导致饱和、伪影或生成样本多样性降低。较低的值会产生更多样但可能与条件对齐度较低的结果。您通常需要通过实验为您的特定任务和模型找到一个好的平衡点。
- **条件输入：** $y$ 和 $\emptyset$ 如何集成到U-Net架构中很重要。常见方法包括将它们的嵌入 (embedding)添加到时间步嵌入，或在U-Net块中使用交叉注意力层，使模型能够关注条件信息的关键部分。我们将在稍后更详细地讨论架构修改。
- **计算成本：** CFG 要求在每个采样步骤中运行模型两次前向传播（一次有条件，一次无条件）。这使得采样计算成本大致翻倍，相比于标准的有条件或无条件生成。考虑到在控制和质量方面的大幅改进，这种权衡通常是可以接受的。

实现CFG涉及修改训练循环（以处理条件dropout）和采样循环（以执行两次前向传播并组合结果）。接下来的章节将阐述常用于条件化的特定架构变化，并提供实际例子。

## 参考资料

- [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487) — Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, Mohammad Norouzi (2022)
  Journal: arXiv preprint arXiv:2205.11487; DOI: [10.48550/arXiv.2205.11487](https://doi.org/10.48550/arXiv.2205.11487)
  将分类器自由引导 (CFG) 正式化，并展示其在实现高质量、文本对齐图像生成方面的有效性，提供了现今使用的标准公式。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems 33 (NeurIPS 2020); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  介绍了去噪扩散概率模型 (DDPMs) 的论文，为应用 CFG 的扩散模型提供了理论和实践依据。
