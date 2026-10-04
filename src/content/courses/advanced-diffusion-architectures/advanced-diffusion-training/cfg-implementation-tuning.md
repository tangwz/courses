---
course: "advanced-diffusion-architectures"
chapter: "advanced-diffusion-training"
lesson: "cfg-implementation-tuning"
sourceId: 5428
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-4-advanced-diffusion-training/cfg-implementation-tuning"
title: "实施与调整CFG尺度"
description: "CFG的实际操作和调整引导尺度的技术，以获得最佳结果。"
order: 3
plots: ["plots/5428-0.json"]
sourceHash: "db08388acdb6fdf701da1ceb2d7b4874e73c3758433695c5d1ef914da37e2e19"
sourceCorrections: []
---

无分类器引导（CFG）是一种控制扩散模型生成的方法。本讨论专注于其实际操作，以及调整引导尺度（通常表示为 $s$ 或 $w$）这一过程。该尺度决定了生成过程对所提供条件信号（如文本提示或类别标签）的依从程度。

### 实施机制

CFG通过巧妙地修改模型在采样过程中预测的噪声（或预测的 $x_0$，取决于参数 (parameter)化方式）来工作。在每个时间步 $t$，模型会做出两次预测：

1. **条件预测：** $\epsilon_\theta(x_t, t, c)$，其中 $c$ 是条件信息（例如，文本嵌入 (embedding)、类别ID）。
2. **无条件预测：** $\epsilon_\theta(x_t, t, \emptyset)$，其中 $\emptyset$ 代表空或缺失条件。

在训练期间，这种无条件预测通常是通过随机丢弃部分训练样本的条件信息来实现的（例如，将文本嵌入替换为学到的空标记 (token)，或将类别标签替换为特殊的“无条件”ID）。这使得模型能够在同一架构内，在有引导和无引导情况下都预测噪声。

在推理 (inference)时，模型会计算这两种预测。用于去噪步骤的最终噪声预测是一个线性组合，从无条件预测向条件预测的方向外推：


$$
\hat{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
$$


这里，$s$ 是引导尺度。请注意，如果 $s=0$，我们恢复无条件预测，即 $\hat{\epsilon}_\theta = \epsilon_\theta(x_t, t, \emptyset)$。如果 $s=1$，我们恢复标准条件预测，即 $\hat{\epsilon}_\theta = \epsilon_\theta(x_t, t, c)$。当 $s > 1$ 时，会将预测进一步推向条件所指示的方向，从而增强引导效果。

这种组合预测 $\hat{\epsilon}_\theta$ 随后被用于所选的采样器（例如 DDIM、DPM-Solver）中，以从 $x_t$ 估计 $x_{t-1}$。

### 引导尺度参数 (parameter)（$s$）

引导尺度 $s$ 是一个超参数 (hyperparameter)，它控制着样本质量（对条件的依从性）与多样性之间的权衡。

- **低 $s$（例如，0-3）：** 生成对条件的依赖较少。
  - $s=0$：纯粹的无条件生成，忽略 $c$。
  - $s=1$：标准条件生成（按学习所得）。
  - 略大于1：样本多样，可能更具创意或更抽象，但可能不会严格遵循提示或条件。
- **中等 $s$（例如，4-10）：** 一个常见范围，通常能产生良好平衡。样本通常很好地遵循条件，同时保持合理的多样性。许多标准文本到图像模型的默认值都在7-8左右。
- **高 $s$（例如，11-20+）：** 对条件有强烈的依从性。样本可以非常准确地表现提示，但可能出现多样性降低、潜在的饱和问题（例如，过亮或对比度过高的图像）或视觉伪影，因为模型被推向其所学分布中较少触及的区域。

$s$ 的影响高度依赖于特定的模型、数据集和任务。没有单一的“最优”值；它需要调整。

### 引导尺度调整策略

找到一个有效的引导尺度通常需要实验：

1. **视觉检查：** 使用相同的初始噪声和条件生成样本，但改变尺度 $s$。观察对条件的依从性如何变化，以及在较高尺度下是否出现伪影。这通常是图像生成等任务的主要方法。
2. **基于指标的评估：** 如果有定量指标可用（例如，用于文本到图像对齐 (alignment)的CLIP分数，用于类别条件生成的分类准确率），则在不同尺度下生成批量样本，并将指标绘制为 $s$ 的函数。这可以显示出趋势，但应与视觉检查结合使用，因为指标并非总能完美捕捉感知质量。
3. **权衡分析：** 考虑针对不同尺度绘制依从性指标与多样性指标（如果可用）的关系图。这有助于可视化随着 $s$ 增加而产生的权衡。



![CFG尺度（s）的影响](plots/5428-0.json)



> 引导尺度（$s$）与提示依从性、样本多样性以及伪影可能性之间的关系。增加 $s$ 通常能提高依从性，但会降低多样性并可能引入伪影。

### 代码示例片段

下面是CFG逻辑如何修改采样循环的简化示例。假设 `model` 可以预测噪声 $\epsilon_\theta$，`sampler` 处理去噪步骤，`latents` 是 $x_t$，`t` 是时间步，`cond_embedding` 是条件，`uncond_embedding` 是空条件。

```python
import torch

# 假设 model, sampler, latents, t, cond_embedding, uncond_embedding 已定义
# 假设 guidance_scale (s) 已设置，例如 s = 7.5

# 拼接输入以便进行批量推理
latent_model_input = torch.cat([latents] * 2)
time_input = torch.cat([t] * 2)
context_input = torch.cat([uncond_embedding, cond_embedding])

# 预测无条件和有条件输入的噪声
noise_pred_uncond, noise_pred_cond = model(latent_model_input, time_input, context=context_input).chunk(2)

# 使用CFG公式组合预测
noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

# 在采样器的步进函数中使用组合预测
latents = sampler.step(noise_pred_cfg, t, latents)

# ... 采样循环的其余部分 ...
```

### 实际考虑事项

- **计算成本：** CFG的主要缺点是在每个采样步骤都需要两次模型前向传播（一次条件，一次无条件）。与仅使用条件模型或未经CFG训练的模型相比，这大致使推理 (inference)时间加倍。优化这方面的研究正在进行。
- **与采样器的关系：** 最佳 $s$ 值可能会根据所使用的采样器（例如 DDIM vs. DPM-Solver++）和采样步数略有变化。使用较少步数的更快速采样器有时可能从略高的引导尺度中受益，以保持提示的忠实度。
- **训练稳定性：** 尽管CFG在推理期间应用，但模型必须通过条件 dropout 机制进行充分训练。如果模型未能很好地学习无条件预测，CFG的表现可能会不佳。后面会讨论的技术，例如EMA和梯度裁剪，有助于确保基础模型的稳定性。

掌握CFG尺度的实施和调整是控制现代扩散模型的基本技能。它提供了一个强大的调节手段，用于平衡对期望输出的忠实度与生成过程固有的创造性和多样性。在视觉反馈和相关指标的指导下进行实验，对于为您的特定应用找到最佳点是不可或缺的。

## 参考资料

- [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487) — Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, Mohammad Norouzi (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 35; Pages: 16457-16474; DOI: [10.48550/arXiv.2205.11487](https://doi.org/10.48550/arXiv.2205.11487)
  本文展示了无分类器引导在大型高保真文本到图像扩散模型架构中的成功集成，进一步强调了其重要性。
