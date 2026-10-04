---
course: "generative-adversarial-networks-gans"
chapter: "gan-training-stabilization"
lesson: "wgan-gradient-penalty"
sourceId: 2695
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-3-gan-training-stabilization/wgan-gradient-penalty"
title: "梯度惩罚 (WGAN-GP)"
description: "使用梯度惩罚作为一种更佳的实施Lipschitz约束的方法。"
order: 5
plots: ["plots/2695-0.json"]
sourceHash: "d4261a6ab281ebbacd3c729fb3650ca69960b0854e4c8266c6e7efd745523023"
sourceCorrections: []
---

尽管Wasserstein GAN (WGAN) 引入了基于地球移动距离的更稳定的损失函数 (loss function)，但其对判别器施加必要1-Lipschitz约束的原始方式——权重 (weight)剪裁——存在自身的一些问题。将权重剪裁到一个小范围（例如 $[-0.01, 0.01]$）可能导致优化上的困难：如果剪裁范围过小，梯度可能消失，阻碍训练；如果过大，梯度可能爆炸，引起不稳定。此外，权重剪裁会使判别器偏向学习过于简单的函数，可能降低其捕获真实数据分布复杂性的能力。

为解决这些局限，提出了一种更好的方法，称为梯度惩罚（WGAN-GP）。WGAN-GP没有粗暴地将权重强制限制在一个区间，而是直接惩罚判别器相对于输入的梯度范数，促使其保持接近1。这是一种更温和、更有针对性的实施Lipschitz约束的方式。

### 梯度惩罚项

主要思路是在判别器的损失函数 (loss function)中加入一个惩罚项。该惩罚旨在促使判别器梯度的L2范数（欧几里得范数）趋近于1，特别是对于在真实数据分布和生成数据分布*之间*采样的点。

在数学上，梯度惩罚项定义为：


$$
\lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}} [ (\| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1)^2 ]
$$


我们来逐一解释：

- $D(\hat{x})$: 判别器（评论器）对于输入样本 $\hat{x}$ 的输出。
- $\nabla_{\hat{x}} D(\hat{x})$: 判别器输出相对于其输入 $\hat{x}$ 的梯度。这表示判别器输出对输入变化的敏感度。
- $\| \cdot \|_2$: 梯度向量 (vector)的L2范数（欧几里得范数）。对于向量 $v = (v_1, v_2, ..., v_n)$，$\|v\|_2 = \sqrt{v_1^2 + v_2^2 + ... + v_n^2}$。
- $(\| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1)^2$: 这一项计算梯度范数与1之间的平方差。当范数恰好为1时，该项为零；当范数偏离1时，该项呈二次方增长。这惩罚了范数远离1的梯度。
- $\hat{x} \sim P_{\hat{x}}$: 这表示期望是针对从特定分布 $P_{\hat{x}}$ 中采样的 $\hat{x}$ 样本计算的。这些样本 $\hat{x}$ 是通过在真实样本 ($x \sim P_{data}$) 和生成样本 ($\tilde{x} \sim P_g$) 对之间进行插值得到的。具体来说：
  
  $$
  \hat{x} = \epsilon x + (1 - \epsilon) \tilde{x}
  $$
  
  其中 $\epsilon$ 是从 $[0, 1]$ 均匀采样的随机数。
- $\lambda$: 一个超参数 (parameter) (hyperparameter)，用于控制梯度惩罚相对于原始WGAN损失的权重 (weight)。常用值为 $\lambda = 10$。

下图说明了插值点 $\hat{x}$ 的采样过程：

> 真实样本 $x$ 与生成样本 $\tilde{x}$ 之间进行插值以创建 $\hat{x}$。梯度惩罚在这些插值点 $\hat{x}$ 处进行评估。

### 为何选择插值样本？

Wasserstein距离理论要求判别器（评论器）在*所有位置*都是1-Lipschitz的。在全局范围强制执行这一点在计算上很困难。WGAN-GP论文通过实验证明，仅沿着真实样本和生成样本之间的这些直线强制约束足以实现稳定训练。直观地看，这使得约束集中在当前与生成器训练过程相关的输入空间区域。

惩罚函数 $( \|g\|_2 - 1 )^2$ 促使梯度范数 $\|g\|_2$ 接近1，如下所示：



![梯度惩罚函数](plots/2695-0.json)



> 梯度惩罚 $( \|g\|_2 - 1 )^2$ 对应梯度范数 $\|g\|_2$ 的图示。当范数恰好为1时，惩罚项达到最小值（为零），这促使判别器满足此条件。

### WGAN-GP 目标函数

加入梯度惩罚后，判别器的目标是最大化 $L_D$：


$$
L_D = \mathbb{E}_{x \sim P_{data}}[D(x)] - \mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x})] - \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}} [ (\| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1)^2 ]
$$


请注意，在实际操作中我们最小化此损失的*负值*。生成器的目标与原始WGAN相同，旨在最小化 $L_G$（这相当于最大化判别器对假样本的评分）：


$$
L_G = - \mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x})]
$$


### 梯度惩罚的优势

使用梯度惩罚比权重 (weight)剪裁带来了多项重要的优势：

1. **训练更稳定：** 与权重剪裁相比，它通常会带来更稳定的训练收敛，避免了因剪裁参数 (parameter)选择不当而导致的特定失败模式。
2. **模型容量更高：** 通过不直接限制权重，判别器可以学习更复杂的函数，这可能带来对Wasserstein距离的更好近似以及更高质量的生成样本。
3. **无需为剪裁调整超参数 (hyperparameter)：** 消除了调整剪裁范围 $c$ 的需要，取而代之的是惩罚系数 $\lambda$，该系数通常不那么敏感（通常 $\lambda=10$ 效果良好）。

### 实现注意事项

- **计算开销：** 计算梯度惩罚需要计算梯度的梯度 ($\nabla_{\hat{x}} D(\hat{x})$)。这涉及在计算图中从 $\hat{x}$ 计算判别器输出的部分进行第二次反向传播 (backpropagation)，相较于标准GAN或带权重 (weight)剪裁的WGAN，增加了计算负担。大多数深度学习 (deep learning)框架都提供了高效计算这些高阶梯度的工具。
- **归一化 (normalization)：** 原始WGAN-GP论文建议在判别器中避免使用批归一化，因为它会在批次内的样本之间引入依赖关系，这可能干扰梯度惩罚的计算（梯度惩罚假设样本是独立的）。可以考虑使用层归一化、实例归一化等其他归一化方法，或者完全不使用归一化。
- **采样 $\hat{x}$：** 对于每个批次，你需要从 $\epsilon \sim U[0, 1]$ 采样（通常每对样本一个 $\epsilon$，有时是一个 $\epsilon$ 广播到整个批次），并计算插值样本 $\hat{x}$。

WGAN-GP是稳定GAN训练的一个重要进展。通过用一种有理论依据的梯度惩罚取代权重剪裁，它使得训练更深层、更复杂的GAN成为可能，能够生成更高质量的结果，同时减轻了困扰早期方法的许多优化问题。它已成为许多后续最先进GAN架构中使用的标准方法。

## 参考资料

- [Improved Training of Wasserstein GANs](https://doi.org/10.5555/3157382.3157640) — Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: Curran Associates, Inc.; Pages: 5767-5777; DOI: [10.5555/3157382.3157640](https://doi.org/10.5555/3157382.3157640)
  引入梯度惩罚机制以在WGAN中强制执行Lipschitz约束的开创性论文，相对于权重裁剪显著提高了训练稳定性。
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875) — Martin Arjovsky, Soumith Chintala, Léon Bottou (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.07875](https://doi.org/10.48550/arXiv.1701.07875)
  这篇论文介绍了WGAN和用于GAN训练的地球移动距离，指出了原始GAN损失的问题并提出了权重裁剪，WGAN-GP在此基础上进行了改进。
- [Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play](https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/) — David Foster (2019)
  Publisher: O'Reilly Media
  提供生成模型全面且实用的概述，包括WGAN和WGAN-GP的概念和实现细节。
- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — PyTorch Team (2024)
  Publisher: PyTorch
  理解PyTorch中自动微分工作原理的重要文档，包括梯度惩罚所需的高阶梯度。
