---
course: "synthetic-data-gans-diffusion"
chapter: "gan-training-stability-optimization"
lesson: "gan-regularization-techniques"
sourceId: 5677
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-3-gan-training-stability-optimization/gan-regularization-techniques"
title: "GAN 的正则化方法"
description: "应用谱归一化、一致性正则化及其他方法。"
order: 4
plots: []
sourceHash: "8ce4b3abece0d4539bda4dac0c8687b65ac3ba9c2f98887e73b4b7bf6571afec"
sourceCorrections: []
---

正则化 (regularization)方法是改进生成对抗网络 (GAN) 训练稳定性、动态并防止常见失败的重要工具。GAN 中的正则化主要目的是控制判别器的行为，防止它过快变得过于强大或对训练数据过拟合 (overfitting)，这可能导致生成器出现梯度消失或模式崩溃。有效的正则化策略在此审视。

## 谱归一化 (normalization)

谱归一化是 GAN 中应用最广且最有效的正则化 (regularization)方法之一。该方法由 Miyato 等人（2018）提出，通过限制判别器每个层的 Lipschitz 常数来稳定其训练。

**核心思路：** 回顾一下，如果对于任意输入 $x_1, x_2$，函数 $f$ 满足不等式 $||f(x_1) - f(x_2)|| \le K ||x_1 - x_2||$，则称函数 $f$ 为 $K$-Lipschitz 连续。较小的 Lipschitz 常数限制了函数输出随输入微小变化而发生剧烈改变的程度。在 GAN 中，一个 Lipschitz 常数过大且能力过强的判别器会产生过大或不稳定的梯度，从而阻碍生成器的训练。

谱归一化通过根据判别器网络中每个层的权重 (weight)矩阵 $\mathbf{W}$ 的谱范数 $\sigma(\mathbf{W})$ 进行归一化来控制 Lipschitz 常数。谱范数是矩阵 $\mathbf{W}$ 的最大奇异值，它对应于矩阵能够拉伸输入向量 (vector)的最大因子。

归一化方法如下：


$$
\mathbf{W}_{SN} = \frac{\mathbf{W}}{\sigma(\mathbf{W})}
$$


在每次前向传播之前，通过将权重除以它们的谱范数，我们确保归一化权重矩阵 $\mathbf{W}_{SN}$ 的谱范数精确为 1。此操作有效限制了每个层的 Lipschitz 常数，防止判别器的梯度爆炸，并带来更稳定的训练。

**实现：** 精确计算谱范数在计算上可能很耗时。实际中，通常使用幂迭代法进行有效近似。PyTorch 和 TensorFlow 等深度学习 (deep learning)框架提供了内置层或包装器（例如 `torch.nn.utils.spectral_norm`），可以透明地处理谱归一化的计算和应用。

> 谱归一化 (SN) 在判别器层中的应用。每个可学习层（卷积层、全连接层）的权重在使用前向传播之前，会除以其估计的谱范数。

在某些情况下，谱归一化常比梯度惩罚（来自 WGAN-GP）更受青睐，因为它计算成本较低，通常需要较少的超参数 (parameter) (hyperparameter)调整，同时仍能提供显著的稳定作用。它常用于 StyleGAN 等先进的 GAN 架构。

## 一致性正则化 (regularization)

一致性正则化 (CR) 促使判别器对应用于其输入的微小、保留语义的增强具有鲁棒性。其思路是，如果图像经过轻微增强（例如翻转、旋转、添加噪声），判别器对增强图像的输出应与其对原始图像的输出保持一致。

**工作原理：** CR 会在判别器的损失函数 (loss function)中添加一个惩罚项。该项衡量判别器对原始样本输出与增强版本输出之间的差异。该正则化适用于真实样本和生成样本：


$$
L_{CR} = \lambda_{CR} \left( \mathbb{E}_{\mathbf{x} \sim p_{data}} [ ||D(aug(\mathbf{x})) - D(\mathbf{x})||^2 ] + \mathbb{E}_{\mathbf{z} \sim p_z} [ ||D(aug(G(\mathbf{z}))) - D(G(\mathbf{z}))||^2 ] \right)
$$


这里，$aug(\cdot)$ 表示一个随机增强函数（或随机应用的一组固定增强），而 $\lambda_{CR}$ 是控制正则化强度的超参数 (parameter) (hyperparameter)。常见的增强包括随机翻转、旋转、平移、缩放、裁剪、噪声注入或色彩抖动。

**为何有效：** 通过在增强下强制保持一致性，CR 作为一种强大的数据增强策略，*尤其针对判别器*。这可以防止判别器简单地记忆训练集，并促使其学习更具泛化能力的特征。更稳定的判别器为生成器提供更有意义的梯度，从而提升整体训练效果和样本质量。CR 在数据有限的情况下被证实特别有效。

## 梯度惩罚再审视

如在 WGAN-GP 的语境中所述，梯度惩罚项本身是一种正则化 (regularization)形式。其特定目标是对判别器强制执行 1-Lipschitz 约束，这是 Wasserstein 距离近似的核心所在。

惩罚项通常表示为：


$$
L_{GP} = \lambda_{GP} \mathbb{E}_{\hat{\mathbf{x}} \sim p_{\hat{x}}} [ (||\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})||_2 - 1)^2 ]
$$


其中 $\hat{\mathbf{x}}$ 是沿着连接真实样本对（$\mathbf{x} \sim p_{data}$）和生成样本（$G(\mathbf{z}), \mathbf{z} \sim p_z$）的直线采样的点，且 $\lambda_{GP}$ 是惩罚系数。

尽管有效，但计算梯度惩罚需要额外进行一次反向传播 (backpropagation)，以计算判别器输出相对于其输入的梯度（$\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})$），这使其计算成本高于谱归一化 (normalization)等方法。

## 其他正则化 (regularization)方法

尽管谱归一化 (normalization)和一致性正则化很受关注，但其他标准的深度学习 (deep learning)正则化方法有时也可以应用于 GANs，不过效果常常不一：

- **Dropout：** 在判别器或生成器中使用 dropout 有时会有帮助，但它也可能干扰 GAN 训练的细微平衡。应谨慎使用，可能需采用比标准分类任务更低的 dropout 率。
- **权重 (weight)衰减 (L2 正则化)：** 对判别器或生成器的权重添加 L2 惩罚是可行的。然而，与专门控制判别器 Lipschitz 性质或梯度行为的方法相比，它在稳定 GAN 方面通常效果不佳。
- **实例噪声：** 直接向判别器的输入（真实和伪造样本）添加少量噪声有时可以平滑决策边界，并防止判别器过于自信。

## 正则化 (regularization)的组合与调整

正则化方法并非互斥，常能有效组合使用。例如，在判别器中使用谱归一化 (normalization)并结合 WGAN-GP 损失（有效结合了 SN 和梯度惩罚，尽管 SN 常使得梯度惩罚不那么必要），或者在谱归一化的判别器上应用一致性正则化，这些都是常见的做法。

正则化方法的选择和组合取决于具体的 GAN 架构、数据集以及观察到的训练问题。应用正则化会引入需要仔细调整的超参数 (parameter) (hyperparameter)（如 $\lambda_{CR}$、$\lambda_{GP}$）。正则化的强度必须保持平衡；过少可能无法解决稳定性问题，而过多则可能过度限制判别器，减缓学习速度或影响性能。应监测训练动态（损失曲线、梯度范数）和样本质量指标（如 FID）以指导调整过程。

正则化提供了一套管理 GAN 训练复杂性的重要工具。通过仔细控制判别器的行为，这些技术显著增加了实现稳定收敛和生成高质量合成数据的可能性。

## 参考资料

- [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957) — Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1802.05957](https://doi.org/10.48550/arXiv.1802.05957)
  介绍了谱范数归一化，通过约束判别器层中的Lipschitz常数来稳定GAN训练，使其成为一种标准正则化技术。
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) — Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1704.00028](https://doi.org/10.48550/arXiv.1704.00028)
  提出了梯度惩罚方法，以强制Wasserstein GANs中的Lipschitz约束，显著提高了训练稳定性和样本质量。
- [Differentiable Augmentation for Data-Efficient GAN Training](https://arxiv.org/abs/2006.10738) — Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, Song Han (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 33; DOI: [10.48550/arXiv.2006.10738](https://doi.org/10.48550/arXiv.2006.10738)
  提出了可微分增强，在GAN训练中对真实和虚假图像应用增强，作为一种一致性正则化，以提高稳定性和数据效率。
