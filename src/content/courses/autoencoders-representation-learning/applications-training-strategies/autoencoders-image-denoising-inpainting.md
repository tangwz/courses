---
course: "autoencoders-representation-learning"
chapter: "applications-training-strategies"
lesson: "autoencoders-image-denoising-inpainting"
sourceId: 2923
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-7-applications-training-strategies/autoencoders-image-denoising-inpainting"
title: "图像去噪和修补应用"
description: "应用自动编码器，尤其是去噪自动编码器，进行图像恢复任务。"
order: 4
plots: []
sourceHash: "cd098af7c54c44d636f5025af2946731e0626de233309be54699412086189a70"
sourceCorrections: []
---

无论是发现异常还是降低维度，自动编码器的重建能力都非常适合图像恢复任务，例如去噪和修补。这些应用直接利用了自动编码器学习数据内在结构并分离信号与损坏的能力。具体来说，去噪自动编码器（DAE）等变体，为这些任务提供了扎实的基础。

### 自动编码器在图像去噪中的应用

图像去噪的目标是从损坏的版本 $\tilde{x}$ 中恢复出清晰图像 $x$，其中 $\tilde{x}$ 可能是被噪声（例如高斯噪声、椒盐噪声）污染的 $x$。在干净数据上训练的标准自动编码器，当在测试时遇到带噪声的输入时可能会遇到困难，因为噪声可能会使输入分布显著偏离编码器所学的内容。

去噪自动编码器（DAEs）直接处理这个问题。核心思路简单而有效：训练自动编码器，使其即使在输入是*损坏的*版本 $\tilde{x}$ 时，也能重建*原始、清晰的*图像 $x$。

训练过程包括：

1. **随机损坏：** 从训练集中取一个干净图像 $x$。应用随机损坏过程以获得 $\tilde{x}$。常见的损坏过程包括：
   - 添加零均值高斯噪声：$\tilde{x} = x + \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$。
   - 椒盐噪声：随机将部分像素设置为其最小值（0）或最大值（1或255）。
   - 遮蔽噪声：随机将部分输入元素设置为零。
2. **编码：** 将损坏的图像 $\tilde{x}$ 通过编码器 $f_\theta$ 得到潜在表示 $z = f_\theta(\tilde{x})$。
3. **解码：** 将潜在表示 $z$ 通过解码器 $g_\phi$ 得到重建图像 $x' = g_\phi(z)$。
4. **损失计算：** 计算*原始清晰*图像 $x$ 与重建图像 $x'$ 之间的重建损失 $L(x, x')$。图像数据的常见选择包括均方误差（MSE）或二元交叉熵（BCE），如果像素值归一化 (normalization)到 $[0, 1]$。

目标函数通常是数据分布和损坏过程上的预期重建误差：


$$
\mathcal{L}(\theta, \phi) = \mathbb{E}_{x \sim p_{\text{数据}}(x)} \left[ \mathbb{E}_{\tilde{x} \sim p(\tilde{x}|x)} [L(x, g_\phi(f_\theta(\tilde{x})))] \right]
$$


通过最小化此损失，DAE 学习隐式地逆转损坏过程。它必须捕获清晰图像的基本结构和统计规律（数据流形），才能有效地忽略噪声并重建原始内容。这使得编码器学习到对训练过程中引入的特定类型噪声不变的特征。对于图像数据，通常使用卷积自动编码器（第5章讨论过）作为DAE，它们利用卷积层更好地处理空间层级和局部性。

### 自动编码器在图像修补中的应用

图像修补是填充图像中缺失或损坏区域的任务。这可以被看作是一种特定形式的去噪，其中“噪声”是表示缺失部分的遮罩。自动编码器可以通过训练来预测基于周围上下文 (context)的缺失像素值，从而有效地用于此任务。

设置与去噪类似：

1. **遮蔽：** 取一个原始图像 $x$。通过将待修补区域的像素设置为特定值（例如零、灰色或平均像素值）来创建遮蔽版本 $\tilde{x}$。这可以通过使用预定义遮罩（例如中心方块、随机块）或从实际图像损坏中得出的遮罩来完成。
2. **编码与解码：** 将遮蔽图像 $\tilde{x}$ 通过编码器-解码器架构（$x' = g_\phi(f_\theta(\tilde{x}))$）。
3. **损失计算：** 计算*原始完整*图像 $x$ 与重建图像 $x'$ 之间的重建损失 $L(x, x')$。损失可以在整个图像上计算，或者有时更有效的是，仅在遮蔽区域内的像素上计算。

自动编码器通过利用训练集中完整图像学习到的模式和结构，学习推断缺失区域的内容。编码器将不完整的输入映射到捕获图像主要信息的潜在表示，解码器利用此表示及其学习到的典型图像结构知识，生成缺失像素的合理值。

同样，卷积自动编码器通常是修补任务的优选，因为它们能有效地处理空间上下文。修补的质量在很大程度上取决于缺失区域的大小和复杂性，以及自动编码器学习图像代表性特征的能力。虽然基本自动编码器在小的、简单的缺失部分上表现良好，但填充大或结构复杂的区域通常需要更高级的生成模型，有时会包含对抗性组件，类似于对抗性自动编码器（AAEs）或生成对抗网络 (GAN)（GANs）中存在的那些，以提升修补区域的真实感。

### 实际考量

- **架构：** 对于去噪和修补任务，卷积层几乎总是比全连接层更受青睐，因为它们在处理图像空间信息方面更有效。U-Net 类似的架构，在对应的编码器和解码器层之间带有跳跃连接，也很受欢迎，因为它们有助于在重建过程中保持精细的细节。
- **损失函数 (loss function)：** 虽然MSE很常见，但它有时可能导致模糊的重建结果，尤其是在修补中。L1损失可能会产生更清晰的结果。感知损失函数（衡量预训练 (pre-training)网络（如VGG）提取的高级特征空间中的差异）或对抗性损失可以被纳入，以促使生成更具视觉真实感的结果。
- **损坏策略：** 训练期间使用的噪声或遮蔽的类型和强度应与测试时预期的损坏理想匹配。使用多种损坏类型进行训练可以提高模型的鲁棒性。

总之，学习高效数据表示的基本原理使得自动编码器，特别是DAEs和卷积自动编码器，成为图像去噪和修补的重要工具。它们学习将图像的固有结构与各种形式的损坏分离，从而实现有效的恢复。

## 参考资料

- [Extracting and composing robust features with denoising autoencoders](https://dl.acm.org/doi/abs/10.1145/1390156.1390294) — Pascal Vincent, Hugo Larochelle, Yoshua Bengio, Pierre-Antoine Manzagol (2008)
  Journal: Proceedings of the 25th International Conference on Machine Learning; Publisher: ACM; Pages: 1096-1103; DOI: [10.1145/1390156.1390294](https://doi.org/10.1145/1390156.1390294)
  介绍了去噪自编码器（DAE），并展示了其通过从损坏版本重建原始输入来学习稳健表示的有效性。
- [U-Net: Convolutional networks for biomedical image segmentation](https://arxiv.org/abs/1505.04597) — Olaf Ronneberger, Philipp Fischer, Thomas Brox (2015)
  Journal: MICCAI 2015; Volume: 9351; Pages: 234-241; DOI: [10.48550/arXiv.1505.04597](https://doi.org/10.48550/arXiv.1505.04597)
  提出了U-Net架构，因其U型设计和跳跃连接而被广泛应用于包括图像恢复在内的各种图像到图像任务。
- [Context encoders: Feature learning by inpainting](https://arxiv.org/abs/1604.07379) — Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros (2016)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 2536-2544; DOI: [10.48550/arXiv.1604.07379](https://doi.org/10.48550/arXiv.1604.07379)
  介绍了用于图像修复的上下文编码器，结合了编码器-解码器架构和对抗性损失，以生成语义上合理且逼真的缺失区域。
- [Perceptual losses for real-time style transfer and super-resolution](https://arxiv.org/abs/1603.08155) — Justin Johnson, Alexandre Alahi, Li Fei-Fei (2016)
  Journal: European Conference on Computer Vision (ECCV); Volume: 9906; Pages: 694-711; DOI: [10.48550/arXiv.1603.08155](https://doi.org/10.48550/arXiv.1603.08155)
  提出了使用感知损失函数，该函数源自预训练深度网络的高级特征图，以在图像生成任务中获得视觉上令人满意的结果，并适用于图像恢复。
