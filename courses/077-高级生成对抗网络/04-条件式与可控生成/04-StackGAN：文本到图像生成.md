# StackGAN：文本到图像生成

来源：[原文](https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-4-conditional-controllable-generation/stackgan-text-to-image)

[返回章节目录](README.md) · [返回课程目录](../README.md)

生成与详细文本描述相符的图像，是实现更直观的人机交互和创意内容制作的重要进展。尽管基础条件GAN（cGAN）能依据简单标签（如类别）生成图像，但从复杂的自由格式文本描述中生成高分辨率、照片级真实感的图像则是一个更大的难题。文本提供语义信息，但缺少生成高保真图像所需的像素级细节。一次性生成所有内容通常会导致结果模糊，或图像与描述仅有模糊的匹配。

StackGAN 采用分层方法应对此挑战，将复杂的文本到图像问题拆解成更易处理的子问题，由堆叠式生成网络处理。

### 两阶段架构

StackGAN 的核心思想是采用两阶段生成过程：

1. **第一阶段GAN：** 该网络专注于绘制图像的初步低分辨率版本。它接收文本嵌入 (embedding)（使用预训练 (pre-training)的文本编码器，如字符级CNN-RNN或BERT从输入描述中获取）和随机噪声向量 (vector) $z$ 作为输入。其目标是生成一幅捕捉文本中描述的大致布局、颜色和基本形状的图像，同时忽略精细细节，并侧重于语义连贯性。输出通常是低分辨率图像（例如，64x64像素）。
2. **第二阶段GAN：** 该网络接收第一阶段生成的低分辨率图像*以及*原始文本嵌入作为输入。其目标是优化初始草图、修正缺陷、增加细节，并显著提升图像分辨率（例如，到256x256像素）。通过同时以第一阶段图像和文本嵌入为条件，第二阶段可以专注于添加与初始结构和文本描述一致的真实感和细节。

这种分阶段方法使得模型能首先侧重于全局结构和语义（第一阶段），然后集中于像素级优化和细节（第二阶段）。

> StackGAN架构概述。文本经编码和增强后，与噪声 $z$ 一同在第一阶段中使用，以生成低分辨率草图。第二阶段接收此草图和文本嵌入，生成优化后的高分辨率图像。两个阶段都涉及对抗训练。

### 条件增强

条件生成中的一个潜在问题是文本嵌入 (embedding)空间可能稀疏或不够平滑。如果相似描述的嵌入相距甚远，生成器可能难以泛化。StackGAN 引入条件增强（CA）以应对此问题。

CA 不直接使用编码器生成的文本嵌入 $\phi_t$，而是将嵌入建模为高斯分布 $N(\mu(\phi_t), \Sigma(\phi_t))$，其中均值 $\mu(\phi_t)$ 和对角协方差矩阵 $\Sigma(\phi_t)$ 是原始嵌入 $\phi_t$ 的函数（通常通过小型全连接层实现）。在训练期间，从该分布中采样一个条件变量 $\hat{c}$：


$$
\hat{c} \sim N(\mu(\phi_t), \Sigma(\phi_t))
$$


采样的 $\hat{c}$ 随后用作第一阶段生成器 $G_1(z, \hat{c})$ 的条件。这种采样过程有助于增强对文本嵌入微小变化的鲁棒性，并创建更平滑的条件流形，使生成器的学习任务更简单。一个正则化 (regularization)项，即学习到的分布 $N(\mu(\phi_t), \Sigma(\phi_t))$ 与标准高斯分布 $N(0, I)$ 之间的 Kullback-Leibler (KL) 散度，被添加到生成器的目标函数中，以防止方差崩溃：


$$
L_{CA} = D_{KL}(N(\mu(\phi_t), \Sigma(\phi_t)) || N(0, I))
$$


第二阶段GAN通常直接使用原始文本嵌入 $\phi_t$ 进行条件化，因为它主要侧重于根据第一阶段已确定的结构和文本的语义指导添加细节。

### 训练与损失函数 (loss function)

训练 StackGAN 涉及首先训练第一阶段，然后使用已训练的第一阶段生成器的输出来训练第二阶段。

- **第一阶段：** 生成器 $G_1$ 和判别器 $D_1$ 以对抗方式训练。$G_1$ 尝试生成与采样条件 $\hat{c}$ 匹配的真实感低分辨率图像，并包含 $L_{CA}$ 正则化 (regularization)。$D_1$ 尝试区分与相应文本条件 $\hat{c}$ 配对的真实低分辨率图像（从数据集中下采样），以及与 $\hat{c}$ 配对的来自 $G_1$ 的虚假低分辨率图像。
- **第二阶段：** 生成器 $G_2$ 和判别器 $D_2$ 以对抗方式训练。$G_2$ 接收低分辨率输出 $s_0 = G_1(z, \hat{c})$ 和原始文本嵌入 (embedding) $\phi_t$ 作为输入，目标是生成真实感高分辨率图像 $s_1 = G_2(s_0, \phi_t)$。判别器 $D_2$ 学会区分与文本嵌入 $\phi_t$ 配对的真实高分辨率图像，以及由 $G_2$ 生成、同样以 $\phi_t$ 为条件的虚假高分辨率图像 $s_1$。

两个阶段都使用结合条件信息的标准GAN损失函数（或Wasserstein损失等变体）。要点在于，第二阶段的判别器不仅要评估高分辨率图像的真实感，还要评估其与文本嵌入 $\phi_t$ 的一致性。

相较于之前的单阶段方法，StackGAN 在从文本描述生成高分辨率图像（例如，256x256）方面展现了显著提升。分层优化策略被证明能有效处理将抽象文本映射到详细像素的复杂性。后续工作，如 StackGAN++，通过改进的损失函数和训练稳定性技术，进一步完善了此方法。

## 参考资料

- [StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_StackGAN_Text_to_ICCV_2017_paper.pdf) — Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris N. Metaxas (2017)
  Journal: 2017 IEEE International Conference on Computer Vision; Publisher: IEEE; Pages: 118–126; DOI: [10.1109/ICCV.2017.394](https://doi.org/10.1109/ICCV.2017.394)
  介绍了用于从文本描述生成高分辨率图像的原始两阶段StackGAN架构，并引入了条件增强。
- [StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1710.05465.pdf) — Han Zhang, Tao Xu, Hongxu Ma, Zhiming Liu, Gang Huang, Shaoting Zhang, and Dimitris Metaxas (2018)
  Journal: International Conference on Learning Representations (ICLR); Publisher: OpenReview.net; DOI: [10.48550/arXiv.1710.05465](https://doi.org/10.48550/arXiv.1710.05465)
  改进了原始StackGAN的训练稳定性、损失函数和生成器架构，以提升文本到图像的合成效果。
- [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf) — Mehdi Mirza and Simon Osindero (2014)
  Journal: arXiv preprint arXiv:1411.1784; DOI: [10.48550/arXiv.1411.1784](https://doi.org/10.48550/arXiv.1411.1784)
  提出了条件GAN的概念，支持基于类标签等特定条件生成图像，是文本到图像生成的前身。
- [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 27; Pages: 2672-2680; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  本文介绍了生成对抗网络，它是StackGAN及其他高级生成模型的核心组件。

---

[上一节](03-%E4%BF%A1%E6%81%AF%E6%9C%80%E5%A4%A7%E5%8C%96GAN%20%28InfoGAN%29.md) · [下一节](05-%E9%80%9A%E8%BF%87%E6%BD%9C%E5%9C%A8%E7%A9%BA%E9%97%B4%E6%93%8D%E4%BD%9C%E6%8E%A7%E5%88%B6%E5%B1%9E%E6%80%A7.md)
