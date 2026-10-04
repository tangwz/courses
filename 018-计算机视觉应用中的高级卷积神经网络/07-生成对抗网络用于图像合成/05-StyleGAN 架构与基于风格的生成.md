# StyleGAN 架构与基于风格的生成

来源：[原文](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-7-gans-image-synthesis/stylegan-architecture-generation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

早期的 GAN 架构，如 DCGAN，已能生成令人印象深刻的图像，但控制生成输出的特定属性或*风格*仍有难度。条件 GAN (cGAN) 允许基于离散标签或其他输入进行条件化生成，但在控制类别内更精细的方面（如姿态、纹理或光照变化）时，控制力有限。由 NVIDIA 研究人员提出的 StyleGAN，在生成建模方面取得了重大进展，其专门侧重于实现图像合成的直观、尺度特定控制，并提升生成结果的质量和解耦程度。

### StyleGAN 架构：改进

StyleGAN 与传统生成器架构在几个重要方面有所不同。它没有将潜在编码 $z$ 直接输入生成器的卷积堆栈，而是引入了中间步骤和修改，旨在更好地控制风格和分离特征。

#### 1. 映射网络与中间潜在空间 $W$

该过程始于标准潜在编码 $z$，通常从高斯分布 $Z$ 中采样获得。然而，$z$ 并不直接由主合成网络使用。相反，它首先由一个非线性的**映射网络** $f$ 进行变换，该网络通常实现为多层感知器 (MLP)。

$w = f(z)$

该网络将输入潜在编码 $z \in Z$ 映射到中间潜在空间 $w \in W$。这里的核心思想是，$z$ 的分布不必遵循训练数据变化所产生的分布。训练数据的变异因素通常是纠缠不清的。映射网络 $f$ 被训练用于“解缠” $z$，使其成为一种表示 $w$，其中各个分量可能更好地对应语义变异因素，从而可能产生一个解耦程度更高的潜在空间 $W$。这种解耦是有利的，因为与直接操作 $z$ 相比，在 $W$ 中操作方向可以导致最终图像中更局部化和可解释的变化。

#### 2. 带有风格调制功能的合成网络 $g$

**合成网络** $g$ 负责核心图像生成。与将 $z$ 作为第一层输入的传统生成器不同，StyleGAN 的合成网络 $g$ 从一个学习到的常数张量开始。风格信息，编码在中间潜在向量 (vector) $w$ 中，被注入到网络的多个点。

这种注入机制是 StyleGAN 风格控制的根本：**自适应实例归一化 (normalization) (AdaIN)**。回顾实例归一化 (IN)，它按通道和按样本对特征图进行归一化处理：

$\text{IN}(x) = \gamma \frac{x - \mu(x)}{\sigma(x)} + \beta$

其中 $\mu(x)$ 和 $\sigma(x)$ 是对每个通道和每个样本独立地在空间维度上计算得到的均值和标准差，而 $\gamma$ 和 $\beta$ 是可学习的缩放和偏置 (bias)参数 (parameter)。

AdaIN 通过使缩放 ($\gamma$) 和偏置 ($\beta$) 参数成为风格向量 $w$ 的函数来改进这一点。对于注入风格的每个层 $i$，AdaIN 首先归一化激活 $x_i$，然后通过学习到的仿射变换 ($A_i$) 应用从 $w$ 导出的缩放 $y_{s,i}$ 和偏置 $y_{b,i}$：

$\text{AdaIN}(x_i, w) = y_{s,i} \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$
$\text{其中 } (y_{s,i}, y_{b,i}) = A_i(w)$

通过在合成网络中的每个卷积层（或块）之后应用 AdaIN，$w$ 能有效地在不同抽象层次上控制图像的风格。从 $w$ 导出的、用于早期层（低分辨率）的风格，倾向于控制粗略属性，如姿态、脸型或发型；而用于后期层（高分辨率）的风格，则影响更精细的细节，如纹理、具体光照或眼睛颜色。

#### 3. 通过噪声输入引入随机变化

真实图像包含许多随机细节（例如，确切的发丝位置、雀斑、皱纹），这些细节仅通过全局风格向量 $w$ 难以捕捉。为了对这些细节进行建模，StyleGAN 引入了显式噪声输入。从简单高斯分布中采样的缩放噪声，在每次 AdaIN 操作之后、后续卷积之前，直接添加到特征图中。

重要的一点是，这种噪声是逐像素独立应用的。网络会学习这种噪声的缩放因子，从而能够控制不同特征层面随机效应的大小。这种分离作用显著：$w$ 控制整体风格，而噪声输入处理精细、非确定性的细节，使生成的图像看起来更自然。

> StyleGAN 生成器的简化流程图。潜在编码 $z$ 映射到 $w$，然后 $w$ 通过 AdaIN 调制卷积块。独立的噪声在每个分辨率层级添加。

### 基于风格的控制机制

该架构实现了强大的控制技术：

- **风格混合：** 在训练期间，一部分图像使用两种不同的中间潜在编码 $w_1$ 和 $w_2$ 生成。网络使用 $w_1$ 控制到特定层（例如，粗粒度层 4x4 到 8x8）的风格，并使用 $w_2$ 控制剩余层（例如，中/细粒度层 16x16 到 1024x1024）的风格。这种正则化 (regularization)技术可以防止网络假设相邻风格相关联，并鼓励风格控制局限于特定层。它还允许在推理 (inference)时通过显式混合来自不同源图像的风格来生成有趣的组合。
- **$W$ 空间中的截断技巧：** 虽然 GAN 理论上可以模拟整个数据分布，但潜在空间的极端区域通常对应着质量较低或不典型的图像。“截断技巧”通过像往常一样采样 $w$，然后将其移近平均中间潜在向量 (vector) $\bar{w}$（通过大量样本计算得出）来解决这个问题：$w' = \bar{w} + \psi (w - \bar{w})$，其中 $\psi \in [0, 1]$ 是一个截断因子。当 $\psi < 1$ 时，会提高平均图像质量和“正常性”，但代价是多样性/变异性降低。在 $W$ 空间中进行操作，使得这种截断比在初始 $Z$ 空间中截断更有效，且更不容易产生伪影。

### 演进：StyleGAN2

StyleGAN 影响很大，但也存在一些典型的图像伪影，例如水滴状图案。**StyleGAN2** 引入了几项架构改进来解决这些问题：

1. **权重 (weight)解调：** AdaIN 被认为是伪影的来源，因为它的实例归一化 (normalization)步骤可能会破坏特征图幅值中编码的信息。StyleGAN2 用权重解调取代了 AdaIN，在这种方法中，卷积权重根据传入的风格进行缩放，并结合标准归一化。
2. **修正的渐进式增长：** 原始的渐进式增长（训练期间逐步添加层）有时会导致“相位”伪影。StyleGAN2 采用受 MSG-GAN 启发的新型网络设计（跳跃连接、残差块），这些设计在不改变训练期间拓扑结构的情况下，实现了类似的多分辨率优势。
3. **路径长度正则化 (regularization)：** 鼓励从 $W$ 到图像空间的映射更平滑，从而改进条件化效果和图像质量。

StyleGAN 及其后续版本在高清、可控图像合成方面取得了巨大进步，在创意工具、数据增强以及生成模型的基础研究中都有应用。了解其架构原理，特别是通过映射网络实现职责分离、基于 AdaIN 的风格调制以及显式噪声输入，对从事高级计算机视觉生成模型工作的人员来说非常重要。

## 参考资料

- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) — Tero Karras, Samuli Laine, Timo Aila (2019)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 4401-4410; DOI: [10.1109/CVPR.2019.00673](https://doi.org/10.1109/CVPR.2019.00673)
  这篇基础论文介绍了 StyleGAN 架构，包括映射网络、基于 AdaIN 的风格调制和噪声输入。
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://doi.org/10.48550/arXiv.1710.10196) — Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.10196](https://doi.org/10.48550/arXiv.1710.10196)
  引入了用于训练 GAN 的渐进式增长方法，StyleGAN 采纳了这项技术并在 StyleGAN2 中进行了改进。
- [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) — Xun Huang, Serge Belongie (2017)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Pages: 1510-1519; DOI: [10.48550/arXiv.1703.06868](https://doi.org/10.48550/arXiv.1703.06868)
  提出了自适应实例归一化 (AdaIN) 的原始论文，这是 StyleGAN 中风格调制的组成部分。

---

[上一节](04-%E6%9D%A1%E4%BB%B6GANs%E7%94%A8%E4%BA%8E%E5%8F%AF%E6%8E%A7%E7%94%9F%E6%88%90.md) · [下一节](06-GAN%20%E7%9A%84%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.md)
