---
course: "generative-adversarial-networks-gans"
chapter: "advanced-gan-architectures"
lesson: "stylegan-architecture"
sourceId: 2668
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-2-advanced-gan-architectures/stylegan-architecture"
title: "基于风格的生成器架构 (StyleGAN)"
description: "了解 StyleGAN 中的映射网络、AdaIN 和风格混合，以实现解缠控制。"
order: 2
plots: []
sourceHash: "d58ec89094888d07821007149a10c947b9aae4d18d5e6c7dcba95d4346eb4041"
sourceCorrections: []
---

虽然像 ProGAN 这样的架构通过逐步增加网络深度来生成高分辨率图像，并表现出有效的方法，但控制生成图像的*具体特性*依然存在挑战。标准的潜在空间 $z$ 常常表现出缠结，这意味着改变 $z$ 中的一个维度可能会影响输出图像中多个不相关的特征。由 Karras 等人（NVIDIA）提出的 StyleGAN，代表着架构上的一大转变，其设计专门为了解决这个问题，并为合成过程提供更直观的控制。

StyleGAN 没有直接将初始潜在代码 $z$ 输入到生成器网络，而是引入了几个新颖的组件，它们共同作用，使不同精细程度的风格特征分离。

### 映射网络 ($f$)

第一个主要变化是引入了一个**映射网络**，记作 $f$。这通常是一个8层的多层感知器 (MLP)。其作用是将从标准分布（例如高斯分布）中提取的输入潜在代码 $z$ 转换为中间潜在空间 $W$。所以，$w = f(z)$，其中 $w \in W$。

> 映射网络将输入潜在 $z$ 转换为中间潜在 $w$ 的简化图示。

为何引入这个中间空间 $W$？映射网络 $f$ 不必保持输入空间 $Z$ 的分布。这种自由使得 $f$ 能够“解开”潜在空间，使 $W$ 可能比 $Z$ 更少缠结。数据中的变异因素可能在 $W$ 中表示得更线性，从而使得在生成过程中更容易控制特定属性。例如，在 $W$ 中分离身份、姿态和光照等因素可能更可行。

### 合成网络 ($g$)

第二个主要组成部分是**合成网络** $g$。与传统 GAN 生成器在第一层直接接收潜在代码 $z$ 作为输入不同，StyleGAN 合成网络从一个学习到的常数张量开始。中间潜在代码 $w$ 随后用于控制 $g$ 在网络中多个点生成的特征。

在合成网络的每个分辨率级别，也会明确地添加噪声。这种噪声是为每一层独立采样的，并提供了一种机制，使网络能够生成随机细节，例如毛发或雀斑的确切位置，而无需完全依赖于潜在代码 $w$。

### 自适应实例归一化 (normalization) (AdaIN)

将 $w$ 中的风格信息注入合成网络 $g$ 的机制是**自适应实例归一化 (AdaIN)**。AdaIN 根据从 $w$ 获得的风格信息修改合成网络的归一化激活。

回想一下，实例归一化 (IN) 对每个通道*每个样本*的特征统计量（均值和标准差）进行归一化。AdaIN 通过使用从中间潜在向量 (vector) $w$ 获得的风格尺度 ($\sigma(y_s)$) 和偏置 (bias) ($\mu(y_b)$) 来调制这些归一化特征，从而扩展了这一点。对于特定层 $i$ 的激活图 $x_i$:


$$
\text{AdaIN}(x_i, w) = y_{s,i} \left( \frac{x_i - \mu(x_i)}{\sigma(x_i)} \right) + y_{b,i}
$$


这里：

- $x_i$ 是层 $i$ 的激活张量。
- $\mu(x_i)$ 和 $\sigma(x_i)$ 是 $x_i$ 的均值和标准差，针对批次中的每个特征通道和每个样本，在空间维度上独立计算。
- $y_{s,i}$（尺度）和 $y_{b,i}$（偏置）是对中间潜在代码 $w$ 应用的学习到的仿射变换。一个单独的学习变换为应用 AdaIN 的每一层生成尺度和偏置。

简而言之，AdaIN 首先将特征图 $x_i$ 归一化，使其每个通道具有零均值和单位方差，从而移除编码在这些统计量中的原始风格信息。然后，它使用从 $w$ 获得的参数 (parameter)对归一化图进行缩放和偏移，有效地将 $w$ 指定的目标风格注入到该层的特征中。

> StyleGAN 合成网络 ($g$) 的图示。中间潜在 $w$ 通过 AdaIN 层控制风格，独立的噪声输入在不同分辨率下添加随机细节。

通过在合成网络中每个卷积层（或块）之后应用 AdaIN，StyleGAN 使用单个潜在向量 $w$ 控制不同尺度下表示的视觉风格特征（如配色方案、纹理、光照）。常数输入确保网络从头开始学习所有空间信息，仅由 $w$ 的风格输入和注入的噪声引导。

### 解缠与风格混合

这种架构促进解缠，因为映射网络 $f$ 可以学习将 $z$ 映射到一个缠结更少的 $W$ 空间，且合成网络 $g$ 通过 AdaIN 全局使用 $w$，而不仅仅是在输入层。这种结构有助于实现强大的控制机制：

- **风格混合：** 在训练期间，一部分图像使用两个不同的中间潜在代码 $w_1$ 和 $w_2$ 生成。一个代码 ($w_1$) 控制一部分层的风格（例如，粗糙空间分辨率，4x4 到 8x8），而另一个代码 ($w_2$) 控制其余层的风格（例如，更精细的分辨率，16x16 到 1024x1024）。这种技术鼓励网络将风格控制定位于特定的层子集，并防止它假设不同层次风格之间的关联。在推理 (inference)时，这允许创造性地混合风格：从由 $w_1$ 生成的图像中获取粗糙结构（姿态、脸型），并将其与由 $w_2$ 生成的图像中的更精细细节（头发纹理、肤色）结合。
- **感知路径长度 (PPL) 正则化 (regularization)：** StyleGAN 训练常常包含一个正则化项，鼓励从 $W$ 到图像空间的映射平滑。$W$ 中微小的变化应对应生成图像中微小的感知变化，这会进一步改善解缠效果和插值质量。（PPL 在第5章评估部分有更多介绍。）
- **截断技巧：** 尽管并非 StyleGAN 独有，但在这里它常被有效使用。通过采样 $w$ 向量 (vector)，然后将它们移向平均 $w$（根据许多样本计算得出），可以用多样性换取平均样本质量。这通过公式 $w' = \bar{w} + \psi (w - \bar{w})$ 完成，其中 $\bar{w}$ 是平均 $w$，$\psi \in [0, 1]$ 是截断系数。较小的 $\psi$ 会提高保真度但减少多样性。

总之，StyleGAN 架构凭借其映射网络、基于 AdaIN 的风格调制、噪声注入和常数输入，提供了一个生成高分辨率、高质量图像的强大框架，在风格属性控制方面相比之前的 GAN 架构有大为改善。它为 StyleGAN2 和其他先进生成模型的后续发展提供了重要的起始点。

## 参考资料

- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) — Tero Karras, Samuli Laine, Timo Aila (2019)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 4401-4410; DOI: [10.48550/arXiv.1812.04948](https://doi.org/10.48550/arXiv.1812.04948)
  介绍StyleGAN架构的原始论文，详细阐述了映射网络、用于风格注入的AdaIN、噪声添加以及风格混合的概念，以生成高分辨率图像并改进特征解耦。
- [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila (2020)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 8110-8119; DOI: [10.48550/arXiv.1912.04958](https://doi.org/10.48550/arXiv.1912.04958)
  这篇后续论文识别并解决了原始StyleGAN中的几个问题，催生了StyleGAN2。它通过架构更改消除了伪影，并通过非饱和路径长度正则化提高了感知质量。
- [Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play](https://learning.oreilly.com/library/view/generative-deep-learning/9781492041931/) — David Foster (2019)
  Publisher: O'Reilly Media
  一本全面书籍，提供各种生成模型（包括GAN）的实践见解。它提供了对StyleGAN和其他高级生成架构背后原理的更广泛理解，适合高级学习者。
