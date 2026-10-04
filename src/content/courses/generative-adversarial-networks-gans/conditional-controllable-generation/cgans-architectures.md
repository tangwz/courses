---
course: "generative-adversarial-networks-gans"
chapter: "conditional-controllable-generation"
lesson: "cgans-architectures"
sourceId: 2712
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-4-conditional-controllable-generation/cgans-architectures"
title: "cGAN的架构"
description: "将条件信息并入生成器和判别器的常见方法。"
order: 2
plots: []
sourceHash: "9116dc2648d4f3e8794450ae80b77812f708053eaa332d2deb4f83c985d26ab7"
sourceCorrections: []
---

为有效控制生成对抗网络 (GAN)的输出，我们需引入将条件信息（表示为$y$）并入生成过程的机制。此条件$y$可以代表多种事物，比如类别标签（例如“狗”、“猫”）、特定属性（例如发色、姿势）、文本描述，甚至是另一张图像。条件GAN（cGAN）的核心构思是修改生成器$G$和判别器$D$使其基于$y$运行，从而得到$G(z, y)$和$D(x, y)$。实现这种条件设置的常见架构方法如下。

### 生成器的条件设置

生成器的任务是产生一个既属于目标数据分布又与所给条件$y$对应的样本$x$。潜在向量 (vector)$z$仍提供变异的来源，而$y$则指导合成朝向特定的模式或特征。

#### 1. 基于拼接的条件设置

最直接的方法是将条件信息$y$连同潜在向量$z$一起直接输入到生成器。

- **输入拼接：** 如果$y$表示为一个向量（例如独热编码的类别标签或文本/图像的嵌入 (embedding)），它可与潜在向量$z$拼接。这个组合向量$[z, y]$随后作为生成器第一层的输入。

> 将潜在向量$z$与（可能已嵌入的）条件向量$y$拼接，作为生成器网络的输入。

- **中间层拼接：** 条件$y$（通常在通过其自身的嵌入层或小型神经网络 (neural network)处理后）不仅可以在输入端，还可以被重塑并拼接至生成器架构中的中间特征图。这使得条件能够在特征抽象的多个层次上影响生成过程。对于卷积生成器，$y$可能在沿通道轴拼接前被空间复制以匹配特征图的尺寸。

#### 2. 条件调制技术

更复杂的方法涉及使用条件$y$来调制生成器中归一化 (normalization)层的行为。

- **条件批量归一化 (CBN)：** 标准批量归一化在一个层内对激活进行归一化，然后应用学到的仿射变换参数 (parameter)：缩放($\gamma$)和平移($\beta$)。在CBN中，这些参数$\gamma$和$\beta$不再是全局学习的，而是由接收条件$y$作为输入的小型神经网络预测的。即，$\gamma = f_\gamma(y)$和$\beta = f_\beta(y)$。这使得条件能够动态控制整个生成器中的特征统计量（均值和方差），从而对合成过程提供精细的控制，通常影响与$y$相关的风格方面。
- **自适应实例归一化 (AdaIN)：** 由StyleGAN推广，AdaIN在思路上与CBN相似，但作用于实例归一化。它将内容特征（来自$z$）的均值和标准差对齐 (alignment)，以匹配从风格输入（可通过映射网络受$y$影响）派生的均值和标准差。这有效地在合成网络的多个点“注入”风格或条件$y$。

这些调制技术相比简单的拼接，通常能带来更好的解缠和控制，特别是对于复杂、高分辨率的生成任务。

### 判别器的条件设置

判别器在cGAN中的作用是双重的：判断输入样本$x$是真实还是虚假，*并*验证$x$是否与给定条件$y$匹配。若没有第二部分，生成器可能会学着忽略$y$，从而产生真实但无关的样本。

#### 1. 输入拼接

与生成器类似，告知判别器条件的最直接方式是通过拼接。

- **输入与条件拼接：** 输入样本$x$（真实或生成）通常通过早期的卷积层进行处理，以获得特征表示。条件向量 (vector)$y$（可能已嵌入 (embedding)）也经过处理并重塑。随后，这两个表示被拼接并输入到判别器的后续层，由这些层产生最终的真实/虚假预测。

> 在最终判别前，分别处理输入$x$和条件$y$，然后拼接它们的表示。

#### 2. 投影判别器

一种更有效且广泛采用的技术，特别是对于高维输入和大量类别，是投影判别器。它不是简单拼接，而是显式地将输入$x$与条件$y$的匹配整合到判别器的输出对数中。

该架构工作方式如下：

1. 条件$y$（通常是一个类别标签）被嵌入到一个向量$v_y$中。
2. 输入图像$x$通过判别器网络的主体（例如卷积层），获得一个中间特征表示$\phi(x)$。
3. 核心构思是计算图像特征$\phi(x)$与嵌入条件$v_y$之间的点积（或类似的双线性池化）。这一项$\phi(x)^T v_y$衡量图像特征与特定条件之间的兼容性或对齐 (alignment)度。
4. 判别器的最终输出对数被计算为两项之和：一项仅基于图像特征（代表独立于条件的“真实性”），以及在步骤3中计算的投影项。


$$
\text{输出对数} = W \phi(x) + \phi(x)^T v_y
$$


此处，$W \phi(x)$捕获无条件真实/虚假分数，而$\phi(x)^T v_y$显式奖励判别器识别出其特征$\phi(x)$与*正确*条件$y$的嵌入$v_y$良好对齐的图像。这种架构强烈鼓励生成器产生不仅真实而且准确反映给定条件$y$的样本。

> 投影判别器的架构，将无条件分数与条件投影项组合。

#### 3. 辅助分类器GAN (AC-GAN)

AC-GAN虽有不同，但也存在相似之处。在AC-GAN中，判别器同时执行两个任务：它预测输入$x$是真实还是虚假，*并*预测与$x$相关的类别标签$y$。其损失函数 (loss function)既包含标准对抗损失，也包含辅助分类损失（例如交叉熵）。这迫使生成器产生不仅真实而且能被判别器根据预期条件$y$正确分类的样本。该架构的修改在于为判别器添加一个单独的输出头，用于类别预测任务。

选择合适的条件设置架构取决于条件$y$的性质、生成任务的复杂性以及计算限制。拼接更易于实现，而调制技术（如CBN）和投影判别器通常为复杂的条件生成任务提供卓越的性能和控制。

## 参考资料

- [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — Mehdi Mirza, Simon Osindero (2014)
  Journal: arXiv preprint arXiv:1411.1784; DOI: [10.48550/arXiv.1411.1784](https://doi.org/10.48550/arXiv.1411.1784)
  介绍了条件生成对抗网络的基本概念，展示了如何将类别标签融入生成器和判别器以实现受控生成。
- [Conditional Image Synthesis with Auxiliary Classifier GANs](https://arxiv.org/abs/1610.09585) — Augustus Odena, Christopher Olah, Jonathon Shlens (2017)
  Journal: International Conference on Machine Learning (ICML); Pages: 2642-2651; DOI: [10.48550/arXiv.1610.09585](https://doi.org/10.48550/arXiv.1610.09585)
  提出了辅助分类器GAN架构，其中判别器同时预测真/假和类别标签，增强了生成器生成类别条件图像的能力。
- [Large-scale GAN training for high-fidelity natural image synthesis](https://arxiv.org/abs/1809.11096) — Andrew Brock, Jeff Donahue, Karen Simonyan (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1809.11096](https://doi.org/10.48550/arXiv.1809.11096)
  引入了投影判别器，通过在判别器输出中加入图像特征与类别嵌入之间的显式内积项，显著提升了条件GAN的性能。
