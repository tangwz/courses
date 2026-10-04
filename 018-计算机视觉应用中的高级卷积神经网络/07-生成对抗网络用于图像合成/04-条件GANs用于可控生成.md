# 条件GANs用于可控生成

来源：[原文](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-7-gans-image-synthesis/conditional-gans-cgan)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管标准生成对抗网络 (GAN)（GANs）学习生成模拟数据分布的样本，但它们对生成的具体输出几乎没有控制。给定一个噪声向量 (vector) $z$，一个普通GAN会生成一个图像 $G(z)$，看起来像是训练数据集中的一员，但你无法轻易指定你想要哪种类型的图像。条件生成对抗网络（cGANs）扩展了GAN框架来解决这一局限，通过在生成过程中加入额外信息，即“条件”。这使得基于特定属性进行目标合成成为可能。

核心理念是为生成器和判别器都提供一些额外信息 $y$，这些信息可以是类别标签、描述性文本，甚至是另一幅图像。生成器的任务变为生成与条件 $y$ 相符的真实样本，而判别器则必须学会区分真实的（图像，条件）对和伪造的（生成的图像，条件）对。

### 架构调整

在标准GAN中，生成器将随机噪声向量 (vector) $z$ 映射到输出图像 $G(z)$。判别器接收图像 $x$（可以是真实的或生成的）并输出一个概率 $D(x)$，表示其认为 $x$ 是否真实。

在cGAN中，输入进行了调整：

- **生成器：** 同时接收噪声向量 $z$ 和条件信息 $y$ 作为输入。它生成图像 $G(z, y)$，此图像应与条件 $y$ 一致。
- **判别器：** 同时接收图像 $x$（真实的或生成的）和相应的条件信息 $y$ 作为输入。它输出一个概率 $D(x, y)$，表示 $x$ 是与条件 $y$ 匹配的真实图像的可能性。

> 条件GAN架构的简化图示。条件 `y` 作为额外输入提供给生成器 `G` 和判别器 `D`。

条件 $y$ 的整合方式取决于其性质和网络架构。对于离散标签（如MNIST中的数字0-9），$y$ 可以表示为独热向量并直接与 $z$ 拼接后输入生成器。对于判别器，它可能被重塑为特征图并按通道与图像输入拼接，或者进行嵌入 (embedding)后与中间特征层结合。对于文本描述或图像等更复杂的条件，通常会使用嵌入层或单独的编码器网络来将 $y$ 处理成合适的向量表示，然后再进行组合。

### 条件损失函数 (loss function)

cGAN的目标函数通过纳入条件 $y$ 来修改标准最小-最大博弈。值函数 $V(D, G)$ 变为：


$$
\min_G \max_D V(D, G) = \mathbb{E}_{(x, y) \sim p_{data}(x, y)}[\log D(x, y)] + \mathbb{E}_{z \sim p_z(z), y \sim p_y(y)}[\log(1 - D(G(z, y), y))]
$$


其中：

- $\mathbb{E}_{(x, y) \sim p_{data}(x, y)}[\log D(x, y)]$ 代表判别器的目标，即正确识别与真实条件 $y$ 配对的真实图像 $x$。期望是针对从真实数据分布 $p_{data}$ 中抽取的真实数据对 $(x, y)$ 计算的。
- $\mathbb{E}_{z \sim p_z(z), y \sim p_y(y)}[\log(1 - D(G(z, y), y))]$ 代表判别器的目标，即在给定条件 $y$ 的情况下，将生成的图像 $G(z, y)$ 识别为假。生成器 $G$ 同时试图最小化此项，使其输出 $G(z, y)$ 在给定条件 $y$ 的情况下与真实图像无法区分。噪声 $z$ 从其先验分布 $p_z$ 中抽取，条件 $y$ 通常从其边际分布 $p_y$ 中抽取（或与 $x$ 一起从联合数据分布中采样）。

在训练期间，小批量数据由以下对组成：(真实图像 $x$，条件 $y$) 和 (噪声向量 (vector) $z$，条件 $y$)。生成器使用 $(z, y)$ 来生成 $G(z, y)$。判别器训练时，对 (真实图像 $x$，条件 $y$) 的目标输出接近1，对 (生成的图像 $G(z, y)$，条件 $y$) 的目标输出接近0。生成器根据判别器对 (生成的图像 $G(z, y)$，条件 $y$) 的输出进行训练，目标是让判别器输出1。

### 条件GANs的应用

条件生成开启了诸多可能性：

- **类别条件图像合成：** 生成特定类别的图像，例如创建特定犬种的图像或特定数字。这里，$y$ 是类别标签。
- **文本到图像合成：** 根据文本描述生成图像。条件 $y$ 是从输入文本的嵌入 (embedding)中获得的。这使得可以根据诸如“一只红色的鸟停在树枝上”之类的句子创建图像。
- **图像到图像转换：** 将图像从一个领域转换到另一个领域，例如将草图转换为照片、卫星图像转换为地图，或改变艺术风格。在这种情况下，条件 $y$ 是输入图像本身（例如，草图），而 $G(z, y)$ 则生成转换后的输出（例如，照片）。Pix2Pix等架构是其中的突出例子。
- **属性修改：** 通过条件化属性向量 (vector)来修改图像的特定属性，例如改变头发颜色或给脸部添加眼镜。

通过纳入条件信息 $y$，cGANs与非条件GAN相比，对生成过程提供了更多的控制。这使得它们成为需要目标合成或视觉数据转换任务的有力工具。挑战通常在于如何有效地将条件 $y$ 整合到网络架构中，并确保生成器正确使用此信息以生成相关且高质量的输出。

## 参考资料

- [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — Mehdi Mirza, Simon Osindero (2014)
  Journal: arXiv preprint arXiv:1411.1784; DOI: [10.48550/arXiv.1411.1784](https://doi.org/10.48550/arXiv.1411.1784)
  介绍了条件生成对抗网络（cGAN）框架，详细阐述了其架构和目标函数，用于受控数据生成。
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems (NIPS) 27; Pages: 2672-2680; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  这篇开创性论文介绍了生成对抗网络（GAN）框架，为所有后续GAN变体（包括cGAN）奠定了基础。
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros (2017)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 1125-1134; DOI: [10.1109/CVPR.2017.39](https://doi.org/10.1109/CVPR.2017.39)
  介绍了Pix2Pix，一种用于图像到图像转换的通用cGAN，展示了其在各种任务中的有效性。
- [StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1612.03242) — Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas (2017)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Publisher: IEEE; Pages: 5908-5916; DOI: [10.48550/arXiv.1612.03242](https://doi.org/10.48550/arXiv.1612.03242)
  描述了StackGAN，一种从文本描述生成高分辨率图像的创新cGAN方法。

---

[上一节](03-%E6%B7%B1%E5%BA%A6%E5%8D%B7%E7%A7%AF%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%20%28DCGAN%29.md) · [下一节](05-StyleGAN%20%E6%9E%B6%E6%9E%84%E4%B8%8E%E5%9F%BA%E4%BA%8E%E9%A3%8E%E6%A0%BC%E7%9A%84%E7%94%9F%E6%88%90.md)
