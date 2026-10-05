# 第 4 章：条件式与可控生成

来源：[原章节](https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-4-conditional-controllable-generation)

[返回课程目录](../README.md)

标准生成对抗网络生成的样本反映训练数据分布，但对具体输出的直接控制有限。本章介绍指导生成过程的方法。您将学习如何实现条件生成对抗网络（cGAN），它能生成与特定属性或标签（通常表示为 $y$）相符的数据。我们将涵盖如何将这些条件信息 $y$ 整合到生成器 $G(z, y)$ 和判别器 $D(x, y)$ 中。

接着我们将学习信息最大化生成对抗网络（InfoGAN），这是一种通过最大化潜在代码 $c$ 与生成器输出之间的互信息 $I(c; G(z, c))$，以无监督方式学习可解释的潜在代码的方法。本章也提及了操控学习到的潜在空间 $z$ 以修改生成后的输出特征的方法，并讨论了像 StackGAN 这样的方法，用于文本到图像合成等任务。最后，我们将考察学到的表示中解耦的原理与度量。动手实践环节提供构建这些条件模型的实践机会。

## 小节

- 1. [条件式GAN（cGAN）介绍](01-%E6%9D%A1%E4%BB%B6%E5%BC%8FGAN%EF%BC%88cGAN%EF%BC%89%E4%BB%8B%E7%BB%8D.md)
- 2. [cGAN的架构](02-cGAN%E7%9A%84%E6%9E%B6%E6%9E%84.md)
- 3. [信息最大化GAN (InfoGAN)](03-%E4%BF%A1%E6%81%AF%E6%9C%80%E5%A4%A7%E5%8C%96GAN%20%28InfoGAN%29.md)
- 4. [StackGAN：文本到图像生成](04-StackGAN%EF%BC%9A%E6%96%87%E6%9C%AC%E5%88%B0%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90.md)
- 5. [通过潜在空间操作控制属性](05-%E9%80%9A%E8%BF%87%E6%BD%9C%E5%9C%A8%E7%A9%BA%E9%97%B4%E6%93%8D%E4%BD%9C%E6%8E%A7%E5%88%B6%E5%B1%9E%E6%80%A7.md)
- 6. [解耦度量与挑战](06-%E8%A7%A3%E8%80%A6%E5%BA%A6%E9%87%8F%E4%B8%8E%E6%8C%91%E6%88%98.md)
- 7. [构建条件生成对抗网络：实操练习](07-%E6%9E%84%E5%BB%BA%E6%9D%A1%E4%BB%B6%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%EF%BC%9A%E5%AE%9E%E6%93%8D%E7%BB%83%E4%B9%A0.md)
