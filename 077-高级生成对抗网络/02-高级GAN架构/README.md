# 第 2 章：高级GAN架构

来源：[原章节](https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-2-advanced-gan-architectures)

[返回课程目录](../README.md)

基于之前讨论的基础原理和局限，本章研究了使得生成对抗网络在质量、稳定性、和可控性方面大幅提升的架构创新。标准GAN在生成高分辨率图像或保持训练稳定性方面常遇到困难。在此，我们学习旨在解决这些问题的特定架构。

我们将涵盖几项重要进展：

*   **渐进式增长（ProGAN）：** 学习如何在训练期间逐步提升网络分辨率，帮助稳定高清晰度图像的生成过程。
*   **基于风格的生成（StyleGAN / StyleGAN2）：** 理解基于风格的生成器的运作方式，包括映射网络和自适应实例归一化（AdaIN），这些机制增强了对生成图像属性的控制。
*   **GAN扩展（BigGAN）：** 研究正交正则化和自注意力等技术，这些技术使得GAN可以在更大规模上有效训练，生成高质量和多样化的样本。
*   **非配对图像转换（CycleGAN）：** 学习循环一致性损失的思路，其公式为 $L_{cyc}(G, F) = \mathbb{E}_{x \sim p_{data}(x)}[||F(G(x)) - x||_1] + \mathbb{E}_{y \sim p_{data}(y)}[||G(F(y)) - y||_1]$，使得在没有配对数据的情况下进行域间转换成为可能。
*   **注意力机制：** 分析如何引入自注意力层帮助模型捕获长距离空间依赖关系。

通过剖析这些架构，你将了解驱动现代高性能生成模型的原理。本章还包括StyleGAN等重要组件的实际实现指导。

## 小节

- 1. [渐进式生成对抗网络 (ProGAN)](01-%E6%B8%90%E8%BF%9B%E5%BC%8F%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%20%28ProGAN%29.md)
- 2. [基于风格的生成器架构 (StyleGAN)](02-%E5%9F%BA%E4%BA%8E%E9%A3%8E%E6%A0%BC%E7%9A%84%E7%94%9F%E6%88%90%E5%99%A8%E6%9E%B6%E6%9E%84%20%28StyleGAN%29.md)
- 3. [StyleGAN2 改进](03-StyleGAN2%20%E6%94%B9%E8%BF%9B.md)
- 4. [大规模GAN训练 (BigGAN)](04-%E5%A4%A7%E8%A7%84%E6%A8%A1GAN%E8%AE%AD%E7%BB%83%20%28BigGAN%29.md)
- 5. [GAN中的自注意力机制](05-GAN%E4%B8%AD%E7%9A%84%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md)
- 6. [非配对图像到图像转换 (CycleGAN)](06-%E9%9D%9E%E9%85%8D%E5%AF%B9%E5%9B%BE%E5%83%8F%E5%88%B0%E5%9B%BE%E5%83%8F%E8%BD%AC%E6%8D%A2%20%28CycleGAN%29.md)
- 7. [StyleGAN 组件的动手实现](07-StyleGAN%20%E7%BB%84%E4%BB%B6%E7%9A%84%E5%8A%A8%E6%89%8B%E5%AE%9E%E7%8E%B0.md)
