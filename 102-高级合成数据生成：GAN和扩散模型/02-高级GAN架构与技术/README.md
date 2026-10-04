# 第 2 章：高级GAN架构与技术

来源：[原章节](https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-2-advanced-gan-architectures-techniques)

[返回课程目录](../README.md)

在回顾了GAN基础知识之后，本章考察了生成对抗网络设计中的几项重要进展。虽然基本的GAN提供了一个起点，但它们在输出分辨率、训练稳定性和生成过程控制方面常遇到局限。在此，我们将学习专门为应对这些问题而开发的架构。

我们将涉及：

*   **渐进式增长GAN (ProGAN):** 一种能实现稳定训练的方法，通过逐步增加网络深度来生成高分辨率图像。
*   **基于风格的生成器 (StyleGAN变体):** 那些能对图像属性提供更好控制的模型，借助风格混合和自适应实例归一化 (AdaIN) 等技术。
*   **非配对图像到图像转换 (CycleGAN):** 一种采用循环一致性损失 ($L_{\text{cyc}}$) 的方法，用于域适配，无需配对训练样本。

此外，我们将研究根据特定输入生成数据的方法，引入注意力机制以捕获长程依赖，以及分析和操作潜在空间 $z$ 来影响生成输出的技术。一个动手实践部分将侧重于实现StyleGAN架构的核心组成部分。

## 小节

- 1. [渐进式生成对抗网络 (ProGAN)](01-%E6%B8%90%E8%BF%9B%E5%BC%8F%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%20%28ProGAN%29.md)
- 2. [基于风格的生成器（StyleGAN变体）](02-%E5%9F%BA%E4%BA%8E%E9%A3%8E%E6%A0%BC%E7%9A%84%E7%94%9F%E6%88%90%E5%99%A8%EF%BC%88StyleGAN%E5%8F%98%E4%BD%93%EF%BC%89.md)
- 3. [非配对图像到图像转换 (CycleGAN)](03-%E9%9D%9E%E9%85%8D%E5%AF%B9%E5%9B%BE%E5%83%8F%E5%88%B0%E5%9B%BE%E5%83%8F%E8%BD%AC%E6%8D%A2%20%28CycleGAN%29.md)
- 4. [条件生成对抗网络：架构与控制](04-%E6%9D%A1%E4%BB%B6%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%EF%BC%9A%E6%9E%B6%E6%9E%84%E4%B8%8E%E6%8E%A7%E5%88%B6.md)
- 5. [GAN中的注意力机制](05-GAN%E4%B8%AD%E7%9A%84%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md)
- 6. [分析与调整GAN潜在空间](06-%E5%88%86%E6%9E%90%E4%B8%8E%E8%B0%83%E6%95%B4GAN%E6%BD%9C%E5%9C%A8%E7%A9%BA%E9%97%B4.md)
- 7. [动手实践：实现StyleGAN组件](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0StyleGAN%E7%BB%84%E4%BB%B6.md)
