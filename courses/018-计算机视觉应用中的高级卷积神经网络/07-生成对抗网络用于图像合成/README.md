# 第 7 章：生成对抗网络用于图像合成

来源：[原章节](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-7-gans-image-synthesis)

[返回课程目录](../README.md)

本章将从图像内容的分析过渡到使用生成对抗网络（GAN）合成新图像。GAN 提供了一个训练模型的框架，通过生成器 $G$（负责创建数据）和判别器 $D$（试图区分真实数据和生成数据）之间的竞争过程，以生成逼真的输出，通常是图像。

我们将首先回顾 GAN 的基本原理及其对抗性目标函数。随后，我们将讨论模式崩溃和不稳定性等常见的训练难题，以及可能的解决方案。您将学习重要的 GAN 结构，包括深度卷积 GAN (DCGAN)、用于受控生成的条件 GAN (cGAN)，以及 StyleGAN 的基于风格的方法。评估生成图像质量和多样性的方法，例如 Fréchet Inception Distance (FID) 和 Inception Score (IS)，也将涉及。本章包含一个关于实现 DCGAN 的实践练习。

## 小节

- 1. [GAN 基本原理回顾](01-GAN%20%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E5%9B%9E%E9%A1%BE.md)
- 2. [训练生成对抗网络的挑战](02-%E8%AE%AD%E7%BB%83%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%E7%9A%84%E6%8C%91%E6%88%98.md)
- 3. [深度卷积生成对抗网络 (DCGAN)](03-%E6%B7%B1%E5%BA%A6%E5%8D%B7%E7%A7%AF%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%20%28DCGAN%29.md)
- 4. [条件GANs用于可控生成](04-%E6%9D%A1%E4%BB%B6GANs%E7%94%A8%E4%BA%8E%E5%8F%AF%E6%8E%A7%E7%94%9F%E6%88%90.md)
- 5. [StyleGAN 架构与基于风格的生成](05-StyleGAN%20%E6%9E%B6%E6%9E%84%E4%B8%8E%E5%9F%BA%E4%BA%8E%E9%A3%8E%E6%A0%BC%E7%9A%84%E7%94%9F%E6%88%90.md)
- 6. [GAN 的评估指标](06-GAN%20%E7%9A%84%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.md)
- 7. [图像生成实践中的DCGAN实现](07-%E5%9B%BE%E5%83%8F%E7%94%9F%E6%88%90%E5%AE%9E%E8%B7%B5%E4%B8%AD%E7%9A%84DCGAN%E5%AE%9E%E7%8E%B0.md)
