# 第 3 章：GAN训练的动态与稳定性

来源：[原章节](https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-3-gan-training-stabilization)

[返回课程目录](../README.md)

尽管生成对抗网络 (GANs) 的主要构想是生成器与判别器进行一场极小极大博弈，但在实际操作中，要实现稳定且有效的训练，通常需要特定的技术。标准的GAN训练可能会遇到问题，例如模式崩溃（生成器只产生有限的输出种类），或者梯度消失/爆炸，导致收敛困难。

本章将着重介绍这些训练难点，并介绍旨在提升稳定性的方法。我们将考察：

* 产生常见训练问题的原因。
* 基于 Wasserstein 距离 ($W_1$) 的替代损失函数，如 Wasserstein GANs (WGANs) 中所应用的。
* 用于强制执行 WGANs 所需的 Lipschitz 约束的方法，例如权重裁剪和梯度惩罚 (WGAN-GP)。
* 应用于判别器的正则化方法，例如谱归一化。
* 其他稳定方案，例如双时间尺度更新规则 (TTUR) 和相对论性 GANs。

您将了解这些方法背后的动因，以及它们如何改变训练过程，以促进更好的收敛并避免常见的失败情况。我们还将介绍实际的实现细节，使您能够将这些方法应用于您自己的 GAN 项目中。

## 小节

- 1. [不收敛的难题](01-%E4%B8%8D%E6%94%B6%E6%95%9B%E7%9A%84%E9%9A%BE%E9%A2%98.md)
- 2. [模式坍塌：成因与后果](02-%E6%A8%A1%E5%BC%8F%E5%9D%8D%E5%A1%8C%EF%BC%9A%E6%88%90%E5%9B%A0%E4%B8%8E%E5%90%8E%E6%9E%9C.md)
- 3. [其他散度：Wasserstein 距离](03-%E5%85%B6%E4%BB%96%E6%95%A3%E5%BA%A6%EF%BC%9AWasserstein%20%E8%B7%9D%E7%A6%BB.md)
- 4. [WGAN 中的权重剪裁](04-WGAN%20%E4%B8%AD%E7%9A%84%E6%9D%83%E9%87%8D%E5%89%AA%E8%A3%81.md)
- 5. [梯度惩罚 (WGAN-GP)](05-%E6%A2%AF%E5%BA%A6%E6%83%A9%E7%BD%9A%20%28WGAN-GP%29.md)
- 6. [谱范数归一化](06-%E8%B0%B1%E8%8C%83%E6%95%B0%E5%BD%92%E4%B8%80%E5%8C%96.md)
- 7. [双时间尺度更新规则 (TTUR)](07-%E5%8F%8C%E6%97%B6%E9%97%B4%E5%B0%BA%E5%BA%A6%E6%9B%B4%E6%96%B0%E8%A7%84%E5%88%99%20%28TTUR%29.md)
- 8. [相对论生成对抗网络](08-%E7%9B%B8%E5%AF%B9%E8%AE%BA%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C.md)
- 9. [WGAN-GP 的实现：实践](09-WGAN-GP%20%E7%9A%84%E5%AE%9E%E7%8E%B0%EF%BC%9A%E5%AE%9E%E8%B7%B5.md)
