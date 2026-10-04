# 第 3 章：GAN训练稳定性与优化

来源：[原章节](https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-3-gan-training-stability-optimization)

[返回课程目录](../README.md)

生成对抗网络能生成令人印象深刻的结果，但其训练过程常不稳定。在最小最大目标函数 $$ \min_G \max_D V(D, G) $$ 中，实现生成器 $G$ 与判别器 $D$ 之间的收敛常常有难度，这会引起样本质量差或多样性不足等问题。

本章提供实用的方法来诊断并解决这些训练难题。您将学到：
*   识别出模式坍塌、训练震荡和散度等常见问题。
*   实施瓦瑟斯坦GAN（带梯度惩罚WGAN-GP）和最小二乘GAN（LSGAN）等替代损失函数，以提高稳定性。
*   应用正则化技术，包括谱归一化，到生成器和判别器。
*   运用双时间尺度更新规则 (TTUR) 等优化策略。
*   制定针对GAN的有效超参数调优方法。

我们将涵盖这些技术背后的原理，并引导您实现WGAN-GP等核心解决方案。

## 小节

- 1. [诊断训练不稳定：振荡与发散](01-%E8%AF%8A%E6%96%AD%E8%AE%AD%E7%BB%83%E4%B8%8D%E7%A8%B3%E5%AE%9A%EF%BC%9A%E6%8C%AF%E8%8D%A1%E4%B8%8E%E5%8F%91%E6%95%A3.md)
- 2. [模式坍塌：原因与缓解策略](02-%E6%A8%A1%E5%BC%8F%E5%9D%8D%E5%A1%8C%EF%BC%9A%E5%8E%9F%E5%9B%A0%E4%B8%8E%E7%BC%93%E8%A7%A3%E7%AD%96%E7%95%A5.md)
- 3. [替代损失函数（WGAN, WGAN-GP, LSGAN）](03-%E6%9B%BF%E4%BB%A3%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88WGAN%2C%20WGAN-GP%2C%20LSGAN%EF%BC%89.md)
- 4. [GAN 的正则化方法](04-GAN%20%E7%9A%84%E6%AD%A3%E5%88%99%E5%8C%96%E6%96%B9%E6%B3%95.md)
- 5. [双时间尺度更新规则 (TTUR)](05-%E5%8F%8C%E6%97%B6%E9%97%B4%E5%B0%BA%E5%BA%A6%E6%9B%B4%E6%96%B0%E8%A7%84%E5%88%99%20%28TTUR%29.md)
- 6. [GAN的超参数调整策略](06-GAN%E7%9A%84%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E6%95%B4%E7%AD%96%E7%95%A5.md)
- 7. [动手实践：WGAN-GP 的实现](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9AWGAN-GP%20%E7%9A%84%E5%AE%9E%E7%8E%B0.md)
