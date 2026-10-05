# 第 2 章：变分自编码器：数学原理

来源：[原章节](https://apxml.com/zh/courses/vae-representation-learning/chapter-2-vaes-mathematical-deep-dive)

[返回课程目录](../README.md)

在确立了概率生成模型和表示学习的要点之后，本章侧重介绍变分自编码器（VAEs）的数学构造。对这些数学原理的扎实理解，对于有效使用和扩展VAEs来说是不可或缺的。

在这里，你将学习从变分推断的原理推导出VAEs。我们将分析证据下界 (ELBO)，通常写作 $L_{ELBO}$，它构成了训练VAEs的主要目标函数。你将理解重参数化技巧，这是一个重要技巧，它使通过随机隐变量进行基于梯度的优化成为可能。Kullback-Leibler (KL) 散度（通常写作 $D_{KL}(q(z|x) || p(z))$）在VAEs目标函数中的作用和解释也将被详细阐述。此外，我们还将涵盖编码器和解码器网络的设计考虑因素，讨论常见的VAEs训练难题，并分析VAEs目标函数的不同表达形式。本章包含一个实践部分，关于如何实现一个VAEs并执行诊断，以将理论与应用联系起来。

## 小节

- 1. [VAE推导：变分推断](01-VAE%E6%8E%A8%E5%AF%BC%EF%BC%9A%E5%8F%98%E5%88%86%E6%8E%A8%E6%96%AD.md)
- 2. [证据下界 (ELBO) 公式](02-%E8%AF%81%E6%8D%AE%E4%B8%8B%E7%95%8C%20%28ELBO%29%20%E5%85%AC%E5%BC%8F.md)
- 3. [重参数化技巧](03-%E9%87%8D%E5%8F%82%E6%95%B0%E5%8C%96%E6%8A%80%E5%B7%A7.md)
- 4. [VAE中的KL散度：作用与解释](04-VAE%E4%B8%AD%E7%9A%84KL%E6%95%A3%E5%BA%A6%EF%BC%9A%E4%BD%9C%E7%94%A8%E4%B8%8E%E8%A7%A3%E9%87%8A.md)
- 5. [VAE 编码器和解码器网络设计](05-VAE%20%E7%BC%96%E7%A0%81%E5%99%A8%E5%92%8C%E8%A7%A3%E7%A0%81%E5%99%A8%E7%BD%91%E7%BB%9C%E8%AE%BE%E8%AE%A1.md)
- 6. [常见 VAE 训练问题](06-%E5%B8%B8%E8%A7%81%20VAE%20%E8%AE%AD%E7%BB%83%E9%97%AE%E9%A2%98.md)
- 7. [VAE目标函数分析](07-VAE%E7%9B%AE%E6%A0%87%E5%87%BD%E6%95%B0%E5%88%86%E6%9E%90.md)
- 8. [VAE 实现与诊断：动手操作](08-VAE%20%E5%AE%9E%E7%8E%B0%E4%B8%8E%E8%AF%8A%E6%96%AD%EF%BC%9A%E5%8A%A8%E6%89%8B%E6%93%8D%E4%BD%9C.md)
