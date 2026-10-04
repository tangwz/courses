# 第 6 章：变分自编码器（VAEs）与结构化潜在空间

来源：[原章节](https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-6-variational-autoencoders-structured-latent-spaces)

[返回课程目录](../README.md)

前面的章节介绍了自编码器，它们主要是用于降维和特征学习的工具，网络通过学习重建其输入。本章介绍变分自编码器（VAEs），这是一种独特的自编码器，结合了概率视角。VAEs 的设计目的不仅是重建，还在于学习一个连续且有结构的潜在空间，这使它们在生成模型方面特别有效——即生成新的数据样本。

通过学习本章，您将了解到：

*   生成模型的基本思想以及 VAEs 如何适应这个方面。
*   变分自编码器的核心原理，包括它们的概率编码器和解码器。
*   VAE 编码器如何构造以输出参数（通常是均值 $\mu$ 和对数方差 $\log(\sigma^2)$），这些参数为每个输入定义了潜在空间中的概率分布。
*   重参数化技巧，这是一种允许梯度通过采样过程反向传播的方法，这对训练 VAEs 非常重要。
*   VAE 解码器作为生成器，如何使用从潜在分布中抽取的样本来生成新数据。
*   VAE 损失函数，它通常由两个主要部分组成：重建损失（例如，均方误差或二元交叉熵）和一个 Kullback-Leibler (KL) 散度项。KL 散度作为正则项，促使学到的潜在分布接近先验分布（通常是标准正态分布 $\mathcal{N}(0, I)$）。总损失可以表示为 $$L_{VAE} = \text{ReconstructionLoss} + D_{KL}(q(z|x) || p(z))$$
*   VAE 潜在空间的特性，例如其平滑性以及如何促进数据点之间的插值。
*   使用 VAEs 潜在空间中的均值向量（或样本）作为特征，用于其他机器学习任务的方法。

一个实践部分将演示如何构建 VAE 并直观检查其潜在空间，以观察其学习到的结构。

## 小节

- 1. [基于自编码器的生成模型介绍](01-%E5%9F%BA%E4%BA%8E%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E7%9A%84%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E4%BB%8B%E7%BB%8D.md)
- 2. [变分自编码器构成](02-%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E6%9E%84%E6%88%90.md)
- 3. [VAE编码器：输出分布参数](03-VAE%E7%BC%96%E7%A0%81%E5%99%A8%EF%BC%9A%E8%BE%93%E5%87%BA%E5%88%86%E5%B8%83%E5%8F%82%E6%95%B0.md)
- 4. [重参数化技巧解释](04-%E9%87%8D%E5%8F%82%E6%95%B0%E5%8C%96%E6%8A%80%E5%B7%A7%E8%A7%A3%E9%87%8A.md)
- 5. [VAE解码器：从潜在样本生成数据](05-VAE%E8%A7%A3%E7%A0%81%E5%99%A8%EF%BC%9A%E4%BB%8E%E6%BD%9C%E5%9C%A8%E6%A0%B7%E6%9C%AC%E7%94%9F%E6%88%90%E6%95%B0%E6%8D%AE.md)
- 6. [VAE损失函数：平衡重建与正则化](06-VAE%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%9A%E5%B9%B3%E8%A1%A1%E9%87%8D%E5%BB%BA%E4%B8%8E%E6%AD%A3%E5%88%99%E5%8C%96.md)
- 7. [VAE 潜在空间的特点](07-VAE%20%E6%BD%9C%E5%9C%A8%E7%A9%BA%E9%97%B4%E7%9A%84%E7%89%B9%E7%82%B9.md)
- 8. [使用VAE潜在表示作为特征](08-%E4%BD%BF%E7%94%A8VAE%E6%BD%9C%E5%9C%A8%E8%A1%A8%E7%A4%BA%E4%BD%9C%E4%B8%BA%E7%89%B9%E5%BE%81.md)
- 9. [实作：构建变分自编码器并检验其潜在空间](09-%E5%AE%9E%E4%BD%9C%EF%BC%9A%E6%9E%84%E5%BB%BA%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E5%B9%B6%E6%A3%80%E9%AA%8C%E5%85%B6%E6%BD%9C%E5%9C%A8%E7%A9%BA%E9%97%B4.md)

章节测验：[在线测验](https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-6-variational-autoencoders-structured-latent-spaces/quiz)
