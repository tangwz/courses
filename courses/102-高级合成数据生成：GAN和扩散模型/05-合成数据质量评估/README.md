# 第 5 章：合成数据质量评估

来源：[原章节](https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-5-evaluating-synthetic-data-quality)

[返回课程目录](../README.md)

生成合成数据只是第一步。训练好GAN或扩散模型后，如何判断生成的样本质量是否过关呢？仅仅查看几个例子可能会产生误导。本章侧重于严谨衡量生成模型所产合成数据的质量、多样性和逼真度所需的方法。

你将了解评估生成模型时固有的难点，例如简单指标如准确率并不适用。我们将介绍既定的定量指标，如Inception Score (IS) 和 Fréchet Inception Distance (FID)，并理解它们的计算方式和解读方法。你还将研究像Kernel Inception Distance (KID) 这样的分布指标，以及GAN特有的指标，例如衡量潜在空间质量的Perceptual Path Length (PPL)。

除了自动化指标之外，我们还将讨论定性评估技术，以及评估基于条件（例如类别标签）生成数据的模型时的具体考量。最后，你将获得编写代码计算FID的动手实践经验，FID是一种广泛使用的标准指标。本章结束时，你将拥有一个能够有效评估生成模型输出结果的工具集。

## 小节

- 1. [生成模型评估中的难题](01-%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0%E4%B8%AD%E7%9A%84%E9%9A%BE%E9%A2%98.md)
- 2. [定量指标：IS、FID、精确率、召回率](02-%E5%AE%9A%E9%87%8F%E6%8C%87%E6%A0%87%EF%BC%9AIS%E3%80%81FID%E3%80%81%E7%B2%BE%E7%A1%AE%E7%8E%87%E3%80%81%E5%8F%AC%E5%9B%9E%E7%8E%87.md)
- 3. [分布度量：核Inception距离 (KID)](03-%E5%88%86%E5%B8%83%E5%BA%A6%E9%87%8F%EF%BC%9A%E6%A0%B8Inception%E8%B7%9D%E7%A6%BB%20%28KID%29.md)
- 4. [生成对抗网络（GAN）的感知路径长度（PPL）](04-%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%EF%BC%88GAN%EF%BC%89%E7%9A%84%E6%84%9F%E7%9F%A5%E8%B7%AF%E5%BE%84%E9%95%BF%E5%BA%A6%EF%BC%88PPL%EF%BC%89.md)
- 5. [定性评估方法](05-%E5%AE%9A%E6%80%A7%E8%AF%84%E4%BC%B0%E6%96%B9%E6%B3%95.md)
- 6. [评估条件生成模型](06-%E8%AF%84%E4%BC%B0%E6%9D%A1%E4%BB%B6%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B.md)
- 7. [动手实践：计算FID分数](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%A1%E7%AE%97FID%E5%88%86%E6%95%B0.md)
