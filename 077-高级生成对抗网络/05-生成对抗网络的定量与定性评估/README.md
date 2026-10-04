# 第 5 章：生成对抗网络的定量与定性评估

来源：[原章节](https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-5-evaluation-of-gans)

[返回课程目录](../README.md)

评估生成对抗网络（GAN）的性能，相较于典型的监督学习任务，会面临一些特殊的难点。由于GANs学习逼近复杂的数据分布，仅仅查看训练时的损失函数往往无法体现生成样本的真实质量或多样性。判断GAN是否生成逼真输出，并涵盖真实数据的多样性，需要专门的评估方法。

本章介绍用于评估GAN的方法，涵盖定性与定量两类方法。您将了解此评估过程中本身存在的难题。我们将考察视觉检查等定性方法。更为重要的是，我们将侧重于旨在衡量生成性能不同方面的定量指标。您将学习Inception Score (IS)和Fréchet Inception Distance (FID)等常用指标的公式、解释及其局限性。我们还将介绍Precision和Recall等用于比较分布的指标，以及用于评估潜在空间属性的Perceptual Path Length (PPL)。本章包含关于计算和解释这些分数的实用指导，以便比较不同模型和追踪训练进展。

## 小节

- 1. [评估生成模型的挑战](01-%E8%AF%84%E4%BC%B0%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8C%91%E6%88%98.md)
- 2. [定性评估：视觉图灵测试](02-%E5%AE%9A%E6%80%A7%E8%AF%84%E4%BC%B0%EF%BC%9A%E8%A7%86%E8%A7%89%E5%9B%BE%E7%81%B5%E6%B5%8B%E8%AF%95.md)
- 3. [Inception Score (IS)：计算方法与局限性](03-Inception%20Score%20%28IS%29%EF%BC%9A%E8%AE%A1%E7%AE%97%E6%96%B9%E6%B3%95%E4%B8%8E%E5%B1%80%E9%99%90%E6%80%A7.md)
- 4. [Fréchet Inception 距离 (FID): 公式](04-Fr%C3%A9chet%20Inception%20%E8%B7%9D%E7%A6%BB%20%28FID%29-%20%E5%85%AC%E5%BC%8F.md)
- 5. [解读 FID 分数](05-%E8%A7%A3%E8%AF%BB%20FID%20%E5%88%86%E6%95%B0.md)
- 6. [分布的准确率与召回率](06-%E5%88%86%E5%B8%83%E7%9A%84%E5%87%86%E7%A1%AE%E7%8E%87%E4%B8%8E%E5%8F%AC%E5%9B%9E%E7%8E%87.md)
- 7. [感知路径长度 (PPL)](07-%E6%84%9F%E7%9F%A5%E8%B7%AF%E5%BE%84%E9%95%BF%E5%BA%A6%20%28PPL%29.md)
- 8. [FID分数计算：实践](08-FID%E5%88%86%E6%95%B0%E8%AE%A1%E7%AE%97%EF%BC%9A%E5%AE%9E%E8%B7%B5.md)
