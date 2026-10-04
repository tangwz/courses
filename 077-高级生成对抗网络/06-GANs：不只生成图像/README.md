# 第 6 章：GANs：不只生成图像

来源：[原章节](https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-6-gans-beyond-image-generation)

[返回课程目录](../README.md)

尽管生成对抗网络（GANs）常与生成逼真图像相关联，但其应用已扩展到生成各种其他形式的数据。本章将不再专注于标准图像合成，而是审视GANs如何适应不同的数据模态。

您将了解到在将GANs应用于非图像数据时遇到的具体难题。我们将讨论处理离散序列的方法，例如文本，包括使用强化学习信号和像Gumbel-Softmax技巧这样的连续近似方法。我们还将介绍生成音频波形和频谱图、连贯视频序列、像点云这样的3D数据表示以及像图这样的结构化数据的技术。本章将分析成功地将对抗训练原理应用于这些不同数据类型所需的架构修改和训练策略。

## 小节

- 1. [离散数据带来的难题：文本生成](01-%E7%A6%BB%E6%95%A3%E6%95%B0%E6%8D%AE%E5%B8%A6%E6%9D%A5%E7%9A%84%E9%9A%BE%E9%A2%98%EF%BC%9A%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90.md)
- 2. [强化学习方法 (SeqGAN, RankGAN)](02-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%20%28SeqGAN%2C%20RankGAN%29.md)
- 3. [连续近似（Gumbel-Softmax）](03-%E8%BF%9E%E7%BB%AD%E8%BF%91%E4%BC%BC%EF%BC%88Gumbel-Softmax%EF%BC%89.md)
- 4. [基于GAN的音频合成 (WaveGAN, SpecGAN)](04-%E5%9F%BA%E4%BA%8EGAN%E7%9A%84%E9%9F%B3%E9%A2%91%E5%90%88%E6%88%90%20%28WaveGAN%2C%20SpecGAN%29.md)
- 5. [视频生成与预测](05-%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90%E4%B8%8E%E9%A2%84%E6%B5%8B.md)
- 6. [三维数据生成 (点云, 网格)](06-%E4%B8%89%E7%BB%B4%E6%95%B0%E6%8D%AE%E7%94%9F%E6%88%90%20%28%E7%82%B9%E4%BA%91%2C%20%E7%BD%91%E6%A0%BC%29.md)
- 7. [使用GANs生成图](07-%E4%BD%BF%E7%94%A8GANs%E7%94%9F%E6%88%90%E5%9B%BE.md)
