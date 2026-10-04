---
course: "generative-adversarial-networks-gans"
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-4-conditional-controllable-generation"
sourceId: 631
chapter: "conditional-controllable-generation"
title: "条件式与可控生成"
order: 4
description: "学习使用条件式GAN（cGAN）、InfoGAN以及解耦表示的方法来控制GAN输出。"
hasQuiz: false
---

标准生成对抗网络生成的样本反映训练数据分布，但对具体输出的直接控制有限。本章介绍指导生成过程的方法。您将学习如何实现条件生成对抗网络（cGAN），它能生成与特定属性或标签（通常表示为 $y$）相符的数据。我们将涵盖如何将这些条件信息 $y$ 整合到生成器 $G(z, y)$ 和判别器 $D(x, y)$ 中。

接着我们将学习信息最大化生成对抗网络（InfoGAN），这是一种通过最大化潜在代码 $c$ 与生成器输出之间的互信息 $I(c; G(z, c))$，以无监督方式学习可解释的潜在代码的方法。本章也提及了操控学习到的潜在空间 $z$ 以修改生成后的输出特征的方法，并讨论了像 StackGAN 这样的方法，用于文本到图像合成等任务。最后，我们将考察学到的表示中解耦的原理与度量。动手实践环节提供构建这些条件模型的实践机会。
