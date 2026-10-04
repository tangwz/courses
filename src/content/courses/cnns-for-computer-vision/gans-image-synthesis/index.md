---
course: "cnns-for-computer-vision"
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-7-gans-image-synthesis"
sourceId: 618
chapter: "gans-image-synthesis"
title: "生成对抗网络用于图像合成"
order: 7
description: "学习生成对抗网络 (GAN) 进行图像生成，内容涵盖 DCGAN、条件 GAN、StyleGAN 和训练难题。"
hasQuiz: false
---

本章将从图像内容的分析过渡到使用生成对抗网络（GAN）合成新图像。GAN 提供了一个训练模型的框架，通过生成器 $G$（负责创建数据）和判别器 $D$（试图区分真实数据和生成数据）之间的竞争过程，以生成逼真的输出，通常是图像。

我们将首先回顾 GAN 的基本原理及其对抗性目标函数。随后，我们将讨论模式崩溃和不稳定性等常见的训练难题，以及可能的解决方案。您将学习重要的 GAN 结构，包括深度卷积 GAN (DCGAN)、用于受控生成的条件 GAN (cGAN)，以及 StyleGAN 的基于风格的方法。评估生成图像质量和多样性的方法，例如 Fréchet Inception Distance (FID) 和 Inception Score (IS)，也将涉及。本章包含一个关于实现 DCGAN 的实践练习。
