# 第 1 章：生成模型简介

来源：[原章节](https://apxml.com/zh/courses/intro-diffusion-models/chapter-1-generative-modeling-fundamentals)

[返回课程目录](../README.md)

生成模型学习生成与给定数据集相似的新数据样本。本章提供所需知识，以便后续专门讲解扩散模型。

我们将首先简要回顾常见的生成模型类型，如变分自编码器（VAEs）和生成对抗网络（GANs），以了解它们的目标和运行机制。然后，我们将讨论这些模型面临的难题，说明促成扩散技术发展的原因。

扩散模型的核心思想包含两个过程：系统地向数据添加噪声直至其变为纯噪声，然后学习反转此过程，从噪声开始生成数据。我们将介绍这一核心思想。最后，我们将建立用于描述这些模型的高级概率框架，为您后续章节的数学细节做好准备。

完成本章后，您将了解扩散模型的运行背景，并掌握其基本运行原理。

## 小节

- 1. [生成模型概述](01-%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0.md)
- 2. [扩散模型的缘由](02-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E7%9A%84%E7%BC%98%E7%94%B1.md)
- 3. [核心思路：噪声与去噪](03-%E6%A0%B8%E5%BF%83%E6%80%9D%E8%B7%AF%EF%BC%9A%E5%99%AA%E5%A3%B0%E4%B8%8E%E5%8E%BB%E5%99%AA.md)
- 4. [概率框架概述](04-%E6%A6%82%E7%8E%87%E6%A1%86%E6%9E%B6%E6%A6%82%E8%BF%B0.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intro-diffusion-models/chapter-1-generative-modeling-fundamentals/quiz)
