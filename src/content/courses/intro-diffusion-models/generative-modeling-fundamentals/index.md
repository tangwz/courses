---
course: "intro-diffusion-models"
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-1-generative-modeling-fundamentals"
sourceId: 987
chapter: "generative-modeling-fundamentals"
title: "生成模型简介"
order: 1
description: "回顾生成模型（VAEs，GANs），并介绍扩散模型的核心思想作为一种替代方法。"
hasQuiz: true
---

生成模型学习生成与给定数据集相似的新数据样本。本章提供所需知识，以便后续专门讲解扩散模型。

我们将首先简要回顾常见的生成模型类型，如变分自编码器（VAEs）和生成对抗网络（GANs），以了解它们的目标和运行机制。然后，我们将讨论这些模型面临的难题，说明促成扩散技术发展的原因。

扩散模型的核心思想包含两个过程：系统地向数据添加噪声直至其变为纯噪声，然后学习反转此过程，从噪声开始生成数据。我们将介绍这一核心思想。最后，我们将建立用于描述这些模型的高级概率框架，为您后续章节的数学细节做好准备。

完成本章后，您将了解扩散模型的运行背景，并掌握其基本运行原理。
