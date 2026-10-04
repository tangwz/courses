---
course: "synthetic-data-gans-diffusion"
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-5-evaluating-synthetic-data-quality"
sourceId: 1040
chapter: "evaluating-synthetic-data-quality"
title: "合成数据质量评估"
order: 5
description: "学习用于严谨评估生成数据质量的进阶指标和技术。"
hasQuiz: false
---

生成合成数据只是第一步。训练好GAN或扩散模型后，如何判断生成的样本质量是否过关呢？仅仅查看几个例子可能会产生误导。本章侧重于严谨衡量生成模型所产合成数据的质量、多样性和逼真度所需的方法。

你将了解评估生成模型时固有的难点，例如简单指标如准确率并不适用。我们将介绍既定的定量指标，如Inception Score (IS) 和 Fréchet Inception Distance (FID)，并理解它们的计算方式和解读方法。你还将研究像Kernel Inception Distance (KID) 这样的分布指标，以及GAN特有的指标，例如衡量潜在空间质量的Perceptual Path Length (PPL)。

除了自动化指标之外，我们还将讨论定性评估技术，以及评估基于条件（例如类别标签）生成数据的模型时的具体考量。最后，你将获得编写代码计算FID的动手实践经验，FID是一种广泛使用的标准指标。本章结束时，你将拥有一个能够有效评估生成模型输出结果的工具集。
