---
course: "generative-adversarial-networks-gans"
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-5-evaluation-of-gans"
sourceId: 633
chapter: "evaluation-of-gans"
title: "生成对抗网络的定量与定性评估"
order: 5
description: "应用并理解Inception Score (IS)和Fréchet Inception Distance (FID)等定量指标以评估GAN。"
hasQuiz: false
---

评估生成对抗网络（GAN）的性能，相较于典型的监督学习任务，会面临一些特殊的难点。由于GANs学习逼近复杂的数据分布，仅仅查看训练时的损失函数往往无法体现生成样本的真实质量或多样性。判断GAN是否生成逼真输出，并涵盖真实数据的多样性，需要专门的评估方法。

本章介绍用于评估GAN的方法，涵盖定性与定量两类方法。您将了解此评估过程中本身存在的难题。我们将考察视觉检查等定性方法。更为重要的是，我们将侧重于旨在衡量生成性能不同方面的定量指标。您将学习Inception Score (IS)和Fréchet Inception Distance (FID)等常用指标的公式、解释及其局限性。我们还将介绍Precision和Recall等用于比较分布的指标，以及用于评估潜在空间属性的Perceptual Path Length (PPL)。本章包含关于计算和解释这些分数的实用指导，以便比较不同模型和追踪训练进展。
