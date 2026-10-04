---
course: "rlhf-reinforcement-learning-human-feedback"
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-3-reward-modeling-human-preferences"
sourceId: 954
chapter: "reward-modeling-human-preferences"
title: "基于人类偏好构建奖励模型"
order: 3
description: "学习训练预测人类偏好的模型，用于AI对齐。"
hasQuiz: false
---

监督微调（SFT）提供了基线，但实现对齐需要更直接的方法来纳入人类对响应质量的判断。本章将重心转向构建奖励模型（$RM$），这是一个重要部分，它学习预测人类偏好哪些AI生成的响应。

您将学习创建这个$RM$的流程，从直接从成对比较中学习的思路开始。我们将介绍收集人类偏好数据的方法、组织这些数据集的方式以及选择合适的模型架构。本章详细介绍常见的训练目标，例如基于Bradley-Terry模型的那些，其公式如下：

$$ P(\text{response}_1 \succ \text{response}_2 | \text{prompt}) = \sigma(RM(\text{prompt}, \text{response}_1) - RM(\text{prompt}, \text{response}_2)) $$

$\sigma$代表sigmoid函数，$\succ$表示偏好。

此外，我们将研究校准奖励模型分数以更好地反映偏好强度的方法，并讨论奖励建模过程中可能遇到的问题，包括数据质量问题以及模型发现意外捷径（奖励欺骗）的风险。在本章结束时，您将明白如何训练一个能量化人类偏好的模型，为强化学习优化做好准备。
