---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-26-reinforcement-learning-human-feedback-rlhf"
sourceId: 1104
chapter: "reinforcement-learning-human-feedback-rlhf"
title: "人类反馈强化学习 (RLHF)"
order: 26
description: "实现人类反馈强化学习流程，包括奖励模型构建和强化学习微调，从而使大型语言模型与人类偏好保持一致。"
hasQuiz: false
---

监督微调（SFT）使模型能够遵循指令，但要使其与人类在助益性、诚实性和无害性等方面的更为复杂的价值取向对齐，通常需要额外的步骤。本章介绍人类反馈强化学习（RLHF），这是一种借助人类对输出质量的评估来微调语言模型的方法。

我们将考察标准的RLHF流程，从收集成对偏好数据开始。您将学习训练一个奖励模型（$RM$），该模型能够预测人类可能更喜欢哪种模型生成的内容。在此之后，我们将详细说明如何在强化学习框架（通常是近端策略优化PPO）中，将此$RM$用作奖励信号，以调整SFT模型的表现。我们将阐释Kullback-Leibler（$KL$）散度惩罚在稳定强化学习过程中的作用，并简要概述直接偏好优化（DPO）等其他方法。在本章末尾，您将理解用于大型语言模型对齐的RLHF流程的各个构成和具体实现方式。
