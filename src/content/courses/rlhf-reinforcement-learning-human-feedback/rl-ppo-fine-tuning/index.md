---
course: "rlhf-reinforcement-learning-human-feedback"
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-4-rl-ppo-fine-tuning"
sourceId: 958
chapter: "rl-ppo-fine-tuning"
title: "使用近端策略优化（PPO）进行强化学习微调"
order: 4
description: "关于使用PPO优化语言模型以符合学到奖励模型的详细指南。"
hasQuiz: false
---

在确定了模拟人类偏好的方法之后，下一步是使用这个信号直接改进语言模型的表现。本章主要关注强化学习（RL）的微调阶段，特别是采用近端策略优化（PPO）。PPO 是一种策略梯度方法，常用于 RLHF 中，以优化语言模型策略，使其符合学到的奖励模型，同时确保不会与原始的监督微调模型偏离过远。

您将学习如何在大型语言模型的环境中应用 PPO 算法。我们将分析策略网络和价值网络的设置，KL散度惩罚（$D_{KL}$）对于训练稳定的重要作用，计算广义优势估计（GAE）等优势值的方法，以及超参数调整的实际考量。我们还将查看使用 Hugging Face 的 TRL 等库进行实现的例子，并讨论训练不稳定等常见挑战。目标是让您掌握实现和管理 RLHF 流程中基于 PPO 的优化阶段的知识。
