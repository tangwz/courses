---
course: "llm-alignment-safety"
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-2-reinforcement-learning-human-feedback-rlhf"
sourceId: 683
chapter: "reinforcement-learning-human-feedback-rlhf"
title: "人类反馈强化学习 (RLHF)"
order: 2
description: "详细研究人类反馈强化学习 (RLHF)，旨在对齐大型语言模型。"
hasQuiz: false
---

本章讨论人类反馈强化学习 (RLHF)，这是一种使大型语言模型更符合人类意图的技术。我们将剖析标准的 RLHF 流程，从人类偏好数据如何收集和准备说起。

您将学习奖励模型的训练过程，该模型通常表示为 $r_\theta(x, y)$，旨在根据收集到的偏好对输出进行评分。接下来，我们将介绍该奖励模型如何指导大型语言模型策略（记作 $\pi_\phi(y|x)$）通过诸如近端策略优化 (PPO) 等强化学习算法进行微调。

各章节将详述奖励建模的架构、损失函数以及诸如模型校准等常见难题。我们还将研究 PPO 在大型语言模型中的具体实现，包括超参数调整和稳定性分析。最后，我们将讨论 RLHF 的局限性，并提供一个侧重于实现该过程主要部分的实践练习。
