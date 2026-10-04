---
course: "fine-tuning-adapting-large-language-models"
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-5-advanced-fine-tuning-strategies"
sourceId: 673
chapter: "advanced-fine-tuning-strategies"
title: "高阶微调策略"
order: 5
description: "涵盖多任务微调、序列式调整、人类反馈强化学习 (RLHF) 基本原理，以及灾难性遗忘的缓解方法。"
hasQuiz: false
---

本章在模型运用全参数更新和高效技术进行调整的方法之上，侧重于更复杂的微调场景。

您将学习到在一个模型上训练多个目标（多任务微调）以及按顺序更新模型并管理知识保留（序列式调整）的策略。

我们将介绍旨在应对灾难性遗忘的方法，帮助模型在引入新数据或新目标时，保持在先前学习过的任务上的性能。此外，本节介绍人类反馈强化学习 (RLHF)。我们将阐述其主要组成部分：训练奖励模型，$r_\phi(x, y)$，以获取偏好，以及使用策略优化算法，例如近端策略优化 (PPO)，根据这些偏好调整语言模型，$\pi_\theta(y|x)$。这些技术是调整大型语言模型行为以满足特定需求的进阶方法。
