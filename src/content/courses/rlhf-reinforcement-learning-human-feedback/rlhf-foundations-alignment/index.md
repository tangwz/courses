---
course: "rlhf-reinforcement-learning-human-feedback"
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-1-rlhf-foundations-alignment"
sourceId: 948
chapter: "rlhf-foundations-alignment"
title: "RLHF与语言模型对齐的初步介绍"
order: 1
description: "回顾与RLHF相关的一些强化学习（RL）和大型语言模型（LLM）主要思想，以及AI对齐的难题。"
hasQuiz: false
---

本章将帮助您开始了解人类反馈强化学习（RLHF）在大型语言模型（LLMs）中的应用。我们首先审视主要难题：让大型语言模型（LLMs）与人类意图和价值观对齐。这项任务是传统监督微调通常无法完全做到的。

您将了解到：

*   AI对齐问题在大型语言模型（LLMs）方面的具体情况。
*   为什么传统监督微调（SFT）不足以完成细致的对齐任务。
*   回顾强化学习（RL）的一些重要思想，特别是对RLHF不可或缺的近端策略优化（PPO）。
*   标准三阶段RLHF流程的初步概览：监督微调（SFT）、奖励模型构建和RL微调。
*   关于如何设置整个课程实践操作所需软件环境和库的指导。

在本章结束时，您将清楚地了解RLHF为什么是必需的以及涉及的基本组成部分，从而为后续章节中详细的实现讨论做好准备。
