---
course: "mixture-of-experts"
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts/chapter-3-moe-training-dynamics-optimization"
sourceId: 899
chapter: "moe-training-dynamics-optimization"
title: "训练动态与优化"
order: 3
description: "解决MoE模型训练中的难题，侧重于负载均衡、路由学习及稳定性。"
hasQuiz: false
---

在确定了专家混合模型的基本原理和架构变体之后，本章将重点介绍如何有效训练这些模型的实际操作。与密集型模型相比，训练稀疏MoE模型会带来特有的难题，主要在于确保专家间的计算均衡以及路由机制的稳定学习。

我们将考察负载均衡这一重要问题，即输入分配不均给专家可能导致效率低下并影响模型性能。您将了解到辅助损失函数，这些函数通常以 $$L_{total} = L_{task} + \alpha L_{aux}$$ 的形式添加到主任务损失中，目的是促进训练期间更均匀的专家利用。我们将介绍 $L_{aux}$ 的常见形式以及调整均衡系数 $\alpha$ 的方法。

此外，还将讨论优化路由或门控网络本身的策略，以确保其能学习到有效的专业分工而不会出现塌陷。我们将处理实际考量，例如如何处理超出专家容量的令牌（“丢弃令牌”），以及诊断专家未能区分其功能的情况。最后，我们将研究优化器和学习率调度等标准选择如何影响MoE训练的稳定性。本章以侧重于负载均衡方法的实践实现练习作为结尾。
