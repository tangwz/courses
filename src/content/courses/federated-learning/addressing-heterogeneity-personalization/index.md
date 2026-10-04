---
course: "federated-learning"
sourceUrl: "https://apxml.com/zh/courses/federated-learning/chapter-4-addressing-heterogeneity-personalization"
sourceId: 740
chapter: "addressing-heterogeneity-personalization"
title: "处理异质性和个性化"
order: 4
description: "掌握处理统计（非独立同分布）和系统异质性的方法，并应用个性化联邦学习模型。"
hasQuiz: false
---

标准联邦学习通常基于一些简化假定运行，比如客户端数据同分布和计算资源相似。实际部署中这些假定往往不符合。客户端数据集通常表现出统计异质性（即非独立同分布数据），这表示客户端k的本地数据分布$P_k(x, y)$可能与全局分布$P(x, y)$差异很大。同时，客户端硬件、网络速度和可用性的差异也造成了系统异质性。这两种问题都可能阻碍收敛、降低模型准确性，并导致公平性问题。

本章介绍用于应对这些常见难题的技术。我们将研究旨在减轻非独立同分布数据和系统变动性负面影响的方法。认识到异质性意味着单一全局模型可能无法最好地服务所有客户端，我们也将学习联邦学习中的个性化方法。你会了解到包括以下方面的策略：

*   处理跨客户端的不同数据分布。
*   运用聚类联邦学习将相似客户端分组。
*   应用元学习原则以实现高效的客户端特定模型调整。
*   运用多任务学习框架组织问题。
*   调整模型以适应不同的设备能力。
