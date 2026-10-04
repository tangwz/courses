---
course: "optimization-techniques-ml"
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-5-distributed-ml-optimization"
sourceId: 440
chapter: "distributed-ml-optimization"
title: "分布式优化策略"
order: 5
description: "了解分布式训练架构、同步方法和通信效率。"
hasQuiz: false
---

随着机器学习模型及其训练数据集的规模和复杂度不断增加，仅依靠单机进行计算会变得低效甚至不可能。训练通常需要将工作负载分散到多个处理器或机器上。本章将介绍专门为这些分布式环境设计的优化算法和策略。

你将了解到与分布式训练相关的动机和挑战。我们将考察常见的架构模式，例如参数服务器模型，它用于协调工作节点间的计算。理解同步和异步更新方案的区别与权衡是本章的侧重点。我们还将探讨通信开销这一重要问题，讨论其中的瓶颈以及梯度压缩和All-Reduce等高效同步模式。最后，我们将提及联邦学习设定下的独特优化考量。我们的目标是让你掌握实现和分析适用于大规模分布式机器学习任务的优化方法的知识。
