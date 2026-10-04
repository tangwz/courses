---
course: "advanced-jax"
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-6-large-scale-model-training-jax"
sourceId: 658
chapter: "large-scale-model-training-jax"
title: "大规模模型训练方法"
order: 6
description: "应用进阶JAX进行大规模机器学习。结合pmap、优化和Flax/Haiku等框架，以高效训练大型模型。"
hasQuiz: false
---

训练现代机器学习模型经常会达到单个加速器内存和计算能力的极限。本章着重介绍有效扩展JAX应用所需的方法。

我们将考察如何使用Flax和Haiku等库来构建大型模型，以及如何在这些框架中结合`pmap`实现分布式数据并行。你将学习到一些实用策略，例如梯度累积（模拟更大的有效批量大小）、梯度检查点（`jax.checkpoint`）（以重新计算为代价减少内存使用）以及混合精度训练（进一步节省内存并可能提高速度）。我们还将介绍与模型并行性相关的一些思想，并讨论适用于分布式环境的优化器。

本章结束后，你将明白如何结合这些不同的JAX功能和生态系统工具，以应对训练大型神经网络的挑战。
