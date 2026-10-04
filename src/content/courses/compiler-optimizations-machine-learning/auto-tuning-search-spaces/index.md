---
course: "compiler-optimizations-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/compiler-optimizations-machine-learning/chapter-6-auto-tuning-search-spaces"
sourceId: 1388
chapter: "auto-tuning-search-spaces"
title: "自动调优与搜索空间"
order: 6
description: "通过自动调优实现优化自动化。了解编译器中的搜索空间、遗传算法和成本模型。"
hasQuiz: false
---

手动优化代码需要为循环分块、向量化和循环展开选择具体参数。对于单个矩阵乘法核，可能的有效调度方案数量很大。如果一个循环嵌套有 $I, J, K$ 维，并且我们应用分块因子 $t_i, t_j, t_k$，那么仅这些参数的潜在组合可以计算为：

$$ S = |T_i| \times |T_j| \times |T_k| $$

当与循环重排、线程绑定和内存范围决策结合时，总的配置空间呈指数级增长。这使得手动调优对于复杂的神经网络来说不切实际。

本章讨论通过自动调优实现内核生成的自动化。我们将优化搜索空间定义为给定计算图的所有合法变换的集合。你将学习编译器如何定义这些空间，并使用模拟退火和遗传算法等搜索算法来找出高性能调度方案。

在硬件上评估每个候选调度方案会产生瓶颈。为解决这个问题，我们引入了统计成本模型。这些模型使用机器学习预测特定配置的吞吐量，使搜索算法无需硬件执行即可估计性能。本文介绍了AutoTVM和Ansor等现代框架的架构，并对比了基于模板的方法和基于生成的方法。你将学习如何配置这些系统，以自动找到深度学习算子的最优参数。
