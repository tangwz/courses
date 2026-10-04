---
course: "fundamentals-quantum-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-3-quantum-kernel-methods-theory-implementation"
sourceId: 376
chapter: "quantum-kernel-methods-theory-implementation"
title: "量子核方法：理论与实现"
order: 3
description: "详细分析量子核方法、其特性、计算方法及其在QSVM等算法中的应用。"
hasQuiz: false
---

基于将经典数据$x$编码为量子态$|\phi(x)\rangle$的量子特征映射思想，本章主要讲解量子核方法。这些方法通过评估量子特征空间$\mathcal{H}$中与内积$\langle\phi(x)|\phi(x')\rangle$相关的函数，来计算数据点之间的相似度。这种计算构成了量子核$k(x, x')$的组成部分，使经典核函数机器学习算法能够在潜在的高维量子希尔伯特空间中隐式运行。

你将了解定义量子核的形式体系，以及在模拟器和硬件上使用量子电路估算核矩阵元素的方法。我们将考察这些核的数学性质，例如正半定性，并比较其与经典核的几何差异。重要的实践考量，包括高维中核集中现象以及应对策略，将被分析讨论。本章很大一部分内容专门介绍量子支持向量机 (QSVM)，详细说明其操作方式以及在分类问题中的实际应用。实践部分将指导你实现和比较各种量子核方法。
