---
course: "meta-learning-foundation-models"
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-2-advanced-gradient-based-meta-learning"
sourceId: 846
chapter: "advanced-gradient-based-meta-learning"
title: "高级基于梯度的元学习"
order: 2
description: "细致分析MAML变体及其他基于梯度的元学习算法的可扩展性。"
hasQuiz: false
---

本章着重介绍基于梯度的元学习方法，这是一类旨在学习模型初始化或更新规则、以利于快速适应的主要算法。其主要思想是优化参数 $\theta$，使得在新任务支持集上经过少量梯度步长后，能在其查询集上获得良好性能。

我们将首先细致考察模型无关元学习 (MAML)，介绍其优化目标及所涉及的计算方面。接着，我们分析计算高效的近似方法，例如一阶MAML (FOMAML) 和 Reptile，比较它们的机制和性能表现。隐式MAML (iMAML) 将作为一种替代方法进行介绍，它可能带来稳定性与内存方面的益处。我们还会讨论优化稳定性、梯度方差等常见问题，并介绍缓解这些问题的方法。最后，我们将考量如何将这些基于梯度的方法扩展到大型基础模型的具体挑战和应对策略，并以一个聚焦于FOMAML的实操练习作结。
