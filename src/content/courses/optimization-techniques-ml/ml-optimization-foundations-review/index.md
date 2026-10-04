---
course: "optimization-techniques-ml"
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-1-ml-optimization-foundations-review"
sourceId: 432
chapter: "ml-optimization-foundations-review"
title: "机器学习优化的根本原则"
order: 1
description: "回顾梯度下降变体、凸性、损失曲面、收敛性分析以及非凸优化中的挑战。"
hasQuiz: false
---

优化算法是机器学习模型训练的主要计算部分。在介绍复杂方法之前，本章将回顾构建这些方法所依赖的必要原理。我们将首先回顾标准的一阶优化算法，例如随机梯度下降（SGD）、动量法（Momentum）和涅斯捷罗夫加速梯度法（NAG），以此设定一个用于比较的基准。

理解目标函数的数学性质非常重要。我们将审视凸性这一思想。对于函数 $f$ 及其定义域内的任意 $x, y$ 和 $\lambda \in [0, 1]$，若条件 $$f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$$ 成立，我们将讨论为何此性质能简化优化过程。我们也将了解在机器学习中遇到的损失曲面的典型几何形态，特别是在高维空间中。

衡量评估优化器的指标将通过收敛分析的基本情况来介绍，并查看不同的收敛速度。我们将指出出现的常见困难，特别是在深度学习中常见的非凸优化问题中，例如局部最小值和鞍点。最后，将讨论关于数值稳定性以及浮点运算的潜在影响的实际考虑因素。本章最后附有侧重于分析这些根本性算法行为的练习。
