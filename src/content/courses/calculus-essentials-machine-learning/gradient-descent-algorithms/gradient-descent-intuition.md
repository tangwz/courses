---
course: "calculus-essentials-machine-learning"
chapter: "gradient-descent-algorithms"
lesson: "gradient-descent-intuition"
sourceId: 1386
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-4-gradient-descent-algorithms/gradient-descent-intuition"
title: "梯度下降的直观原理"
description: "了解如何使用梯度迭代地趋向函数最小值的主要思路。"
order: 1
plots: ["plots/1386-0.json"]
sourceHash: "0ba04a86ff142cde9f642d1f01f3d7260c25988b2680987ab3bf0d34877291b8"
sourceCorrections: []
---

梯度$\nabla J(\theta)$指向成本函数$J(\theta)$增长最快的方向。然而，我们在机器学习 (machine learning)优化中的目标通常相反：我们希望*最小化*成本函数，以找到能提供最佳表现的模型参数 (parameter)$\theta$。那么，梯度如何能帮助我们下坡而不是上坡呢？

想象你正站在一个山坡上，周围浓雾弥漫。你看不见谷底（最小值），但能感觉到脚下地面的坡度。要下山，最直观的做法是确定最陡峭的下降方向，然后朝那个方向迈出一步。你重复这个过程，不断检查坡度并向下迈步，希望能最终到达底部。

梯度下降 (gradient descent)的工作原理与此非常相似。在我们的参数空间中，任何点$\theta$处可以测量的“坡度”由梯度$\nabla J(\theta)$给出。由于梯度指向上坡（最陡峭的*上升*方向），因此最陡峭的*下降*方向就是其反方向：$-\nabla J(\theta)$。

因此，梯度下降的核心思想是，从参数$\theta$的某个初始猜测开始，然后通过朝着负梯度方向迈出小步来迭代更新它们。理想情况下，每一步都应使我们更接近成本函数$J(\theta)$最小的点。

设想一个依赖于两个参数$\theta_1$和$\theta_2$的简单成本函数。我们可以将此函数可视化为一个曲面，或者更常见地，使用等高线来表示，其中每条线代表成本相同的点。



![等高线图上的梯度下降路径](plots/1386-0.json)



> 一张显示成本函数$J(\theta_1, \theta_2)$等高线的等高线图。红线描绘了梯度下降所走的步长，从一个初始点开始，并朝着最小值（绿色“x”）移动，始终垂直于等高线（即沿着负梯度的方向）。

每一步都包含两个部分：

1. **计算方向**：在当前的参数值处找到负梯度$-\nabla J(\theta)$。这告诉我们最陡峭的下降方向。
2. **迈出一步**：通过在该方向上移动一定距离来更新参数。

每次迭代中迈出步长的大小是一个重要因素。如果步长太大，我们可能会越过最小值并出现不规则的震荡，甚至可能发散。如果步长太小，算法可能需要很长时间才能达到最小值。这个步长由一个称为*学习率*的参数控制，我们很快将详细讨论它。

本质上，梯度下降提供了一种自动化的方式，来沿着成本函数的“山坡”走下去。通过重复计算局部坡度（梯度）并朝着向下方向迈出一步，它在参数空间中移动以找到使成本$J(\theta)$最小的值$\theta$，从而优化机器学习模型。

## 参考资料

- [CS229 Lecture Notes: Supervised Learning, Generative/Discriminative Learning](http://cs229.stanford.edu/notes/cs229-notes1.pdf) — Andrew Ng (updates by Tengyu Ma) (2018)
  Journal: Stanford University; Publisher: Stanford University; Pages: 3-8
  在机器学习背景下，对梯度下降提供了清晰易懂的介绍。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 83-88
  一本基础教材，涵盖了梯度下降作为核心优化方法的原理。
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; Pages: 93-98
  将包括梯度在内的数学概念直接与其在梯度下降等机器学习算法中的应用联系起来。可在线免费获取。
