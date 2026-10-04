---
course: "calculus-fundamentals-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-5-calculus-action-optimization"
sourceId: 576
chapter: "calculus-action-optimization"
title: "微积分的运用：简单优化"
order: 5
description: "了解微积分原理如何在一个简单的机器学习优化场景中得到运用。"
hasQuiz: true
---

学完导数、优化原理和梯度后，我们现在将这些知识点与一个实际的机器学习场景关联起来。本章将说明微积分如何通过优化来辅助模型训练过程。

在此，您将学习如何：
*   运用代价函数的原理到一个简单的线性模型。
*   计算代价函数相对于模型参数的梯度。
*   观察这些梯度如何在梯度下降的步骤中被用于更新参数。
*   理解学习率在此过程中的作用。

我们将通过一个使用简单线性回归（$y = mx + b$）的例子来学习。您将看到，计算代价函数的偏导数如何使我们能够系统地调整 $m$ 和 $b$ 以最小化误差，这说明了训练许多机器学习模型背后的主要运作方式。
