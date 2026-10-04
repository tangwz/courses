---
course: "deep-learning-regularization-optimization"
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-5-foundational-optimizers"
sourceId: 935
chapter: "foundational-optimizers"
title: "基本优化算法"
order: 5
description: "理解随机梯度下降（SGD）、小批量梯度下降，以及动量和Nesterov加速梯度（NAG）等变体。"
hasQuiz: true
---

在研究了通过正则化控制模型复杂度的方法之后，我们现在关注为深度学习模型寻找最优参数的过程。这个被称为优化的过程，是有效训练神经网络的核心。标准梯度下降提供了理论依据，但它应用于大型数据集时面临实际问题。

本章介绍深度学习中使用的基本优化算法。我们首先回顾标准梯度下降，并讨论它的局限性。然后您将了解：

*   **随机梯度下降（SGD）：** 一种计算高效的近似方法，使用单个数据点更新参数。
*   **小批量梯度下降：** 一种广泛使用的方法，平衡了SGD和批量梯度下降的优点。
*   **动量：** 一种加速SGD收敛的技术，特别是在梯度方向一致时，并抑制振荡。
*   **Nesterov加速梯度（NAG）：** 对动量的一种改进，通常能带来更快的收敛。

在本章结束时，您将理解这些核心算法的运作原理，它们各自的优缺点，以及它们在模型训练过程中如何应对复杂损失曲面带来的挑战。我们还将在实践中实现并比较这些优化器。
