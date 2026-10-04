---
course: "introduction-quantum-computing"
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-2-linear-algebra-quantum-mechanics"
sourceId: 1425
chapter: "linear-algebra-quantum-mechanics"
title: "量子力学中的线性代数"
order: 2
description: "学习量子计算背后的数学：复数、向量、矩阵和特征值。"
hasQuiz: false
---

量子计算依靠线性代数来描述系统状态以及该状态随时间的变化方式。经典逻辑使用布尔代数，而量子力学则通过向量空间和矩阵来预测结果。本章确立了模拟量子比特行为和构建算法所需的数学工具。

我们首先定义复数（记作 $z = a + bi$）的作用，它们充当量子态的概率幅。你将学会把这些状态表示为希尔伯特空间中的向量。文中解释了如何对这些向量进行运算，特别是使用内积来计算测量结果的概率，以及使用外积来表示算符。

接下来，重点转向量符本身。你将处理酉矩阵，之所以需要它们，是因为它们能保持量子系统的总概率不变。我们还会定义特征值和特征向量，这些数学成分对应于测量过程中观察到的实际数值。例如，如果算符 $A$ 作用于向量 $v$，使得：

$$Av = \lambda v$$

那么标量 $\lambda$ 就代表可测量的量。最后，我们将通过编程应用这些内容。你将使用 Python 的 NumPy 库来生成向量和矩阵，从而为控制量子硬件的数学运算构建一个模拟器。
