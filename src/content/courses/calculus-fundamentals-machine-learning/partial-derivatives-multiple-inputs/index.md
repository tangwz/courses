---
course: "calculus-fundamentals-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-4-partial-derivatives-multiple-inputs"
sourceId: 573
chapter: "partial-derivatives-multiple-inputs"
title: "处理多输入：偏导数"
order: 4
description: "学习多变量函数、偏导数和梯度向量。"
hasQuiz: true
---

大多数机器学习模型处理带有多个特征的数据，或需要调整大量参数。与我们迄今研究的单变量函数 $f(x)$ 不同，实际模型通常更像是 $f(x_1, x_2, \dots, x_n)$。当一个函数依赖于多个输入时，我们需要方法弄明白它仅当*某个*输入变化时是如何变化的。

本章将介绍做到这一点的方法：**偏导数**。你将学到：

*   如何理解和处理多变量函数。
*   偏导数的定义和计算，使用诸如 $\frac{\partial f}{\partial x}$ 的符号。
*   如何将偏导数组合成**梯度向量** ($\nabla f$)。
*   梯度的几何意义，以及它在指明函数最陡峭变化方向方面的作用。

这些知识将我们对导数的理解扩展到机器学习优化中常见的多维情况。
