---
course: "intro-normalizing-flows"
sourceUrl: "https://apxml.com/zh/courses/intro-normalizing-flows/chapter-3-autoregressive-flows"
sourceId: 1450
chapter: "autoregressive-flows"
title: "自回归流"
order: 3
description: "构建用于生成模型的自回归流，内容涵盖掩码自编码器、MAF 和逆自回归流。"
hasQuiz: false
---

在上一章中，我们通过叠加简单的数学变换构建了基本的归一化流。虽然平面流和径向流具有一定功能，但在处理高维分布时往往难以扩展。为了解决这一限制，我们转而使用自回归模型。

自回归模型将联合概率分布分解为一系列条件概率的乘积。从数学表达上看，$n$ 维变量 $x$ 的联合概率 $p(x)$ 可以写成：

$$p(x) = \prod_{i=1}^{n} p(x_i | x_1, x_2, ..., x_{i-1})$$

通过强制执行这种结构，我们能确保概率分布的有效性，并自然地生成下三角雅可比（Jacobian）矩阵。下三角雅可比矩阵使行列式计算变得简单且高效，将时间复杂度从 $O(D^3)$ 降低到 $O(D)$。

本章主要介绍自回归流的设计与实现。你将学习用于分布估计的掩码自编码器（MADE）架构，该架构利用二进制掩码来强制执行自回归特性，而无需依赖缓慢的顺序循环。随后，你将对比两种标准架构：掩码自回归流（MAF）和逆自回归流（IAF）。MAF 针对快速密度评估进行了优化，使其在精确最大似然估计方面表现出色。IAF 则针对快速采样速度进行了优化，更适合生成合成数据。

最后，你将通过编写 Python 和 PyTorch 代码来实现掩码线性层，从而构建出你自己的自回归流模型。
