---
course: "intro-normalizing-flows"
sourceUrl: "https://apxml.com/zh/courses/intro-normalizing-flows/chapter-4-coupling-architectures"
sourceId: 1451
chapter: "coupling-architectures"
title: "耦合架构"
order: 4
description: "利用仿射耦合层和多尺度设计，实现 RealNVP 和 Glow 等耦合架构。"
hasQuiz: false
---

自回归流提供了一种对概率分布建模的有效方法，但它们存在结构上的局限性。由于按顺序处理数据，这类模型通常在一个方向上运行很快，而在另一个方向上则很慢。如果模型生成数据的速度很快，那么它计算概率密度的速度往往会变慢。

为了在正向和反向过程中都获得高效的性能，我们采用耦合架构。这些设计将输入数据分为两部分。第一部分直接通过该层而不发生改变。然后，模型使用这一未改变的部分来确定第二部分变换所需的参数。

以输入向量 $x$ 为例，将其分为 $x_1$ 和 $x_2$ 两部分。仿射耦合层通过以下运算得出输出 $y$：

$$ y_1 = x_1 $$

$$ y_2 = x_2 \odot \exp(s(x_1)) + t(x_1) $$

由于前半部分保持不变，该变换的雅可比矩阵是三角矩阵。无论神经网络 $s$ 和 $t$ 的构成多么复杂，这一特性都能让你几乎瞬间算出行列式并完成逆变换。

在本章中，你将学习如何构建这些计算效率极高的模型。你将分析 RealNVP 架构，重点了解它如何使用棋盘格掩码和通道掩码来处理图像的不同部分。你还将研究多尺度架构，即通过在不同阶段丢弃变量来降低计算成本。随后，你将学习 Glow 模型，了解激活归一化（Activation Normalization）和 $1 \times 1$ 可逆卷积如何提升训练稳定性和模型表现。

在本章末尾，你将通过 PyTorch 从零开始构建仿射耦合层，在 Python 代码中应用这些数学原理。
