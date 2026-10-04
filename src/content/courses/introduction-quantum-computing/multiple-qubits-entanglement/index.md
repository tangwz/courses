---
course: "introduction-quantum-computing"
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-4-multiple-qubits-entanglement"
sourceId: 1427
chapter: "multiple-qubits-entanglement"
title: "多量子比特与纠缠"
order: 4
description: "学习张量积、CNOT 门以及贝尔态的生成。理解量子纠缠。"
hasQuiz: false
---

前面的章节主要讲解了单个量子比特的特性和操作。尽管叠加和酉演化等原理非常基本，但单个量子比特的计算用途有限。为了执行复杂的算法，我们需要将多个量子比特组合成一个系统。这需要一套数学框架，用来描述随着比特增加而扩大的状态空间。

在本模块中，我们将学习张量积（记作 $\otimes$）。这一运算让我们能够将独立的向量空间合并成更大的复合空间。对于两个量子比特，基态从标准的 $|0\rangle$ 和 $|1\rangle$ 扩展为四种组合：$|00\rangle$、$|01\rangle$、$|10\rangle$ 和 $|11\rangle$。你会看到状态向量的大小是如何按照 $2^n$（其中 $n$ 是量子比特的数量）增长的。

有了多个量子比特，我们就可以加入多比特门。这里的重点是控制非门（CNOT）。与旋转状态向量的单比特门不同，CNOT 门具备条件逻辑，仅在控制比特为 $|1\rangle$ 时才翻转目标比特。这种关联是产生纠缠的机制。

我们将纠缠定义为这样一种情形：复合系统的量子态无法分解为每个量子比特的独立状态。当量子比特处于纠缠态时，对其中一个的测量会与另一个的测量结果相关联。你将通过构建贝尔态（即特定的最大纠缠对）来进行数学分析。

学完本章后，你将能够：

*   使用张量积计算复合系统的状态向量。
*   应用 CNOT 门来控制量子比特之间的关联。
*   通过数学方式验证一个状态是纠缠态还是可分态。
*   编写 Python 代码来生成并测量贝尔态。
