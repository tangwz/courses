---
course: "introduction-quantum-computing"
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-3-single-qubit-gates-superposition"
sourceId: 1426
chapter: "single-qubit-gates-superposition"
title: "单比特门与叠加态"
order: 3
description: "掌握单量子比特门。学习泡利门、哈达玛门以及叠加态的构建。"
hasQuiz: false
---

到目前为止，我们一直将量子比特视为希尔伯特空间内的静态向量。为了进行计算，我们必须操作这个状态。正如经典计算机使用逻辑门处理比特一样，量子系统使用酉算子来变换量子比特。本章集中讲解这些被称为量子门的各种操作，以及它们如何改变状态向量的概率振幅。

我们首先确立叠加态的数学定义。经典比特必须处于确定的状态，而量子比特则可以作为标准基矢 $|0\rangle$ 和 $|1\rangle$ 的线性组合存在。您将学习使用以下等式来表示这种状态：

$$ |\psi\rangle = \alpha|0\rangle + \beta|1\rangle $$

随后，教材将涵盖单比特操作的标准库。我们会分析泡利门（$X$、$Y$ 和 $Z$），它们对应于绕布洛赫球坐标轴的旋转。接着，我们将介绍哈达玛门（Hadamard gate），这是从基态产生叠加态的主要算子。

最后，我们将处理测量的机制。您将看到观察量子系统如何迫使状态向量坍缩为单一结果。为了巩固理论知识，您将编写 Python 代码来初始化量子比特、应用门操作，并在本地环境中模拟测量结果。
