---
course: "introduction-quantum-computing"
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-1-from-bits-to-qubits"
sourceId: 1424
chapter: "from-bits-to-qubits"
title: "从比特到量子比特"
order: 1
description: "了解经典比特与量子比特的区别。学习狄拉克符号以及量子信息的基本特性。"
hasQuiz: false
---

经典计算机依赖二进制逻辑，其中信息的基本单位——比特，以 0 或 1 这两种确定状态之一存在。量子计算引入了一种独特的单位，称为量子比特。与非开即关的经典开关不同，量子比特根据量子力学定律运行，在测量发生之前，它可以处于非严格二进制的状态。

本章确立了构建量子系统所需的理论与实际定义。我们首先分析经典二进制处理在物理和数学上的局限性。随后，你将定义量子比特，并学习使用狄拉克符号（左矢-右矢语法）来表示其状态。经典比特的状态是一个标量值，而量子比特的状态 $|\psi\rangle$ 则被描述为复矢量空间中的一个矢量：

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

在这里，$\alpha$ 和 $\beta$ 代表概率幅而非确定的数值。为了直观呈现这些特性，我们将使用布洛赫球，这是一种将量子比特状态映射到单位球面上的几何表示方法。

最后，我们将完成 Python 环境的实际搭建。你将安装必要的软件开发工具包 (SDK) 和库，它们将作为后续章节中模拟量子线路的接口。
