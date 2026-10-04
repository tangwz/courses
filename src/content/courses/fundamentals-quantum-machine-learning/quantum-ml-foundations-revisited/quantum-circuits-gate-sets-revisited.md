---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-ml-foundations-revisited"
lesson: "quantum-circuits-gate-sets-revisited"
sourceId: 1034
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-1-quantum-ml-foundations-revisited/quantum-circuits-gate-sets-revisited"
title: "量子电路与通用门集重温"
description: "从与QML算法相关的高级视角回顾量子电路构建和通用门集。"
order: 2
plots: []
sourceHash: "567e8c369d5fb16272abbb1f2b011aa2697fcc530135467e0899bf2e142bfb74"
sourceCorrections: []
---

构建有效的量子机器学习 (machine learning)（QML）模型需要对计算框架有扎实的了解。量子电路由量子门组成，是定义和执行量子算法（包括QML核心算法）的基本结构。本节审视这些核心概念，重点阐述与设计和实现高级QML技术特别相关的方面。

### 量子电路：量子算法的表达方式

回顾一下，量子电路作用于一个量子比特寄存器。它描述了应用于这些量子比特的一系列量子门操作，可能随后进行测量。每个量子比特通常表示为一条水平线，时间从左到右流动。门在特定的时间步对一个或多个量子比特进行作用。

量子态 $|\psi\rangle$ 在一系列门 $U_1, U_2, \dots, U_k$ 作用下的演化通过应用酉算符来描述：
$|\psi_{最终}\rangle = U_k \dots U_2 U_1 |\psi_{初始}\rangle$
整体操作 $U = U_k \dots U_2 U_1$ 本身是一个酉变换，满足 $UU^\dagger = U^\dagger U = I$，其中 $I$ 是单位算符。这种酉性保持量子态的范数，体现了量子力学中固有的概率守恒（测量除外）。

考虑一个基本例子，涉及量子比特0上的哈达玛门（H）和一个受控非（CNOT）门，其中量子比特0是控制比特，量子比特1是目标比特：

> 一个简单的双量子比特电路，对第一个量子比特应用哈达玛门，随后用受控非门作用于第二个量子比特。该电路生成贝尔态 $(|00\rangle + |11\rangle)/\sqrt{2}$。

像受控非门这样的多量子比特门对于在量子比特之间建立纠缠不可或缺。纠缠是一种非经典关联，被认为是一种资源，能在某些量子算法中提供计算优势，可能包括QML模型。像受控非门、SWAP门和三量子比特Toffoli门这样的门是构建更复杂的参数 (parameter)化量子电路（PQCs）的基本元件，这些电路广泛用于变分量子算法（VQAs）和量子神经网络 (neural network)（QNNs）。

### 通用门集

量子计算中一个重要的成果是通用门集的存在。如果任何数量量子比特上的任何酉操作都可以由仅包含该门集中门的量子电路以任意精度近似，则该量子门集被认为是通用的。

通用门集的常见例子有：

1. **Clifford + T：** {哈达玛门 (H), 相位门 (S), $\pi/8$ 门 (T), 受控非门 (CNOT)}
2. **连续旋转 + CNOT：** {H, 相位($\theta$) (任意旋转), 受控非门}

有限通用门集的存在很有用。这意味着任何量子算法，无论其目标酉变换多么复杂，都可以分解为一系列基本、物理上可实现的操作。Solovay-Kitaev 定理提供了理论支持，表明这种近似通常可以高效完成，这意味着所需门的数量随 $\log(1/\epsilon)$ 多项式增长，其中 $\epsilon$ 是所需的近似精度。

从QML的角度看，通用性与模型的*表达能力*相关联。原则上，使用通用门集中的门构建的参数 (parameter)化量子电路（PQC）可以近似任何目标酉变换或生成任何目标量子态，只要有足够的深度和适当的参数。这种能力对于从数据中学习复杂模式或函数非常重要。

### 电路资源：深度与宽度

设计量子电路时，特别是对于近期量子硬件，有两个主要的资源衡量标准：

- **电路宽度：** 电路所需的量子比特数量。
- **电路深度：** 作用于任何单个量子比特的最大门数量，考虑可以并行执行的门层。更正式地说，它是电路有向无环图表示中从输入到输出的最长路径。

当前的噪声中等规模量子（NISQ）设备在宽度（数十到数百个量子比特）和深度（由于有限的相干时间，门会累积误差）上都有限制。这要求开发*硬件高效*的QML算法和电路设计（ansätze），在最小化深度和宽度的同时，仍提供足够的计算能力。

### 程序化表示

Qiskit、Pennylane、Cirq等量子计算框架提供了程序化接口来定义、操作和执行量子电路。电路通常通过实例化电路对象并按顺序附加针对特定量子比特的门操作来构建。

```python
# 伪代码示例（语法因库而异）
from quantum_framework import QuantumCircuit

# 创建一个2量子比特的电路
circuit = QuantumCircuit(2)

# 应用门
circuit.h(0)      # 量子比特0上的哈达玛门
circuit.cx(0, 1)  # 受控非门，控制量子比特0，目标量子比特1

# 添加测量（可选，取决于算法）
circuit.measure_all()

# 在模拟器或量子设备上执行
# results = backend.run(circuit)
```

这种程序化方法对于实现QML算法很重要，其中电路结构可能动态生成或参数 (parameter)化，并且它们的执行是更大规模的混合量子-经典工作流程的一部分（特别是对于VQAs）。

### 在高级QML中的作用

理解量子电路和门集对于本课程后续高级QML主题的学习不可或缺：

- **量子数据编码（第2章）：** 特征映射作为量子电路实现，将经典数据 $x$ 编码为量子态 $|\phi(x)\rangle$。门的选择和电路结构决定了量子特征空间的特性。
- **量子核方法（第3章）：** 核 $k(x_i, x_j) = |\langle\phi(x_i)|\phi(x_j)\rangle|^2$ 通过准备量子态 $|\phi(x_i)\rangle$ 和 $|\phi(x_j)\rangle$ 并测量其重叠的电路进行估计，通常使用哈达玛测试或SWAP测试电路等技术。
- **变分量子算法（第4章）：** VQAs采用参数 (parameter)化量子电路（PQCs） $U(\theta)$，其中门参数 $\theta$ 通过经典方法优化。PQC拟设（电路布局和门选择）的设计对于模型性能和可训练性很重要。
- **量子神经网络 (neural network)（第5章）：** QNNs使用分层PQC结构，与经典神经网络有相似之处。门的选取和排列确定了网络架构。
- **硬件考量（第7章）：** 将复杂电路高效编译为原生硬件门序列以及管理电路深度对于在真实量子处理器上执行很重要。

总之，量子电路是量子计算机的可执行程序。透彻掌握它们的构建、通用门集的特性以及深度和宽度等资源考量，为开发和分析本课程重点介绍的复杂QML算法提供了必要的知识基础。

## 参考资料

- [Quantum computing in the NISQ era and beyond](https://quantum-journal.org/papers/q-2018-08-06-79/) — John Preskill (2018)
  Journal: Quantum; Publisher: Verein zur Förderung des Open Access Publizierens in den Quantenwissenschaften; Volume: 2; Pages: 79; DOI: [10.22331/q-2018-08-06-79](https://doi.org/10.22331/q-2018-08-06-79)
  这篇文章详细说明了噪声中等规模量子（NISQ）设备上量子计算的挑战和机遇，涉及电路深度和宽度。
- [Qiskit Documentation](https://qiskit.org/documentation/) — Qiskit Development Team (2024)
  Publisher: IBM
  Qiskit的官方文档，提供了构建和执行量子电路的实用指南和API参考。
