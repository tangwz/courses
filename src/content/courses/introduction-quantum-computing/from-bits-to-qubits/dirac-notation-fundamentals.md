---
course: "introduction-quantum-computing"
chapter: "from-bits-to-qubits"
lesson: "dirac-notation-fundamentals"
sourceId: 8143
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-1-from-bits-to-qubits/dirac-notation-fundamentals"
title: "狄拉克符号基础"
description: "介绍用于表示量子态的狄拉克符号（bra-ket）。"
order: 3
plots: []
sourceHash: "d289f60bd9474e83e41479bfd863f0d43051a3e984158bcf974fa6706e9e6e0c"
sourceCorrections: []
---

当你在研究论文或文档中遇到量子算法时，第一个难关通常是符号。量子力学使用一种特定的线性代数简写方式，称为狄拉克符号（Dirac notation），或称为左括-右括符号（bra-ket notation）。对于熟悉 Python 和 NumPy 的人来说，这只是表示向量 (vector)和矩阵运算的一种不同语法。

在经典计算中，我们将比特的状态表示为一个标量值，即 0 或 1。在量子计算中，我们将量子比特的状态表示为一个向量。狄拉克符号提供了一种简洁的方式来书写这些向量，而不必频繁绘制矩阵。

### 右括向量 (vector) (Ket Vector)

你最常看到的符号是“右括”（ket），写作一条垂直线，后跟一个标签和一个角括号：$|v\rangle$。

在线性代数中，右括向量代表一个列向量。括号内的标签充当变量名，用于标识状态。当我们讨论量子比特的标准状态时，使用标签“0”和“1”。这些被称为计算基态。

在数学上，状态 $|0\rangle$ 对应于经典比特 0，而 $|1\rangle$ 对应于经典比特 1。然而，它们被定义为单位长度的正交列向量：

$|0\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$

$|1\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$

这种区别非常重要。在 Python 代码中，`0` 是一个整数，但 $|0\rangle$ 是一个形状为 `(2, 1)` 的 NumPy 数组。

> 将狄拉克符号映射到其底层的列向量结构。

### 左括向量 (vector) (Bra Vector)

与右括向量对应的是“左括”（bra），写作 $\langle v|$。如果右括是列向量，那么左括就是行向量。

具体来说，左括是右括的共轭转置（或埃尔米特共轭）。要将右括转换为左括，你需要转置矩阵（交换行和列）并取每个元素的复共轭。

如果我们有一个由复数 $\alpha$ 和 $\beta$ 表示的一般状态 $|\psi\rangle$：

$|\psi\rangle = \begin{bmatrix} \alpha \\ \beta \end{bmatrix}$

那么对应的左括 $\langle \psi |$ 为：

$\langle \psi | = |\psi\rangle^\dagger = \begin{bmatrix} \alpha^* & \beta^* \end{bmatrix}$

这里，$\dagger$ 符号（剑号）表示共轭转置操作，而 $*$ 表示复共轭。由于基向量 $|0\rangle$ 和 $|1\rangle$ 通常只包含实数（0 和 1），复共轭操作不会改变数值，只会将方向从列变为行。

$\langle 0 | = \begin{bmatrix} 1 & 0 \end{bmatrix}$

$\langle 1 | = \begin{bmatrix} 0 & 1 \end{bmatrix}$

### 内积

“bra”和“ket”的名字来源于单词“bracket”（括号）。当你把它们放在一起时，$\langle a | b \rangle$，就形成了一个完整的括号。这个操作代表两个向量 (vector)的内积（或点积）。

在机器学习 (machine learning)中，点积计算一个向量在另一个向量上的投影。在量子力学中，我们使用内积来计算概率并确定两个状态的重叠程度。

计算遵循标准矩阵乘法规则：行向量 ($1 \times 2$) 乘以列向量 ($2 \times 1$) 得到一个标量 ($1 \times 1$)。

例如，计算 $|0\rangle$ 与自身的内积：

$\langle 0 | 0 \rangle = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = (1)(1) + (0)(0) = 1$

结果为 1。这表明该状态已归一化 (normalization)。

如果我们计算 $|0\rangle$ 和 $|1\rangle$ 的内积：

$\langle 0 | 1 \rangle = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = (1)(0) + (0)(1) = 0$

结果为 0。这证实了这些状态是正交的，意味着它们是互斥的结果，可以完全区分开来。

### 外积

内积产生标量，而外积则产生矩阵（算符）。这写作 $|a\rangle\langle b|$。注意顺序：右括（列）在前，左括（行）在后。

将 $2 \times 1$ 向量 (vector)乘以 $1 \times 2$ 向量会得到一个 $2 \times 2$ 矩阵。这一操作是定义量子门和密度矩阵的基础。

例如，外积 $|0\rangle\langle 0|$ 充当投影算符：

$|0\rangle\langle 0| = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 \cdot 1 & 1 \cdot 0 \\ 0 \cdot 1 & 0 \cdot 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$

我们将在第 2 章定义幺正矩阵和量子算符时大量使用外积。

### Python 实现参考

为了将其与日常工作流联系起来，请考虑这些符号如何映射到 NumPy 操作。当你稍后在本章搭建环境时，你将编写反映这些数学对象的代码。

- **右括 (Ket)**: `psi = np.array([[alpha], [beta]])`
- **左括 (Bra)**: `psi_dagger = psi.conj().T`
- **内积 ($\langle \phi | \psi \rangle$)**: `np.dot(phi_dagger, psi)`
- **外积 ($| \psi \rangle \langle \phi |$)**: `np.dot(psi, phi_dagger)`

理解这些底层的线性代数结构有助于你有效地调试量子电路。虽然像 Qiskit 或 Cirq 这样的框架抽象掉了向量 (vector)数学，但系统的行为完全由这些矩阵运算决定。
