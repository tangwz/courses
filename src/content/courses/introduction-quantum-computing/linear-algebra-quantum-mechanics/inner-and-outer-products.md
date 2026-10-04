---
course: "introduction-quantum-computing"
chapter: "linear-algebra-quantum-mechanics"
lesson: "inner-and-outer-products"
sourceId: 8148
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-2-linear-algebra-quantum-mechanics/inner-and-outer-products"
title: "内积与外积"
description: "用于计算概率和投影的数学运算。"
order: 3
plots: ["plots/8148-0.json"]
sourceHash: "18f52c17e244ef1f23fbfea1552c5dde9f15d1ea67ee4eac2f9eae70b0434472"
sourceCorrections: []
---

量子态在希尔伯特空间中表示为向量 (vector)。定义这些向量之间的相互作用是分析量子系统的基础。在经典力学中，通常使用点积来比较向量，从而确定角度或投影。量子力学则使用两种特定的运算来从系统中提取信息或进行变换：**内积**和**外积**。

这些运算是量子算法的计算引擎。内积让我们可以计算概率并检查状态是否不同。外积则让我们可以从向量构建算符（矩阵），这也是我们构建量子门和测量的方式。

## 内积

内积是两个向量 (vector)的标量乘法。在狄拉克符号中，它将左矢 $\langle \psi |$ 和右矢 $| \phi \rangle$ 结合形成括号 $\langle \psi | \phi \rangle$。该运算的结果是一个复数，而不是向量。

在数学上，要计算两个向量 $a$ 和 $b$ 的内积，需要计算第一个向量的共轭转置，并将其与第二个向量相乘。如果我们有两个列向量：


$$
|a\rangle = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix}, \quad |b\rangle = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}
$$


内积 $\langle a | b \rangle$ 为：


$$
\langle a | b \rangle = \begin{bmatrix} a_1^* & a_2^* \end{bmatrix} \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} = a_1^* b_1 + a_2^* b_2
$$


符号 $*$ 表示复共轭。这意味着如果一个元素是 $x + iy$，它会变成 $x - iy$。这种共轭处理是必须的。与实欧几里得空间中的标准点积不同，在复向量空间中，运算顺序非常重要。$\langle a | b \rangle$ 是 $\langle b | a \rangle$ 的复共轭。

### 向量 (vector)乘法的直观理解

内积和外积之间的结构差异取决于乘法顺序。内积将维度“压缩”为单个数值。

> 内积计算中的降维过程。

### 物理意义：概率幅

内积提供了在特定状态下测量系统的**概率幅**。如果你有一个处于状态 $|\psi\rangle$ 的量子比特，并且想知道在状态 $|0\rangle$ 中找到它的概率，你可以计算内积 $\langle 0 | \psi \rangle$。

为了得到实际概率（一个介于 0 和 1 之间的实数），你需要取该结果的模平方：


$$
P = |\langle 0 | \psi \rangle|^2
$$


如果内积为 0，则说明这些状态是**正交的**。这意味着它们互斥。如果系统处于其中一个状态，那么被测量为另一个状态的概率为零。

### Python 实现

在 Python 中，我们使用 NumPy 来处理这些运算。虽然 `np.dot` 适用于实数，但量子力学需要仔细处理复共轭。对于一维数组，通常首选 `np.vdot`，因为它会自动处理第一个参数 (parameter)的复共轭。但在处理显式矩阵（列向量 (vector)）时，我们通常手动定义共轭转置以保持清晰。

```python
import numpy as np

# 定义标准态 |0>
ket_0 = np.array([[1], [0]])

# 定义状态 |+> (叠加态)
# 1/sqrt(2) * [1, 1]
ket_plus = np.array([[1], [1]]) / np.sqrt(2)

# 计算内积 <0|+>
# 步骤 1：创建左矢 (ket_0 的共轭转置)
bra_0 = ket_0.conj().T

# 步骤 2：矩阵乘法
overlap = np.dot(bra_0, ket_plus)

print(f"概率幅: {overlap[0][0]}")
print(f"概率: {abs(overlap[0][0])**2:.2f}")
```

运行此代码会得到大约 0.5 的概率，这证实了状态 $|+\rangle$ 有 50% 的机会坍缩到 $|0\rangle$。

## 外积

内积产生标量，而**外积**产生矩阵（算符）。在狄拉克符号中，这被写为 $| \psi \rangle \langle \phi |$。请注意，右矢在前，左矢在后。

如果采用之前的向量 (vector)，乘法顺序会发生变化：


$$
|a\rangle \langle b| = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} \begin{bmatrix} b_1^* & b_2^* \end{bmatrix} = \begin{bmatrix} a_1 b_1^* & a_1 b_2^* \\ a_2 b_1^* & a_2 b_2^* \end{bmatrix}
$$


结果是一个 $N \times N$ 矩阵。在量子计算中，矩阵代表算符或量子门。外积在构建投影算符和密度矩阵时非常有用，而这些是高级噪声建模所需要的。

### 维度扩展

外积将两个向量 (vector)扩展为矩阵表示。

> 外积计算中的维度扩展。

### 应用场景：投影

外积的一个主要用途是创建**投影算符**。投影算符允许我们过滤一个状态。例如，将通用量子比特状态投影到 $|0\rangle$ 轴上的算符定义为：


$$
P_0 = |0\rangle \langle 0| = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}
$$


当你将此矩阵应用于向量 (vector)时，它会消除与 $|1\rangle$ 相关的分量，仅留下与 $|0\rangle$ 相关的分量。

下图展示了外积 $|+\rangle \langle +|$ 的结果。由于 $|+\rangle$ 是等权重 (weight)叠加态，生成的密度矩阵在所有元素上都具有相等的 0.5 值。



![外积矩阵可视化 (| + >< + |)](plots/8148-0.json)



> 正态（plus state）与其自身外积生成的矩阵热力图。

### Python 实现

我们使用函数 `np.outer` 或标准矩阵乘法来计算外积。请注意，NumPy 中的 `np.outer` 不会自动对第二个向量 (vector)取复共轭，因此如果你不是手动构建左矢，则必须显式传递共轭。

以下是使用标准矩阵乘法逻辑（列 $\times$ 行）的方法：

```python
# 定义状态 |1>
ket_1 = np.array([[0], [1]])

# 创建投影算符 |1><1|
# 我们需要用右矢 ket_1（列）乘以左矢 bra_1（行）
bra_1 = ket_1.conj().T

projector_1 = np.dot(ket_1, bra_1)

print("|1> 的投影算符:")
print(projector_1)
```

## 范数与归一化 (normalization)

量子力学的一个基本公理是概率之和必须为 1。这引出了**归一化**的概念。

为了使 $|\psi\rangle$ 成为一个有效的物理状态，它与自身的内积必须等于 1：


$$
\langle \psi | \psi \rangle = 1
$$


这个量也称为向量 (vector)的 **L2 范数**平方。如果你进行的计算导致所得向量的内积大于或小于 1，则该向量是未归一化的，不代表有效的闭合量子系统。

当你在模拟器中初始化量子比特时，它通常从 $|0\rangle$ 开始。

1. $\langle 0 | 0 \rangle = 1$（已归一化）
2. $\langle 0 | 1 \rangle = 0$（正交）

理解这两个检查对于调试量子电路非常有用。如果本应正交的状态内积不为零，说明相位未对齐 (alignment)。如果自身内积不为 1，说明系统损失了概率（非幺正误差）。

在下一节中，我们将研究**幺正矩阵**，这是一种确保状态向量长度（以及总概率）始终保持为 1 的特殊矩阵类型。
