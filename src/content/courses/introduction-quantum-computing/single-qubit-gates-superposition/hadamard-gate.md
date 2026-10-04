---
course: "introduction-quantum-computing"
chapter: "single-qubit-gates-superposition"
lesson: "hadamard-gate"
sourceId: 8154
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-3-single-qubit-gates-superposition/hadamard-gate"
title: "Hadamard 门"
description: "使用 Hadamard 门从基态创建叠加态。"
order: 3
plots: []
sourceHash: "ad04647df1d49e225c72c964b1135d6077d1bdeedba77ffcd90587411206f84e"
sourceCorrections: []
---

Pauli 门提供了在量子系统中翻转比特和旋转相位的能力。然而，这些操作通常将标准基态直接映射到其他标准基态。例如，从 $|0\rangle$ 开始并应用 $X$ 门会得到 $|1\rangle$。这种行为模仿了经典逻辑运算。为了获得量子系统的计算能力，有必要离开这些确定状态并进入叠加态。实现这一点的常用工具是 Hadamard 门，记作 $H$。

Hadamard 门可以说是最实用的单量子比特门。它将确定状态转换为测量 $0$ 或 $1$ 的概率相等的叠加态。如果你想到一枚经典的硬币，它总是正面或反面。Hadamard 门就像是让硬币在边缘立住的投掷动作，在测量迫使其倒下之前，它在两种结果之间保持完美的平衡。

### 数学定义

当 Hadamard 门作用于标准基态 $|0\rangle$ 时，它会产生一个 $|0\rangle$ 和 $|1\rangle$ 的概率振幅相等的状态。我们使用以下等式定义此操作：

$H|0\rangle = \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle$

你可能会好奇为什么我们使用系数 $\frac{1}{\sqrt{2}}$。回顾之前的章节，概率是通过对振幅的模进行平方来计算的。为了确保总概率之和为 1 (100%)，我们必须满足：

$\left|\frac{1}{\sqrt{2}}\right|^2 + \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2} + \frac{1}{2} = 1$

我们将得到的状态称为“正态”（plus state），记作 $|+\rangle$。

$|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$

当 Hadamard 门作用于状态 $|1\rangle$ 时，它会产生类似的叠加态，但在相位上有一个明显的区别。这被称为“负态”（minus state），或 $|-\rangle$。

$H|1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}} = |-\rangle$

请注意负号。如果你计算负态的概率，你仍然会得到 $|\frac{-1}{\sqrt{2}}|^2 = \frac{1}{2}$。$|+\rangle$ 和 $|-\rangle$ 都有 50% 的几率被测量为 0，50% 的几率被测量为 1。区别在于矢量之间的相对相位，当这些量子比特随后与电路中的其他门交互时，这一点会变得很有意义。

### 矩阵表示

在线性代数中，Hadamard 门由一个特定的酉矩阵表示。我们可以将 $H$ 写为：

$H = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$

这个矩阵允许我们计算任何输入状态矢量的输出。如果我们将此矩阵应用于 $|0\rangle$ 的矢量表示（即 $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$），我们可以通过线性代数进行推导：

$H|0\rangle = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$

结果是一个矢量，其中顶部分量对应 $|0\rangle$，底部分量对应 $|1\rangle$，两者都带有 $\frac{1}{\sqrt{2}}$ 的缩放因子。

### 变换的可视化

从几何角度观察这种变换很有帮助。在布洛赫球上，基态 $|0\rangle$ 和 $|1\rangle$ 分别位于 Z 轴上的北极和南极。Hadamard 门执行一次旋转，将这些状态移动到赤道（X 轴）。

具体来说，Hadamard 门绕 Y 轴旋转 90 度（$\pi/2$ 弧度），然后绕 X 轴旋转 180 度（$\pi$ 弧度）。一种更简单的方法是将其看作绕 X 轴和 Z 轴之间的一根对角轴进行单次 180 度旋转。

下图展示了从基态到叠加态的转换过程。

> Hadamard 门将 0 态映射到正态，将 1 态映射到负态。

### 可逆性

Hadamard 门的一个特性是它是其自身的逆。在量子力学中，操作必须是可逆的（酉变换）。如果你连续两次应用 Hadamard 门，你会回到原始状态。

$H(H|0\rangle) = H(|+\rangle) = |0\rangle$

$H(H|1\rangle) = H(|-\rangle) = |1\rangle$

从数学上讲，这意味着 $H^2 = I$，其中 $I$ 是单位矩阵。这个特性允许我们撤销叠加。在许多量子算法中，我们首先通过应用 Hadamard 门来初始化叠加态，进行计算，然后再次应用 Hadamard 门通过干涉将状态变回可测量的基态。

### 在 Python 中实现 H 门

为了将理论付诸实践，我们可以使用 NumPy 模拟 Hadamard 门的矩阵力学。这有助于巩固量子电路模拟器背后运行的线性代数逻辑。

我们定义 $|0\rangle$ 的态矢量，创建 Hadamard 矩阵，并计算点积。

```python
import numpy as np

# 定义标准基态 |0>
ket_0 = np.array([[1], 
                  [0]])

# 定义 Hadamard 矩阵
# 1/sqrt(2) * [[1, 1], [1, -1]]
H_gate = (1 / np.sqrt(2)) * np.array([[1, 1], 
                                      [1, -1]])

# 使用矩阵乘法（点积）应用门操作
superposition_state = np.dot(H_gate, ket_0)

print("输入状态:\n", ket_0)
print("\nHadamard 矩阵:\n", H_gate)
print("\n结果状态 (|+>):\n", superposition_state)
```

运行此代码可以得到 $|+\rangle$ 状态的数值结构：

$\begin{bmatrix} 0.70710678 \\ 0.70710678 \end{bmatrix}$

这里，$0.707$ 是 $\frac{1}{\sqrt{2}}$ 的数值近似值。如果你通过对这些数字进行平方来计算概率（$0.707^2$），你将得到大约 $0.5$，从而验证了 50/50 的概率分布。

在下一节中，我们将学习相位门，它允许我们在不改变概率大小的情况下修改这些状态的复数角度。
