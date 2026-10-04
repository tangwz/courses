---
course: "introduction-quantum-computing"
chapter: "linear-algebra-quantum-mechanics"
lesson: "unitary-matrices-operations"
sourceId: 8149
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-2-linear-algebra-quantum-mechanics/unitary-matrices-operations"
title: "酉矩阵与操作"
description: "酉矩阵如何保持量子系统中的概率不变。"
order: 4
plots: ["plots/8149-0.json"]
sourceHash: "6363216565275ef9d67368bfbf7a9e0206e17cc62f0a536162e0bfb8a02a3500"
sourceCorrections: []
---

量子态在复向量 (vector)空间中表示为向量。如果向量代表量子比特的**状态**，那么我们需要一种数学机制来表示改变该状态的**操作**。正如经典编程中的函数可能会将布尔值从 `True` 切换为 `False` 一样，量子力学使用矩阵来完成这些操作。

当一个操作作用于量子比特时，我们在数学上通过将状态向量乘以矩阵来表示。然而，并非任何矩阵都可以。量子演化必须遵循特定的规则，以确保物理上的有效性。这类管理有效操作的矩阵被称为**酉矩阵** (Unitary Matrices)。

### 保持概率不变

量子力学中最重要的规则是：所有可能结果的总概率之和必须始终等于 1（或 100%）。

回想一下，测量特定状态的概率与状态向量 (vector)的幅值平方（或长度）相关。如果我们使用一个拉伸向量的矩阵，总概率将超过 1。如果我们使用一个收缩向量的矩阵，总概率将降至 1 以下。

因此，有效的量子算符必须是**保长**的。它们可以在希尔伯特空间内的任何位置旋转状态向量，但绝不能改变其长度。这就是酉矩阵的几何定义。



![作为旋转的酉操作](plots/8149-0.json)



> 单位圆表示总概率为 1 的所有有效量子态。酉操作会旋转向量（从蓝色变为青绿色），但保持箭头的顶端正好落在圆上。

### 共轭转置

为了在数学上定义酉矩阵，首先需要理解**共轭转置**。在线性代数教材中，这通常由剑标符号 ($^\dagger$) 表示，有时也被称为厄米共轭 (Hermitian conjugate)。

计算共轭转置包含两个步骤：

1. **转置：** 交换行和列。
2. **共轭：** 对每个数字取复共轭（翻转虚部的符号，使 $a+bi$ 变为 $a-bi$）。

如果我们有一个矩阵 $A$：

$A = \begin{pmatrix} 1 & 1-i \\ 0 & 2i \end{pmatrix}$

转置 $A^T$ 为：

$A^T = \begin{pmatrix} 1 & 0 \\ 1-i & 2i \end{pmatrix}$

共轭转置 $A^\dagger$ 为：

$A^\dagger = \begin{pmatrix} 1 & 0 \\ 1+i & -2i \end{pmatrix}$

### 数学定义

如果一个矩阵 $U$ 的共轭转置也是它的逆矩阵，那么该矩阵就是酉矩阵。在数学上写为：

$U^\dagger U = I$

其中 $I$ 是**单位矩阵**，这是一个对角线上全为 1，其他位置全为 0 的方阵。乘以单位矩阵相当于一个数乘以 1；它保持状态不变。

$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$

这个定义保证了可逆性。如果你对量子比特应用量子门 $U$，你可以通过应用 $U^\dagger$ 回到原始状态。这表明，从理论上讲，量子信息永远不会丢失（直到发生测量）。

### 使用 NumPy 验证酉矩阵

你经常会使用 Python 来模拟这些操作。我们可以使用 NumPy 来验证一个矩阵是否为酉矩阵。让我们看看 **Pauli X** 门，通常被称为“量子非门”，它会翻转状态向量 (vector)的振幅。

Pauli X 的矩阵是：

$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$

我们可以通过计算 $X^\dagger X$ 并观察是否得到单位矩阵来检查它是否为酉矩阵。

```python
import numpy as np

# 定义 Pauli X 矩阵
X = np.array([[0, 1], 
              [1, 0]])

# 计算共轭转置 (dagger)
# .conj() 取复共轭
# .T 进行转置
X_dagger = X.conj().T

# 执行矩阵乘法（点积）
result = np.dot(X_dagger, X)

print("X_dagger * X 的结果:")
print(result)
```

输出将是：

```text
[[1 0]
 [0 1]]
```

由于结果是单位矩阵，因此 Pauli X 门是一个有效的酉算符。

### 将操作应用于状态

为了模拟量子计算，我们使用点积将矩阵应用于状态向量 (vector)。

$|\psi_{\text{新}}\rangle = U |\psi_{\text{当前}}\rangle$

让我们从处于 $|0\rangle$ 态的量子比特开始，由向量 $\begin{pmatrix} 1 \\ 0 \end{pmatrix}$ 表示。我们将应用 Pauli X 门。

$X|0\rangle = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} (0\cdot1) + (1\cdot0) \\ (1\cdot1) + (0\cdot0) \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix} = |1\rangle$

这个线性代数运算证实了 X 门将状态从 $|0\rangle$ 翻转为 $|1\rangle$。

以下是在 Python 中实现此状态转换的方法：

```python
# 定义初始态 |0>
state_0 = np.array([[1], 
                    [0]])

# 使用点积应用 X 门
new_state = np.dot(X, state_0)

print("新状态向量:")
print(new_state)
```

### 可视化操作流程

在构建算法时，理解数据流很有帮助。我们从一个输入向量 (vector)开始，让它通过一系列酉矩阵（门），最后得到一个输出向量。

> 矩阵充当一个函数，在保持输入向量范数不变的情况下，将其转换为新的输出向量。

### 属性总结

在使用量子算符时，请记住以下属性：

1. **线性：** 如果你将矩阵 $U$ 应用于状态的叠加（例如 $a|0\rangle + b|1\rangle$），该矩阵会分别作用于每个部分：$U(a|0\rangle + b|1\rangle) = a(U|0\rangle) + b(U|1\rangle)$。
2. **可逆性：** 每个酉量子门都有一个逆。如果你知道输出状态和所使用的门，就可以在数学上重建输入状态。
3. **范数保持：** 向量 (vector)分量的绝对值平方之和保持不变。

在下一节中，我们将学习**特征值和特征向量**。这些概念让我们能够确定在对这些变换后的状态进行测量时，实际上会看到哪些值。
