---
course: "introduction-quantum-computing"
chapter: "linear-algebra-quantum-mechanics"
lesson: "numpy-for-quantum-math-practice"
sourceId: 8151
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-2-linear-algebra-quantum-mechanics/numpy-for-quantum-math-practice"
title: "Numpy 量子数学实践"
description: "使用 Numpy 解决与量子计算相关的线性代数问题。"
order: 6
plots: []
sourceHash: "b9b3e50dd739e6eafc810b9b883c9cd2bc80e5072a47543d062b950efd1a1dce"
sourceCorrections: []
---

理论提供了蓝图，但软件实现驱动了量子系统的实际模拟。量子比特的数学行为使用狄拉克符号和矩阵代数来描述。要构建一个可运行的量子模拟器或算法，必须将这些数学对象转化为代码。

Python 及其 NumPy 库是完成此任务的标准工具，因为它可以高效地处理向量 (vector)和矩阵运算。在本节中，你将实现本章涵盖的线性代数内容，将抽象的数学状态转换为具体的 Python 对象。

### 环境设置

在操作量子态之前，先导入 NumPy。我们通常将量子力学视为复数上的线性代数。因此，在定义数组时，务必将数据类型指定为复数（complex），以支持虚部（振幅）。

```python
import numpy as np

# 定义标准基态 |0> 和 |1>
# 我们使用列向量（2 行 1 列）
ket_0 = np.array([[1], [0]], dtype=complex)
ket_1 = np.array([[0], [1]], dtype=complex)

print(f"|0> 态:\n{ket_0}")
```

注意这些数组的形状。在量子力学文献中，右矢 $|\psi\rangle$ 始终是一个列向量 (vector)。在 NumPy 中，一维数组（例如 `[1, 0]`）的行为与二维列数组（例如 `[[1], [0]]`）不同。保持二维结构可以防止在用矩阵乘以这些向量时出现形状不匹配的错误。

### 矩阵乘法可视化

当量子门作用于量子比特时，在数学上是矩阵乘以向量 (vector)。使用标准的矩阵乘法流程来观察运算过程中维度的对齐 (alignment)方式会很有帮助。

> 应用量子门时的数据流。 (2,2) 矩阵作用于 (2,1) 向量，生成一个新的 (2,1) 向量。

### 使用内积计算概率

系统处于特定状态的概率由振幅的模平方决定。在数学上，对于状态向量 (vector) $| \psi \rangle$，测量得到 $|0\rangle$ 的概率是 $|\langle 0 | \psi \rangle|^2$。

要在 NumPy 中计算这一点，你需要 **埃尔米特共轭**（共轭转置）。在狄拉克符号中，如果 $|\psi\rangle$ 是一个列向量，那么 $\langle\psi|$ 就是一个行向量，其中每个元素都是原始元素的复共轭。

NumPy 提供了 `.conj().T` 属性链来执行此操作。或者，如果你使用 matrix 子类，可以使用 `.H` 属性，但在现代 NumPy 中更推荐使用标准数组。

```python
# 创建一个叠加态: 1/sqrt(2) * (|0> + |1>)
psi = 1/np.sqrt(2) * (ket_0 + ket_1)

# 计算左矢 (行向量，复共轭)
bra_0 = ket_0.conj().T

# 计算内积 <0|psi>
amplitude = bra_0 @ psi

# 计算概率 (振幅平方)
probability = np.abs(amplitude)**2

print(f"|0> 的振幅: {amplitude[0][0]}")
print(f"测量得到 |0> 的概率: {probability[0][0]}")
```

注意 `@` 运算符的使用。这是 Python 中矩阵乘法的标准运算符。如果你使用 `*`，NumPy 将尝试进行逐元素相乘，这在数学上是不正确的。

### 将量子门建模为矩阵

量子门是酉矩阵。要模拟一个门，你需要定义矩阵并将其与状态向量 (vector)相乘。让我们实现 Pauli-X 门，通常被称为“量子非门”，它会将 $|0\rangle$ 翻转为 $|1\rangle$。

$X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$

```python
# 定义 Pauli-X 矩阵
X_gate = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

# 将门作用于 |0>
new_state = X_gate @ ket_0

print(f"原始状态:\n{ket_0}")
print(f"作用 X 门后的状态:\n{new_state}")
```

如果你运行这段代码，`new_state` 将等于 `[[0.+0.j], [1.+0.j]]`，这对应于 $|1\rangle$ 态。

### 验证酉矩阵

量子力学的一个基本属性是，操作必须是酉的以保持概率守恒。如果矩阵 $U$ 满足以下条件，则它是酉矩阵：

$U^\dagger U = I$

其中 $I$ 是单位矩阵。在定义自定义门或检查数学运算时，验证这一属性非常有用。

```python
def is_unitary(matrix):
    # 计算 U 的共轭转置 (dagger)
    dagger = matrix.conj().T

    # 计算乘积 U dagger * U
    product = dagger @ matrix

    # 创建一个相同大小的单位矩阵
    identity = np.eye(len(matrix))

    # 检查它们是否接近（处理浮点数误差）
    return np.allclose(product, identity)

print(f"Pauli-X 是酉矩阵吗? {is_unitary(X_gate)}")
```

这里使用 `np.allclose` 而不是 `==`，是因为浮点运算可能会导致极其微小的差异（例如 0.9999999 而不是 1.0）。

### 查找特征值和特征向量 (vector)

量子力学中的测量与特征值相关。当你测量一个状态时，系统会坍缩到算符的一个本征态上，你读出的值就是特征值。

NumPy 通过 `np.linalg.eig` 简化了这一过程。

```python
# 定义 Pauli-Z 矩阵
Z_gate = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(Z_gate)

print("特征值:", eigenvalues)
print("特征向量:\n", eigenvectors)
```

对于 Z 门，你应该会看到特征值为 $1$ 和 $-1$。这些对应于基态 $|0\rangle$ 和 $|1\rangle$。这证实了如果你在 Z 基（标准计算基）下测量量子比特，你将得到区分零态和一态的结果。

### 计算外积

内积（左矢-右矢）产生一个标量（数字），而外积（右矢-左矢）产生一个矩阵。这用于定义投影算符。符号 $|\psi\rangle\langle\psi|$ 表示外积。

在 NumPy 中，你可以使用 `np.outer` 或标准的列向量 (vector)乘以行向量的矩阵乘法来计算它。

```python
# 计算投影算符 |0><0|
# ket_0 是 (2,1)，bra_0 是 (1,2)
projection_0 = ket_0 @ ket_0.conj().T

print("投影算符 |0><0|:\n", projection_0)
```

结果是矩阵：
$\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$

如果你将此算符作用于叠加态，它会将该态投影到 $|0\rangle$ 子空间，有效地模拟了“过滤”测量的数学部分。

通过掌握这些 NumPy 操作——向量定义、矩阵乘法、共轭转置和特征值分解，你已经构建了一个微型量子模拟器。在接下来的章节中，你将使用这些工具来模拟涉及叠加和纠缠的更复杂的系统。
