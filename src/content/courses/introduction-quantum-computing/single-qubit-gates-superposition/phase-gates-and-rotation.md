---
course: "introduction-quantum-computing"
chapter: "single-qubit-gates-superposition"
lesson: "phase-gates-and-rotation"
sourceId: 8155
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-3-single-qubit-gates-superposition/phase-gates-and-rotation"
title: "相位门与旋转"
description: "在不改变概率大小的情况下操作量子比特的相位。"
order: 4
plots: ["plots/8155-0.json"]
sourceHash: "b0bcb62e6fe46c5c3c26c68b1b0972f183573baa9439ac3ba8ba59566490e2d4"
sourceCorrections: []
---

Pauli X、Y 和 Z 门在量子比特上执行固定操作，非常类似于经典逻辑中的非门。Hadamard 门使我们能够进入叠加态。然而，构建实用的量子算法不仅需要翻转比特或创建 50/50 的叠加态。通常情况下，需要精确的角度调整来修改量子状态。

这就涉及到了相位门和旋转算子。这些操作允许你操作量子比特的复相位，而不一定会改变测量到 $|0\rangle$ 或 $|1\rangle$ 的概率。理解这些旋转对于量子傅里叶变换和量子相位估计等算法非常有用。

### 相位的意义

在前一节中，我们将 $|+\rangle$ 态定义为：

$|+\rangle = \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle$

如果测量这个状态，观测到 0 的概率为 50%，观测到 1 的概率也为 50%。现在考虑 $|-\rangle$ 态：

$|-\rangle = \frac{1}{\sqrt{2}}|0\rangle - \frac{1}{\sqrt{2}}|1\rangle$

$|-\rangle$ 的测量统计数据与 $|+\rangle$ 完全相同。两者都以相等的概率产生 0 或 1。区别在于相对相位，即两个基向量 (vector)之间的符号。虽然这种差异不会直接体现在标准测量中，但它从根本上改变了量子比特在电路中与其他状态干涉的方式。相位门就是我们用来改变这种相对相位的工具。

### S 门和 T 门

Pauli Z 门绕布洛赫球的 Z 轴将量子比特旋转 $\pi$ 弧度 ($180^\circ$)。此操作翻转 $|1\rangle$ 分量的相位。我们通常需要比完全翻转更精细的控制。

**S 门**（有时称为相位门）绕 Z 轴执行 $\pi/2$ ($90^\circ$) 的旋转。在数学上，它给 $|1\rangle$ 分量应用了一个复数因子 $i$。

$S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$

如果我们将 S 门应用两次，就会得到 $S^2 = Z$。

**T 门**（有时称为 $\pi/8$ 门）绕 Z 轴执行 $\pi/4$ ($45^\circ$) 的旋转。它是 S 门的平方根。

$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$

这些门使我们能够在布洛赫球的赤道上移动。下图说明了从 $|+\rangle$ 态开始，这些相位操作之间的关系。

> 图示显示了相位门的累加效果。T 门的旋转比 S 门更精细，而 S 门的旋转比 Z 门更精细。

### 任意旋转算子

虽然 S 和 T 门提供固定的增量，但量子硬件支持连续旋转。我们使用记为 $R_x(\theta)$、$R_y(\theta)$ 和 $R_z(\theta)$ 的旋转算子来描述这些操作。这些算子带有一个参数 (parameter) $\theta$，表示以弧度为单位的旋转角度。

当将数据编码到量子系统时，这种灵活性非常有用，例如将图像中的像素值归一化 (normalization)为角度。

#### 绕 Z 轴旋转 ($R_z$)

$R_z$ 门是 Z、S 和 T 门的泛化版本。它改变相位而不影响基态的概率。

$R_z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$

请注意，有些教科书在全局相位因子方面的定义略有不同，但 $|0\rangle$ 和 $|1\rangle$ 之间的相对相位差保持为 $e^{i\theta}$。

#### 绕 Y 轴旋转 ($R_y$)

与 Z 旋转不同，绕 Y 轴旋转会改变 $|0\rangle$ 和 $|1\rangle$ 的相对振幅。这意味着 $R_y$ 旋转会改变测量结果的概率。这经常用于将量子比特初始化为特定的概率分布。

$R_y(\theta) = \begin{pmatrix} \cos(\frac{\theta}{2}) & -\sin(\frac{\theta}{2}) \\ \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2}) \end{pmatrix}$

### 旋转的 Python 实现

我们可以使用 Python 的 NumPy 库来验证这些旋转。在这个例子中，我们将手动构建 S 门和通用 $R_y$ 旋转的酉矩阵。我们将它们应用于量子比特并观察态矢量的演变。

这种方法巩固了第 2 章中建立的线性代数基础。

```python
import numpy as np

# 1. 定义基态
ket_0 = np.array([[1], [0]])
ket_1 = np.array([[0], [1]])

# 创建 |+> 态 (叠加态)
# 我们用它来演示 S 门，S 门需要叠加态才能产生明显的效果
ket_plus = (1/np.sqrt(2)) * (ket_0 + ket_1)

print(f"初始状态 (|+>):\n{ket_plus}")

# 2. 定义 S 门矩阵
# 1j 是虚数单位 'i' 的 Python 表示法
s_gate = np.array([
    [1, 0],
    [0, 1j]
])

# 对 |+> 应用 S 门
state_after_s = np.dot(s_gate, ket_plus)

print(f"\n应用 S 门后的状态:\n{state_after_s}")
# 注意第二个元素现在是虚数 (0.707j)

# 3. 定义旋转 90 度 (pi/2) 的 Ry 旋转矩阵
theta = np.pi / 2
ry_gate = np.array([
    [np.cos(theta/2), -np.sin(theta/2)],
    [np.sin(theta/2),  np.cos(theta/2)]
])

# 对标准 |0> 态应用 Ry
# 将 |0> 绕 Y 轴旋转 90 度会产生 |+> 态
state_after_rotation = np.dot(ry_gate, ket_0)

print(f"\n对 |0> 应用 Ry(pi/2) 后的状态:\n{state_after_rotation}")
```

### 可视化相位效果

理解相位需要想象布洛赫球的第三个维度。当我们使用 Hadamard 门时，我们从两极（Z 轴）移动到赤道（X 轴）。一旦进入赤道，相位门（$R_z$, S, T）就会使矢量绕赤道旋转。

如果你将量子比特状态想象成平放在桌面上的时钟指针：

1. **Hadamard 门**：将钟针平放，指向 12 点钟方向。
2. **S 门**：将指针移动到 3 点钟方向。
3. **Z 门**：将指针移动到 6 点钟方向。

最重要的是，只要指针平放在桌面上（赤道），“高度”就是零。这对应于 50/50 的概率。旋转只改变方向（相位），而不改变高度（概率偏向）。

下面的图表绘制了量子比特态矢量在旋转过程中的路径。



![Ry 旋转期间的概率振幅](plots/8155-0.json)



> 概率振幅 $\alpha$ 和 $\beta$ 在 $R_y$ 旋转期间随旋转角度 $\theta$ 变化的函数。在 $\pi/2$（约 1.57 弧度）时，振幅相交，代表完美的叠加态。

掌握这些旋转提供了操控整个希尔伯特空间所需的控制力。在下一节中，我们将讨论这些连续状态如何通过测量公设塌缩回二进制结果。
