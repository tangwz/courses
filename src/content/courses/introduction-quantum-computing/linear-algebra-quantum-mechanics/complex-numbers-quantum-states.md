---
course: "introduction-quantum-computing"
chapter: "linear-algebra-quantum-mechanics"
lesson: "complex-numbers-quantum-states"
sourceId: 8146
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-2-linear-algebra-quantum-mechanics/complex-numbers-quantum-states"
title: "量子态中的复数"
description: "理解复数和概率幅的作用。"
order: 1
plots: ["plots/8146-0.json"]
sourceHash: "1b568ae75cf1b08d6bdfe39f2982903a3a8925a617b9fb2fed0551be57c9c760"
sourceCorrections: []
---

为了准确描述量子态，仅使用实数是不够的。经典系统通过 0 到 1 之间的概率来衡量确定性。然而，量子力学运行在**概率幅**之上。概率幅是描述测量前量子系统状态的值。与标准概率不同，概率幅可以是负数或虚数。这种特性允许量子态相互干涉，以经典概率无法模拟的方式构建或抵消结果。

为了处理概率幅，我们必须使用复数。复数将传统的数轴扩展到了二维平面。本部分介绍了计算量子概率和旋转所需的复数运算。

### 复数的结构

复数 $z$ 由两个不同的部分组成：实部和虚部。它通常写成以下形式：

$z = a + bi$

在这里，$a$ 代表实部分量，而 $b$ 代表虚部分量。符号 $i$ 是虚数单位，其定义性质是它的平方等于负一：

$i^2 = -1$

在 Python 和许多工程背景中，虚数单位通常用 `j` 而不是 `i` 来表示。例如，数字 $3 + 4i$ 在 Python 代码中写作 `3 + 4j`。

理解复数需要将它们想象在平面上，而不是直线上。横轴代表实数，纵轴代表虚数。这种几何解释允许我们将量子态视为向量 (vector)。



![z = 3 + 4i 的几何表示](plots/8146-0.json)



> 在复平面上绘制复数 $3 + 4i$。向量长度代表模长，与 x 轴的夹角代表相位。

### 计算模长

在量子计算中，复数最重要的性质是它的模长（或模）。模长将抽象的量子概率幅与概率联系起来。

从几何上看，模长是原点 $(0,0)$ 到点 $(a,b)$ 的距离。我们使用勾股定理来计算：

$|z| = \sqrt{a^2 + b^2}$

以 $z = 3 + 4i$ 为例，计算过程为：

$|z| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$

这个模长非常有用，因为特定量子结果的概率是其概率幅模长的平方。如果一个状态的概率幅为 $\alpha$，那么测量到该状态的概率就是 $|\alpha|^2$。

### 共轭复数

为了进行内积等数学运算（我们用内积来计算量子态之间的重叠），我们需要共轭复数。复数 $z$ 的共轭记作 $z^*$ 或 $\bar{z}$，通过反转虚部的符号来得到。

如果 $z = a + bi$，那么：

$z^* = a - bi$

一个复数与其共轭相乘总是得到一个非负实数，该实数等于模长的平方：

$z \cdot z^* = (a + bi)(a - bi) = a^2 + b^2 = |z|^2$

这种运算确保了我们在量子力学中计算概率时，结果总是 0 到 1 之间的实数。

### 极坐标与相位

虽然代数形式 $a + bi$ 方便加法运算，但极坐标形式通常更适合乘法运算以及理解量子门。在极坐标中，复数由以下两个要素定义：

1. **半径 ($r$):** 即模长 $|z|$。
2. **角度 ($\theta$):** 相对于正实轴的相位角。

利用欧拉公式，我们可以将复数表示为：

$z = r e^{i\theta}$

在这种表示法中，$e$ 是自然对数的底。这种格式在量子力学中是标准形式，因为许多量子操作（门）的作用类似于旋转。它们改变相位角 $\theta$ 而不改变模长 $r$。这意味着测量的概率保持不变，但该状态与其他状态的关系发生了变化。

### Python 实现

你经常会使用 NumPy 等 Python 库来处理这些计算。Python 原生支持复数。

以下是如何在 Python 中定义复数并计算其性质：

```python
import numpy as np

# 定义一个复数（使用 j 表示虚数单位）
z = 3 + 4j

# 计算实部和虚部
real_part = np.real(z)
imag_part = np.imag(z)

# 计算模长（绝对值）
magnitude = np.abs(z)

# 计算共轭
conjugate = np.conj(z)

print(f"复数: {z}")
print(f"模长: {magnitude}")
print(f"共轭: {conjugate}")
```

运行这段代码会得到模长 `5.0` 和共轭 `3-4j`。

在处理极坐标时，NumPy 提供了 `angle` 函数来获取弧度制的相位 $\theta$：

```python
# 计算相位角
theta = np.angle(z)

# 使用欧拉公式逻辑转回代数形式
# z = r * (cos(theta) + i * sin(theta))
z_reconstructed = magnitude * (np.cos(theta) + 1j * np.sin(theta))
```

掌握这些基本运算是构建量子电路所需的线性代数工具的第一步。在下一节中，我们将把这些复数组合成向量 (vector)，以表示量子比特的完整状态。
