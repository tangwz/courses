---
course: "introduction-quantum-computing"
chapter: "linear-algebra-quantum-mechanics"
lesson: "vectors-and-state-spaces"
sourceId: 8147
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-2-linear-algebra-quantum-mechanics/vectors-and-state-spaces"
title: "向量与状态空间"
description: "在希尔伯特空间中使用向量定义量子态。"
order: 2
plots: ["plots/8147-0.json"]
sourceHash: "54c289104c5dd7085b4ead813bc4ea9896cd3119cd7e37abaf0a555c6a412825"
sourceCorrections: []
---

为了用数学方式描述量子比特的状态，我们将复数排列成向量 (vector)。经典编程通常使用具有多个属性的类来表示对象的状态。而在量子力学中，系统的状态由列向量表示。该向量包含了在特定时间点描述系统所需的全部信息。

### 状态向量 (vector)与狄拉克符号

我们使用一种称为狄拉克符号（或 bra-ket 符号）的特定记法来标记 (token)这些向量。通用量子态的标准符号是希腊字母 psi ($\psi$)。我们将这个字母放在“ket”中，写成 $|\psi\rangle$。

当你看到 $|\psi\rangle$ 时，它代表一个包含复数的列向量。对于单个量子比特，该向量有两个分量。我们称这些分量为概率幅。

$|\psi\rangle = \begin{bmatrix} \alpha \\ \beta \end{bmatrix}$

这里，$\alpha$ 和 $\beta$ 是我们在前一节讨论过的复数 ($z = a + bi$)。

- $\alpha$ 代表与状态“0”相关的概率幅。
- $\beta$ 代表与状态“1”相关的概率幅。

这个二维向量存在于一种被称为希尔伯特空间的特定数学结构中。对于工程师来说，你可以将希尔伯特空间看作是一个可以处理复数并允许我们计算向量之间长度和夹角的标准向量空间。

### 基向量 (vector)

为了使这些向量发挥作用，我们定义了一个标准坐标系。在经典的二维图形中，你会使用 $x$ 轴和 $y$ 轴。在量子计算中，我们的轴是计算基态，标记 (token)为 $|0\rangle$ 和 $|1\rangle$。

这些状态对应于经典比特值 0 和 1，但它们现在被表示为正交单位向量：

$|0\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$

$|1\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$

任何单量子比特状态 $|\psi\rangle$ 都是这两个基向量的线性组合。我们可以使用这组基重写状态向量方程：

$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle = \alpha \begin{bmatrix} 1 \\ 0 \end{bmatrix} + \beta \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} \alpha \\ \beta \end{bmatrix}$

这个等式用数学方式表达了叠加原理。量子比特不只是 0 或 1。它是一个指向 $|0\rangle$ 轴和 $|1\rangle$ 轴之间某个位置的向量。

> 状态向量相对于计算基向量的几何表示。

### 归一化 (normalization)条件

在物理系统中，概率之和必须恰好等于 1 (100%)。因为概率幅幅值的平方决定了测量结果的概率，所以一个有效的量子状态向量 (vector)的长度（或模长）必须为 1。

复向量的模长是使用欧几里得范数计算的。对于我们的状态 $|\psi\rangle = \begin{bmatrix} \alpha \\ \beta \end{bmatrix}$，条件是：

$|\alpha|^2 + |\beta|^2 = 1$

满足此条件的向量称为归一化向量或单位向量。如果你执行的数学运算导致向量长度不为 1，则该向量不再代表封闭量子系统的有效物理状态。

例如，状态 $|\psi\rangle = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ 不是一个有效的量子态，因为 $1^2 + 1^2 = 2$。为了使其有效，我们必须通过除以平方和的平方根来进行归一化，得到 egin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix}。

### 在 Python 中实现状态向量 (vector)

我们可以使用 NumPy 来处理这些线性代数运算。虽然像 Qiskit 这样的特定量子框架会在后台处理这些内容，但手动创建状态向量有助于理清底层的数学逻辑。

我们将状态向量表示为 NumPy 数组。请注意，在 NumPy 中，简单的列表 `[1, 0]` 会创建一个一维数组。为了正确表示用于线性代数运算的列向量，我们通常将其定义为形状为 `(2, 1)` 的二维数组。

以下是如何定义基向量和叠加态的方法：

```python
import numpy as np

# 定义标准基向量
ket_0 = np.array([[1], [0]])
ket_1 = np.array([[0], [1]])

print(f"基态 |0> 的形状: {ket_0.shape}")

# 创建一个叠加态：0 和 1 的概率相等
# 我们使用 1/sqrt(2) 来确保归一化
alpha = 1 / np.sqrt(2)
beta = 1 / np.sqrt(2)

psi = alpha * ket_0 + beta * ket_1

print("状态向量 |psi>:")
print(psi)
```

输出确认了我们的叠加态是一个包含概率幅的列向量。

为了验证状态是否有效，我们可以写一个简单的函数来检查它是否已归一化 (normalization)。我们计算元素绝对值平方的总和。

```python
def is_normalized(state_vector):
    # 计算绝对值平方的和（即概率）
    probabilities = np.abs(state_vector)**2
    total_prob = np.sum(probabilities)

    # 检查总和是否接近 1（处理浮点数精度问题）
    return np.isclose(total_prob, 1.0)

# 检查我们的 psi 向量
print(f"|psi> 是否已归一化? {is_normalized(psi)}")

# 检查一个无效向量
invalid_vec = np.array([[1], [1]])
print(f"[1, 1] 是否已归一化? {is_normalized(invalid_vec)}")
```

### 可视化概率

理解向量 (vector)本身很重要，但通常我们想知道它所代表的概率分布。向量存储概率幅，但测量产生概率。

以下图表显示了状态 $|\psi\rangle = \sqrt{0.2}|0\rangle + \sqrt{0.8}|1\rangle$ 的概率分布。请注意，虽然向量保存的是平方根（概率幅），但观测逻辑关注的是平方值。



![测量的概率分布](plots/8147-0.json)



> 从状态向量概率幅的幅值平方推导出的测量结果概率。

通过掌握向量并理解归一化 (normalization)，你现在已经拥有了表示量子比特所需的数据结构。下一步是对这些向量应用运算，以模拟状态的变化。
