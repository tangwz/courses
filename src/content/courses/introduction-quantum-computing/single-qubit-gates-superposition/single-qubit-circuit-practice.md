---
course: "introduction-quantum-computing"
chapter: "single-qubit-gates-superposition"
lesson: "single-qubit-circuit-practice"
sourceId: 8157
sourceUrl: "https://apxml.com/zh/courses/introduction-quantum-computing/chapter-3-single-qubit-gates-superposition/single-qubit-circuit-practice"
title: "单比特量子线路实践"
description: "编写简单的量子线路代码，直观观察量子门的效果。"
order: 6
plots: ["plots/8157-0.json"]
sourceHash: "f9e1ae2984bfe5d6101033a3ea8ecc9430551a96836aa7c04a48cdced5382ee6"
sourceCorrections: []
---

编写代码来模拟量子态可以使数学表达变得具体。这些表达可以转化为执行向量 (vector)运算的 Python 脚本。量子比特通常使用狄拉克符号和矩阵乘法来描述。在编程环境中，我们可以构建一个**量子线路**。这个对象作为一个容器，承载了我们希望在量子比特上执行的一系列操作，包括初始化、门操作和测量。

### 定义线路结构

一个基础的量子程序由三个阶段组成。首先，我们初始化系统。默认情况下，大多数量子软件开发工具包 (SDK) 会将量子比特初始化为基态 $|0\rangle$。其次，我们应用一系列量子门来操作状态向量 (vector)。最后，我们进行测量以使量子态坍缩，并将结果记录为经典比特（0 或 1）。

为了存储测量结果，我们的线路既需要量子寄存器（用于存放量子比特），也需要经典寄存器（用于存放输出）。

下图展示了一个标准单比特量子线路的流程。量子比特从左向右移动，在被测量之前经过一个量子门。

> 包含初始化、操作和测量的基础量子线路信息流。

### 实现 X 门

我们从 Pauli-X 门开始。正如之前所确定的，X 门的作用是比特翻转。如果我们将其应用于初始化的状态 $|0\rangle$，状态向量 (vector)会旋转到 $|1\rangle$。

在 Python 中，使用类似 Qiskit 等库的通用语法结构，实现步骤如下：

1. **初始化**：创建一个具有一个量子比特和一个经典比特的线路。
2. **应用量子门**：在目标量子比特索引 (0) 上调用 X 函数。
3. **测量**：将量子比特 0 的量子态映射到经典比特 0。

```python
# 线路逻辑的伪代码表示
circuit = QuantumCircuit(1, 1)  # 1 个量子比特，1 个经典比特

# 应用 Pauli-X 门
circuit.x(0)

# 测量量子比特 0 并存储在经典比特 0 中
circuit.measure(0, 0)
```

由于 X 门在从基态开始时是确定性的，因此结果是可以预见的。向量从 $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ 变换为 $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$。如果我们在线路模拟器上运行，将以 100% 的概率得到结果 `1`。

### 使用 Hadamard 门创建叠加态

当我们使用 Hadamard 门 ($H$) 时，其行为表现出量子力学特有的性质。这个门将量子比特置于均等叠加态。

$H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}} = |+\rangle$

当我们为此编写代码时，指令看起来与 X 门的例子相似，但计算结果却截然不同。

```python
# 叠加态线路
circuit = QuantumCircuit(1, 1)

# 应用 Hadamard 门创建叠加态
circuit.h(0)

# 测量
circuit.measure(0, 0)
```

在这种情况下，量子比特处于一种测量得到 0 的概率为 $| \frac{1}{\sqrt{2}} |^2 = 0.5$，测量得到 1 的概率同样为 0.5 的状态。

### 模拟与重复次数 (Shots)

上述代码的一次执行只会产生一个二进制数字：0 或 1。这单个数据点并不能证明量子比特处于叠加态，它只告诉我们状态坍缩成了什么。为了验证概率振幅，我们必须多次执行该线路。

在量子编程中，我们定义一个通常称为 **shots** 的参数 (parameter)。这代表实验重复的次数。常见的重复次数有 1024、4096 或 8192。通过汇总多次重复的结果，我们可以构建概率分布。

如果我们以 1000 次重复运行 Hadamard 线路，我们预期大约得到 500 个 0 和 500 个 1。由于统计偏差，实际数字可能是 489 和 511，但随着重复次数的增加，它们会趋向于 50/50 分布。

下图显示了在模拟器上运行 1024 次 Hadamard 门实验的典型结果分布。



![测量结果：Hadamard 门 (1024 次重复)](plots/8157-0.json)



> 在基态应用 Hadamard 门后测量结果的统计分布。

### 顺序操作

我们可以将多个量子门串联在一起来执行更复杂的旋转。操作的顺序非常关键，因为矩阵乘法不满足交换律。先应用 X 门再应用 Hadamard 门产生的状态向量 (vector)，与先应用 Hadamard 门再应用 X 门是不同的。

考虑操作序列 $Z H |0\rangle$：

1. 从 $|0\rangle$ 开始。
2. 应用 $H$：变换为 $|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$。
3. 应用 $Z$：Z 门翻转 $|1\rangle$ 分量的相位。状态变为 $|-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$。

如果我们测量这个状态，概率仍然是 50/50，因为 Z 门只改变了*相位*（正负号），而没有改变*振幅*的大小。这凸显了测量的一个限制：我们无法通过标准的 Z 基测量直接观察到 $|+\rangle$ 和 $|-\rangle$ 之间的相位差异。我们需要在测量前应用另一个 Hadamard 门来区分这两个状态。

通过练习这些线路构建，你可以验证前几节中所做的线性代数预测。操作单量子比特的能力是迈向更复杂阶段的起点：让两个量子比特相互作用以创建量子纠缠。
