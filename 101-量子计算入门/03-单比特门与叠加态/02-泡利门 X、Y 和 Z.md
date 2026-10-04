# 泡利门 X、Y 和 Z

来源：[原文](https://apxml.com/zh/courses/introduction-quantum-computing/chapter-3-single-qubit-gates-superposition/pauli-gates-x-y-z)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在经典数字逻辑中，针对单个比特可用的唯一操作是非门（NOT gate）。如果一个比特是 0，该门将其翻转为 1，反之亦然。量子系统提供了更丰富的操作集。由于量子比特是以复向量 (vector)空间中的向量形式存在的，我们不仅限于翻转数值。我们可以让状态向量绕着不同的轴进行旋转。

泡利门（Pauli gates）是基础的单量子比特算子。它们对应于绕布洛赫球（Bloch sphere）x、y 和 z 轴的旋转。我们将这些门表示为 $2 \times 2$ 的幺正矩阵。当你对量子比特施加一个门时，你实际上是在进行矩阵-向量乘法来计算新的状态。

### 泡利-X 门

泡利-X 门通常被称为“量子非门”或“比特翻转”门。它执行的操作类似于经典的反相器。如果量子比特处于基态 $|0\rangle$，X 门会将其翻转为激发态 $|1\rangle$。从几何上看，这个操作对应于绕布洛赫球 x 轴旋转 $\pi$ 弧度（180 度）。

泡利-X 门的矩阵表示为：

$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$

为了解这如何影响量子比特，我们将状态向量 (vector)乘以这个矩阵。如果我们从状态 $|0\rangle$ 开始，表示为列向量 $\begin{pmatrix} 1 \\ 0 \end{pmatrix}$：

$X|0\rangle = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix} = |1\rangle$

同样地，如果作用于 $|1\rangle$：

$X|1\rangle = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix} = |0\rangle$

X 门交换了基态的振幅。如果你有一个处于叠加态 $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ 的量子比特，施加 X 门的结果是 $\beta|0\rangle + \alpha|1\rangle$。

### 泡利-Z 门

泡利-Z 门起到“相位翻转”门的作用。它保持基态 $|0\rangle$ 不变，但翻转基态 $|1\rangle$ 的符号（即相位）。这个操作对应于绕 z 轴旋转 $\pi$ 弧度。

泡利-Z 门的矩阵为：

$Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$

将其作用于基态会产生与 X 门不同的结果。对于 $|0\rangle$：

$Z|0\rangle = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix} = |0\rangle$

状态 $|0\rangle$ 是 Z 门的本征向量 (vector)，其本征值为 1，这意味着它保持不变。然而，对于 $|1\rangle$：

$Z|1\rangle = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ -1 \end{pmatrix} = -|1\rangle$

这里的负号表示 $\pi$ 的相位偏移。虽然这种全局相位不会影响单个量子比特的测量概率（因为 $|-1|^2 = 1$），但当量子比特处于叠加态或与其他量子比特纠缠时，它就变得非常有意义。它改变了状态向量各分量之间的相对相位。

下图展示了这些门在标准基态上的输入和输出关系。

> 展示基态转换的门操作。

### 泡利-Y 门

泡利-Y 门表示绕 y 轴旋转 $\pi$ 弧度。它结合了比特翻转和相位偏移的效果。由于量子力学中的旋转涉及复向量 (vector)空间，Y 门引入了虚数单位 $i$。

泡利-Y 门的矩阵为：

$Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$

将 Y 作用于 $|0\rangle$：

$Y|0\rangle = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ i \end{pmatrix} = i|1\rangle$

将 Y 作用于 $|1\rangle$：

$Y|1\rangle = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix} \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} -i \\ 0 \end{pmatrix} = -i|0\rangle$

Y 门将基态映射到其相反的状态，并增加了一个复数相位。这个门对于使用布洛赫球的整个范围非常有用，并常用于量子纠错和层析成像中。

### 在布洛赫球上的可视化

为了直观理解这些门，观察它们操作的轴会有所帮助。量子比特的状态是布洛赫球表面上的一个点。

1. **X 轴：** 指向球体的“赤道”。绕此轴旋转会使状态向量 (vector)在两极之间垂直移动。
2. **Y 轴：** 在赤道上指向侧面。绕此轴旋转也会在两极之间移动向量，但遵循一条涉及复相位的不同路径。
3. **Z 轴：** 垂直穿过 $|0\rangle$ 和 $|1\rangle$ 两极。这里的旋转像陀螺一样旋转向量，改变其经度（相位）而不改变其纬度（概率幅度）。



[交互图表：泡利旋转轴](https://apxml.com/zh/courses/introduction-quantum-computing/chapter-3-single-qubit-gates-superposition/pauli-gates-x-y-z#plot-1246xw5)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "泡利旋转轴",
    "scene": {
      "xaxis": {
        "title": "X 轴",
        "range": [
          -1.5,
          1.5
        ],
        "showgrid": false,
        "zeroline": false
      },
      "yaxis": {
        "title": "Y 轴",
        "range": [
          -1.5,
          1.5
        ],
        "showgrid": false,
        "zeroline": false
      },
      "zaxis": {
        "title": "Z 轴",
        "range": [
          -1.5,
          1.5
        ],
        "showgrid": false,
        "zeroline": false
      },
      "aspectmode": "cube"
    },
    "margin": {
      "l": 0,
      "r": 0,
      "b": 0,
      "t": 30
    },
    "showlegend": false
  },
  "data": [
    {
      "type": "scatter3d",
      "mode": "lines+text",
      "x": [
        -1.2,
        1.2,
        null,
        0,
        0,
        null,
        0,
        0
      ],
      "y": [
        0,
        0,
        null,
        -1.2,
        1.2,
        null,
        0,
        0
      ],
      "z": [
        0,
        0,
        null,
        0,
        0,
        null,
        -1.2,
        1.2
      ],
      "line": {
        "color": "#868e96",
        "width": 4
      },
      "text": [
        "",
        "X 轴",
        "",
        "",
        "Y 轴",
        "",
        "",
        "Z 轴"
      ],
      "textposition": "top center"
    },
    {
      "type": "scatter3d",
      "mode": "markers+text",
      "x": [
        0,
        0,
        1,
        0
      ],
      "y": [
        0,
        0,
        0,
        1
      ],
      "z": [
        1,
        -1,
        0,
        0
      ],
      "marker": {
        "size": 6,
        "color": [
          "#1c7ed6",
          "#fa5252",
          "#4dabf7",
          "#be4bdb"
        ]
      },
      "text": [
        "|0⟩ (Z+)",
        "|1⟩ (Z-)",
        "X 门旋转轴",
        "Y 门旋转轴"
      ],
      "textposition": "top left"
    }
  ]
}
```

</details>



> 定义泡利旋转的轴向。

### 在 Python 中实现泡利门

在 Python 中构建量子电路时，可以将这些门视为应用于电路寄存器中特定量子比特的方法。使用 NumPy 等标准线性代数库可以让你在迁移到完整的量子模拟器之前，手动验证矩阵数学运算。

以下是如何使用 NumPy 验证泡利-X 操作的方法：

```python
import numpy as np

# 定义标准基态
zero_state = np.array([[1], [0]])
one_state = np.array([[0], [1]])

# 定义泡利-X 矩阵
pauli_x = np.array([
    [0, 1],
    [1, 0]
])

# 应用门（矩阵乘法）
# 结果应该是 |1>
result = np.dot(pauli_x, zero_state)

print(f"输入态:\n{zero_state}")
print(f"输出态:\n{result}")
```

在量子软件开发工具包 (SDK) 中，抽象程度更高。你通常只需将门添加到电路对象中。例如：

```python
# 量子电路的伪代码
circuit = QuantumCircuit(1) # 创建一个包含 1 个量子比特的电路
circuit.x(0)  # 对量子比特 0 施加泡利-X
circuit.z(0)  # 对量子比特 0 施加泡利-Z
```

理解这三个矩阵为所有单量子比特操作奠定了基础。它们是在希尔伯特空间中操作的主要工具。在下一节中，我们将学习 Hadamard 门，它能让我们离开两极，创建真正的量子叠加态。

---

[上一节](01-%E5%8F%A0%E5%8A%A0%E6%80%81%E7%9A%84%E6%A6%82%E5%BF%B5.md) · [下一节](03-Hadamard%20%E9%97%A8.md)
