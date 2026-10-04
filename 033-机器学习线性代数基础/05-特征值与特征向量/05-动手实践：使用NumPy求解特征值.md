# 动手实践：使用NumPy求解特征值

来源：[原文](https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-5-eigenvalues-and-eigenvectors/numpy-finding-eigenvalues-eigenvectors)

[返回章节目录](README.md) · [返回课程目录](../README.md)

手动求解特征方程有助于培养直觉，但对于机器学习 (machine learning)中遇到的矩阵而言并不实用。对于计算任务，可以利用NumPy等数值计算库高效地找到特征值和特征向量 (vector)。

### 使用 `numpy.linalg.eig`

NumPy的线性代数模块`linalg`包含`eig()`函数，它正是我们所需要的。它接受一个方阵作为输入，并返回两个NumPy对象：

1. 一个包含特征值的一维数组。
2. 一个二维数组，其中每个**列**是与第一个数组中相同索引处的特征值对应的特征向量 (vector)。

让我们从一个简单的2x2矩阵开始。考虑变换矩阵 $A$：


$$
A = \begin{pmatrix} 4 & -2 \\ 1 & 1 \end{pmatrix}
$$


现在，我们使用NumPy来找到它的特征值和特征向量。

```python
import numpy as np

# 定义我们的方阵
A = np.array([[4, -2],
              [1,  1]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print("特征值：")
print(eigenvalues)

print("\n特征向量（每列为一个特征向量）：")
print(eigenvectors)
```

运行此代码会产生以下输出：

```
特征值：
[3. 2.]

特征向量（每列为一个特征向量）：
[[0.89442719 0.70710678]
 [0.4472136  0.70710678]]
```

从这个结果中，我们可以识别出我们的特征值-特征向量对：

- 第一个特征值是 $\lambda_1 = 3$，对应的特征向量是 $v_1 = \begin{pmatrix} 0.894 \\ 0.447 \end{pmatrix}$。
- 第二个特征值是 $\lambda_2 = 2$，对应的特征向量是 $v_2 = \begin{pmatrix} 0.707 \\ 0.707 \end{pmatrix}$。

请注意，NumPy返回的特征向量是**单位向量**，这表示它们的长度（L2范数）为1。这是数值计算库中的一种标准做法，因为特征向量的方向才是主要考量，而不是其大小。

### 验证特征方程

为了确认我们的结果，我们可以检查它们是否满足定义方程 $Av = \lambda v$。让我们为第一个特征值 $\lambda_1 = 3$ 及其特征向量 (vector) $v_1$ 进行测试。

首先，我们从`eigenvectors`矩阵中分离出特征向量。记住，它是第一列。

```python
# 分离第一个特征值和特征向量
lambda1 = eigenvalues[0]
v1 = eigenvectors[:, 0] # 第一列

print("Lambda 1：", lambda1)
print("向量 1：", v1)
```

现在，我们来计算方程的两边。

```python
# 方程的左边：A @ v1
left_side = A @ v1

# 方程的右边：lambda1 * v1
right_side = lambda1 * v1

print("\nA @ v1 =", left_side)
print("lambda1 * v1 =", right_side)
```

输出将是：

```
A @ v1 = [2.68328157 1.34164079]
lambda1 * v1 = [2.68328157 1.34164079]
```

这两个结果是相同的，确认我们找到了有效的特征值和特征向量对。您可以对第二对（$\lambda_2=2$ 和 $v_2$）执行相同的检查，以进一步验证结果。

### 特征向量 (vector)的几何视角

我们可以将这种关系可视化，以更好地理解“方向不变”的含义。以下图表显示了原始特征向量 $v_1$（蓝色）和变换后的向量 $A v_1$（橙色）。



[交互图表：特征向量的变换](https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-5-eigenvalues-and-eigenvectors/numpy-finding-eigenvalues-eigenvectors#plot-ea0p60)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "range": [
        0,
        3.5
      ],
      "title": "X轴",
      "zeroline": false
    },
    "yaxis": {
      "range": [
        0,
        2
      ],
      "title": "Y轴",
      "zeroline": false
    },
    "title": "特征向量的变换",
    "annotations": [
      {
        "x": 0.894,
        "y": 0.447,
        "text": "v1",
        "showarrow": true,
        "ax": 40,
        "ay": -40
      },
      {
        "x": 2.68,
        "y": 1.34,
        "text": "A * v1",
        "showarrow": true,
        "ax": -40,
        "ay": -40
      }
    ],
    "showlegend": true,
    "width": 600,
    "height": 400
  },
  "data": [
    {
      "x": [
        0,
        0.89442719
      ],
      "y": [
        0,
        0.4472136
      ],
      "mode": "lines+markers",
      "name": "原始特征向量 (v1)",
      "line": {
        "color": "#4263eb",
        "width": 3
      },
      "marker": {
        "color": "#4263eb",
        "size": 8
      }
    },
    {
      "x": [
        0,
        2.68328157
      ],
      "y": [
        0,
        1.34164079
      ],
      "mode": "lines+markers",
      "name": "变换后的向量 (A * v1)",
      "line": {
        "color": "#fd7e14",
        "width": 3,
        "dash": "dot"
      },
      "marker": {
        "color": "#fd7e14",
        "size": 8
      }
    }
  ]
}
```

</details>



> 原始特征向量 $v_1$ 和经过变换 $A v_1$ 得到的向量。两个向量都位于从原点延伸的完全相同的直线上。变换只是将向量按其特征值（$\lambda_1 = 3$）的因子进行了缩放。

正如图表所示，将矩阵 $A$ 作用于其特征向量 $v_1$ 并没有使其旋转。它只是将其拉伸，使其指向相同的方向。新向量的长度是原始向量长度的三倍，这与特征值 $\lambda_1 = 3$ 完全对应。这就是特征方程的几何意义的体现。

计算线性变换主轴的这种能力是一个有用的工具。在最后一章中，我们将看到这个过程如何构成主成分分析（PCA）的核心，这是一种在机器学习 (machine learning)中广泛使用的降维技术。

## 参考资料

- [numpy.linalg.eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html) — NumPy Developers (2023)
  使用NumPy计算特征值和特征向量的官方文档。
- [Linear Algebra and Learning from Data](https://math.mit.edu/~gs/learningfromdata/) — Gilbert Strang (2019)
  Publisher: Wellesley-Cambridge Press
  一本涵盖线性代数概念（包括特征值和特征向量）的教科书，具有在机器学习和数据科学中的直接应用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  本书提供了机器学习的数学基础，包括与特征值和特征向量相关的线性代数主题。

---

[上一节](04-%E7%89%B9%E5%BE%81%E6%96%B9%E7%A8%8B.md) · [下一节](../06-%E8%BF%9E%E6%8E%A5%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/01-%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%95%B0%E6%8D%AE%E8%A1%A8%E7%A4%BA.md)
