---
course: "linear-algebra-essentials-ml"
chapter: "solving-linear-systems-matrix-inverses"
lesson: "practice-solving-model-coefficients"
sourceId: 1492
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-3-solving-linear-systems-matrix-inverses/practice-solving-model-coefficients"
title: "动手实践：求解模型系数"
description: "使用 NumPy 计算矩阵逆，并求解表示简单模型参数的线性方程组。"
order: 8
plots: ["plots/1492-0.json"]
sourceHash: "2610dd5e8af07731160b834587a3f6d3ca7963068daa8dda213965b7771d70a7"
sourceCorrections: []
---

让我们将理论付诸实践。求解线性系统在机器学习 (machine learning)中一个很直接的用途是确定线性回归等模型的最佳参数 (parameter)（系数）。我们的目标是找到最适合数据点的直线或超平面。

设想一个简单的线性回归情境。我们有一组数据点 $(x_i, y_i)$，其中 $x_i$ 表示特征，$y_i$ 表示目标值。对于具有多个特征的模型，$x_i$ 将是一个向量 (vector)。我们的目的是找到一组系数，我们称之为 $\theta$（在 $Ax=b$ 形式中通常用 $x$ 表示），使得模型的预测值 $\hat{y} = X\theta$ 尽可能接近实际目标值 $y$。

线性回归中的标准方法是最小化预测值与实际值之间的平方误差和。这个最小化问题直接引出一个线性方程组，称为**正规方程**：

$X^T X \theta = X^T y$

这里：

- $X$ 是特征矩阵（常被称为设计矩阵）。每行代表一个数据点，每列代表一个特征。通常会在 $X$ 中添加一列全为 1 的值，以考虑截距项。
- $\theta$ 是我们希望找到的模型系数向量。
- $y$ 是实际目标值向量。
- $X^T$ 是特征矩阵 $X$ 的转置。

请注意，这个方程具有熟悉的 $A x = b$ 形式，其中 $A = X^T X$，$x = \theta$，而 $b = X^T y$。假设矩阵 $A = X^T X$ 是可逆的，我们可以使用矩阵逆来求解 $\theta$：

$\theta = (X^T X)^{-1} X^T y$

让我们通过一个 NumPy 示例来操作。假设我们有一些数据，例如表示房屋面积和其相应价格的数据。我们想找到最适合这些数据的线性模型 `price = intercept + coefficient * square_footage`。

### 设置数据

首先，我们定义样本数据。我们将使用两个特征（例如，房屋面积和卧室数量）来预测价格，以作为一个稍微复杂些的例子。

```python
import numpy as np

# 样本数据：[房屋面积，卧室数量]
X_features = np.array([
    [1500, 3],
    [2000, 4],
    [1200, 2],
    [1800, 3],
    [2500, 4]
])

# 相应的房价（单位：千美元）
y_target = np.array([300, 450, 250, 400, 550])

# 为 X_features 添加一列全为1的值作为截距项
# 这创建了我们的设计矩阵 X
X_design = np.hstack([np.ones((X_features.shape[0], 1)), X_features])

print("设计矩阵 X（含截距列）：\n", X_design)
print("\n目标向量 y：\n", y_target)
```

### 使用逆矩阵方法计算系数

现在，我们应用正规方程公式 $\theta = (X^T X)^{-1} X^T y$。

```python
# 计算 X 的转置乘以 X (XTX)
XTX = X_design.T @ X_design  # 使用 @ 运算符进行矩阵乘法

# 计算 X 的转置乘以 y (XTy)
XTy = X_design.T @ y_target

print("\nX^T * X：\n", XTX)
print("\nX^T * y：\n", XTy)

# 计算 XTX 的逆
XTX_inv = np.linalg.inv(XTX)

print("\n(X^T * X) 的逆：\n", XTX_inv)

# 计算系数 theta
theta = XTX_inv @ XTy

print("\n使用逆矩阵计算得到的系数 (theta)：\n", theta)
print(f"截距: {theta[0]:.2f}")
print(f"房屋面积的系数: {theta[1]:.2f}")
print(f"卧室数量的系数: {theta[2]:.2f}")
```

结果 `theta` 向量 (vector)包含截距以及每个特征（房屋面积和卧室数量）的系数。

### 使用 `np.linalg.solve()` 求解

尽管计算逆矩阵可行，但它在数值上通常不如直接求解 $A x = b$ 系统稳定，计算成本也更高。NumPy 提供了 `np.linalg.solve(A, b)` 函数，它可以不显式计算逆矩阵而求解该系统。这通常是更推荐的方法。

让我们使用 `np.linalg.solve()` 来求解 $(X^T X) \theta = X^T y$。这里，$A = X^T X$ 且 $b = X^T y$。

```python
# 直接求解系统 (XTX) * theta = XTy
theta_solve = np.linalg.solve(XTX, XTy)

print("\n使用 np.linalg.solve() 计算得到的系数 (theta)：\n", theta_solve)
print(f"截距: {theta_solve[0]:.2f}")
print(f"房屋面积的系数: {theta_solve[1]:.2f}")
print(f"卧室数量的系数: {theta[2]:.2f}")
```

您会发现，使用 `np.linalg.solve()` 获得的系数与使用显式逆矩阵方法计算得到的系数几乎相同。然而，对于更大或更复杂的系统，`np.linalg.solve()` 能提供更好的性能和数值精确度。

### 拟合结果可视化（简化为单特征情况）

让我们简化为一个特征（例如，房屋面积）来可视化结果。

```python
import numpy as np
# 仅使用房屋面积
X_features_simple = np.array([1500, 2000, 1200, 1800, 2500])[:, np.newaxis] # 确保它是一个列向量
y_target_simple = np.array([300, 450, 250, 400, 550])

# 添加截距列
X_design_simple = np.hstack([np.ones((X_features_simple.shape[0], 1)), X_features_simple])

# 使用 np.linalg.solve 求解
XTX_simple = X_design_simple.T @ X_design_simple
XTy_simple = X_design_simple.T @ y_target_simple
theta_simple = np.linalg.solve(XTX_simple, XTy_simple)

print("\n简单模型的系数（截距，房屋面积系数）：\n", theta_simple)

# 生成回归线的点
x_line = np.linspace(1100, 2600, 100) # 覆盖我们数据的范围
y_line = theta_simple[0] + theta_simple[1] * x_line

# 创建 Plotly 图表数据
plot_data = {
    "layout": {
        "title": "Linear Regression Fit (Price vs. Square Footage)",
        "xaxis": {"title": "Square Footage"},
        "yaxis": {"title": "Price ($1000s)"},
        "hovermode": "closest",
        "template": "plotly_white" # 更适合网页的简洁外观
    },
    "data": [
        {
            "type": "scatter",
            "mode": "markers",
            "x": X_features_simple.flatten().tolist(),
            "y": y_target_simple.tolist(),
            "name": "Data Points",
            "marker": {"color": "#228be6", "size": 10} # 蓝色
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": x_line.tolist(),
            "y": y_line.tolist(),
            "name": "Regression Line",
            "line": {"color": "#f03e3e", "width": 3} # 红色
        }
    ]
}
```



![线性回归拟合（价格 vs. 房屋面积）](plots/1492-0.json)



> 使用通过求解正规方程得到的系数拟合样本数据的线性回归线。

本动手实践部分说明了求解线性方程组是寻找机器学习 (machine learning)模型参数 (parameter)的实用工具。通过将问题表示为矩阵形式（$Ax=b$，或更具体地说，$(X^T X) \theta = X^T y$），我们可以运用 NumPy 等高效数值库来找到最佳系数，这可以通过矩阵求逆实现，或者更推荐使用像 `np.linalg.solve()` 这样的直接求解器。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  为线性回归提供了深入的统计和数学基础，包括正规方程的推导和最小二乘原理。
- [Introduction to Linear Algebra](https://math.mit.edu/linearalgebra) — Gilbert Strang (2023)
  Publisher: Wellesley-Cambridge Press; Pages: 440
  全面介绍了线性代数，涵盖矩阵运算、逆矩阵和求解线性系统，这些对理解正规方程至关重要。
- [numpy.linalg.solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html) — NumPy Developers (2024)
  Publisher: NumPy
  NumPy官方文档中用于求解线性矩阵方程的函数，强调其相较于矩阵求逆的数值稳定性和效率。
- [CS229 Lecture Notes: Linear Regression](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHg7riP6iYdRzAsDQAzf6oZ4uqnioF1DGCj2XKvp_fXbW_hJD7bWvZEINkmOAsmd_-LbGvOz49jMVsBLa3cjpkUjMy0Rre2llwpk4YbkZhiG1maDvPIowcOOnENDyPXllDz8g==) — Andrew Ng, Tengyu Ma (2023)
  Publisher: Stanford University
  著名机器学习课程的讲义，解释了使用正规方程推导线性回归系数的数学方法。
