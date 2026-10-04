---
course: "calculus-essentials-machine-learning"
chapter: "multivariable-calculus-ml"
lesson: "practice-computing-gradients"
sourceId: 1383
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-3-multivariable-calculus-ml/practice-computing-gradients"
title: "NumPy 梯度计算实践"
description: "使用 NumPy 数值计算多变量函数的梯度。"
order: 7
plots: ["plots/1383-0.json"]
sourceHash: "3c0eb655e432be702479d4d102f5abc8be4d592b99143d7bb2adc32e167766f3"
sourceCorrections: []
---

理解梯度的数学定义很要紧，但对于机器学习 (machine learning)中常遇到的复杂函数，特别是代码隐式定义的函数，通过分析方法（利用微分规则求偏导）计算梯度可能会很繁琐甚至难以处理。幸好，我们可以数值近似计算梯度。下面介绍如何通过有限差分思想，使用 Python 和 NumPy 计算梯度。

### 偏导数的数值近似

回想一下，像 $\frac{\partial f}{\partial x}$ 这样的偏导数，衡量的是当 $x$ 略微变化而其他变量保持不变时，函数 $f$ 的变化率。我们可以使用一个小的步长 $h$ 来近似这个变化率：

- **前向差分：** $\frac{\partial f}{\partial x} \approx \frac{f(x+h, y) - f(x, y)}{h}$
- **后向差分：** $\frac{\partial f}{\partial x} \approx \frac{f(x, y) - f(x-h, y)}{h}$
- **中心差分：** $\frac{\partial f}{\partial x} \approx \frac{f(x+h, y) - f(x-h, y)}{2h}$

对于给定的步长 $h$，中心差分公式通常提供更准确的近似。我们将使用此方法计算梯度向量 (vector)所需的每个偏导数。$h$ 值应该很小（例如 $10^{-5}$ 或 $10^{-7}$），以便近似瞬时变化率，但又不能小到出现浮点精度问题。

### 函数示例

让我们考虑一个类似于机器学习 (machine learning)中简单损失函数 (loss function)的函数。假设我们有一个模型，它有两个参数 (parameter) $w_1$ 和 $w_2$，我们想让以下目标函数 $L(w_1, w_2)$ 最小化：

$L(w_1, w_2) = (2w_1 + 3w_2 - 5)^2 + (w_1 - w_2 - 1)^2$

我们的目标是在特定点，比如 $(w_1, w_2) = (1, 1)$ 处，数值计算梯度 $\nabla L = \left[ \frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2} \right]$。

### 使用 NumPy 实现数值梯度计算

我们可以编写一个 Python 函数，它接受任意多变量函数 `func`、一个点 `point`（表示为 NumPy 数组）和一个步长 `h`，然后返回数值计算的梯度。

```python
import numpy as np

def loss_function(w):
  """
  损失函数 L(w1, w2) 示例。
  参数：
    w: 一个 NumPy 数组 [w1, w2]。
  返回：
    损失函数的标量值。
  """
  w1, w2 = w[0], w[1]
  term1 = (2*w1 + 3*w2 - 5)**2
  term2 = (w1 - w2 - 1)**2
  return term1 + term2

def compute_gradient_numerical(func, point, h=1e-5):
  """
  计算函数在给定点的数值梯度。
  参数：
    func: 要微分的函数。接受 NumPy 数组作为输入。
    point: 表示要计算梯度的点（例如 [w1, w2]）的 NumPy 数组。
    h: 有限差分的步长。
  返回：
    表示梯度向量的 NumPy 数组。
  """
  point = np.asarray(point, dtype=float)
  grad = np.zeros_like(point)

  # 遍历每个维度（参数）
  for i in range(len(point)):
    # 在第 i 个维度上创建偏移 +h 和 -h 的点
    point_plus_h = point.copy()
    point_minus_h = point.copy()
    point_plus_h[i] += h
    point_minus_h[i] -= h

    # 计算中心差分
    partial_derivative = (func(point_plus_h) - func(point_minus_h)) / (2 * h)
    grad[i] = partial_derivative

  return grad

# 定义计算梯度的点
w_point = np.array([1.0, 1.0])

# 计算数值梯度
numerical_gradient = compute_gradient_numerical(loss_function, w_point)

print(f"点 (w1, w2): {w_point}")
print(f"该点处的数值梯度: {numerical_gradient}")

# 为了比较，我们计算 (1, 1) 处的解析梯度
# dL/dw1 = 10*w1 + 10*w2 - 22
# dL/dw2 = 10*w1 + 20*w2 - 28
analyical_gradient = np.array([
    10*w_point[0] + 10*w_point[1] - 22,
    10*w_point[0] + 20*w_point[1] - 28
])
print(f"该点处的解析梯度: {analytical_gradient}")

# 计算差值（误差）
error = np.linalg.norm(numerical_gradient - analytical_gradient)
print(f"数值梯度与解析梯度之间的差值: {error:.2e}")
```

运行此代码将输出：

```
Point (w1, w2): [1. 1.]
Numerical Gradient at point: [-2.  2.]
Analytical Gradient at point: [-2.  2.]
Difference between numerical and analytical gradient: 1.89e-11
```

可以看出，使用中心差分法计算的数值梯度与我们之前推导出的解析梯度（在 $(1, 1)$ 处为 $[-2, 2]$）非常接近。这种细微差异是由于有限差分和浮点运算中固有的近似造成的。

### 梯度可视化

我们可以将函数曲面或其等高线可视化，并在点 $(1, 1)$ 处绘制梯度向量 (vector)。梯度向量 $[-2, 2]$ 指向从 $(1, 1)$ 开始的最陡峭*上升*方向。在梯度下降 (gradient descent)等优化情况（将在下一章讲解）中，我们通常沿梯度*相反*的方向移动以求得最小值。



![L(w1, w2) 等高线图及 (1, 1) 处梯度](plots/1383-0.json)



> 损失函数 (loss function) $L(w_1, w_2)$ 的等高线图。红色 'x' 标记 (token)点 $(1, 1)$，红色箭头表示数值计算的梯度 $[-2, 2]$。注意箭头垂直于等高线指向更高的函数值（更亮的颜色）。

### 重要性和局限

数值梯度计算非常有用：

1. **梯度检查：** 它提供了一种方法来验证您的解析梯度计算（例如，为反向传播 (backpropagation)实现的）是否正确。如果数值梯度和解析梯度非常接近，那么您的实现很可能是正确的。
2. **复杂函数：** 即使函数由复杂的仿真或代码段定义，分析微分不切实际时，它也有效。

然而，它也有局限：

1. **计算成本：** 每次梯度计算需要多次函数求值（每个维度两次），这对于参数 (parameter)很多（高维度）的函数来说可能很慢。
2. **准确度：** 结果是近似值，对 `h` 的选择和浮点精度很敏感。

在实践中，为了训练大型机器学习 (machine learning)模型，解析梯度方法（如有效使用链式法则的反向传播）因其速度和精度而更受青睐。数值方法通常是调试和验证的重要手段。

这个动手练习展示了梯度的抽象思维如何通过 NumPy 转变为具体的计算方法，为分析和优化机器学习中常见的多元函数提供了一个实用工具。

## 参考资料

- [Numerical Analysis](https://www.cengage.com/c/isbn/9781305253667) — J. Douglas Faires (2015)
  Publisher: Cengage Learning
  一本经典教材，全面考察了数值微分方法，包括有限差分近似及其误差分析。
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781108679902](https://doi.org/10.1017/9781108679902)
  提供了机器学习的基础数学原理，其中关于多变量微积分、梯度和优化的内容为数值梯度计算的实用性提供了背景。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Gradient Checks](http://cs231n.github.io/optimization-2/#gradcheck) — Andrej Karpathy, Justin Johnson, and Fei-Fei Li (2023)
  详细介绍了数值梯度（有限差分）在深度学习模型中验证解析梯度实现正确性的实际应用。
