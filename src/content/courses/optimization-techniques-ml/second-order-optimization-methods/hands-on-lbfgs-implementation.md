---
course: "optimization-techniques-ml"
chapter: "second-order-optimization-methods"
lesson: "hands-on-lbfgs-implementation"
sourceId: 1257
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-2-second-order-optimization-methods/hands-on-lbfgs-implementation"
title: "动手实践：L-BFGS 的实现"
description: "关于在机器学习任务中实现或应用 L-BFGS 的实际操作环节。"
order: 8
plots: ["plots/1257-0.json"]
sourceHash: "a68f4c6ad69888f27fa86dbe90401b188afad0191f0e96823beb98f0358d70d2"
sourceCorrections: []
---

L-BFGS 在解决优化问题中的实际应用涉及实践操作，它建立在牛顿法和 BFGS 的概念之上。通过 L-BFGS 的实践，获取解决优化问题的实际操作经验，重点在于应用现有库的实现。这是机器学习 (machine learning)工作流中的一种常见方式。L-BFGS 的性能将与更简单的方法进行比较，并展示其实际用法。

回顾一下，L-BFGS 旨在获得与 BFGS 类似的高阶信息（曲率）优势，但避免了维护完整 $n \times n$ Hessian 矩阵近似的存储和计算负担。它通过仅存储最近的步长向量 (vector) $s_k = x_{k+1} - x_k$ 和梯度差向量 $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$ 的有限历史信息（通常为 5 到 20 对）来实现这一点。这些向量隐含地定义了在两阶段循环迭代中用于计算搜索方向 $p_k = -H_k \nabla f(x_k)$ 的 Hessian 矩阵近似，其中 $H_k$ 是逆 Hessian 矩阵近似。

### 建立优化问题

本次实际练习，我们使用一个标准的机器学习 (machine learning)问题：**二元逻辑回归**。目标是找到使负对数似然（或交叉熵损失）函数最小化的参数 (parameter) $\theta$。给定数据集 $(X, y)$，其中 $X \in \mathbb{R}^{m \times n}$（m 个样本，n 个特征），$y \in \{0, 1\}^m$，目标函数 $J(\theta)$ 为：


$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))]
$$


其中 $h_\theta(x) = \sigma(\theta^T x)$ 是 sigmoid 函数 $\sigma(z) = 1 / (1 + e^{-z})$。

我们还需要梯度 $\nabla J(\theta)$：


$$
\nabla J(\theta)_j = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$


为求简洁，我们将使用人工生成的数据集，使我们能够专注于优化器的表现。

```python
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # Using for basic plotting, Plotly for final comparison

# 生成人工数据
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=2,
                           n_classes=2, random_state=42)
# 添加截距项
X = np.hstack((np.ones((X.shape[0], 1)), X))
n_features = X.shape[1]

# 分割数据（可选，良好实践，但对于优化器演示并非严格需要）
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 为简洁起见，此处使用完整数据集
X_train, y_train = X, y
m = X_train.shape[0]

# --- 目标函数和梯度 ---
def sigmoid(z):
    # 裁剪 z 以避免指数运算中的溢出/下溢
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def cost_function(theta, X, y):
    h = sigmoid(X @ theta)
    # 为对数运算中的数值稳定性添加一个小 epsilon
    epsilon = 1e-7
    cost = -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

def gradient(theta, X, y):
    h = sigmoid(X @ theta)
    grad = (X.T @ (h - y)) / len(y)
    return grad

# 初始参数
initial_theta = np.zeros(n_features)

print(f"数据集形状: {X_train.shape}")
print(f"初始成本: {cost_function(initial_theta, X_train, y_train):.4f}")
```

### 通过 SciPy 使用 L-BFGS

大多数科学计算库都提供了 L-BFGS 的可靠实现。我们将使用 `scipy.optimize.minimize`，这是一个多功能函数，可与多种优化算法交互。常使用的特定变体是 L-BFGS-B，它允许箱型约束（参数 (parameter)的下限和上限），尽管我们在此不使用约束。

要通过 L-BFGS 使用 `scipy.optimize.minimize`，我们需要提供：

1. `fun`：要最小化的目标函数（我们的 `cost_function`）。
2. `x0`：参数的初始猜测（我们的 `initial_theta`）。
3. `args`：目标函数和梯度函数所需的额外参数元组（我们的 `X_train`，`y_train`）。
4. `method`：指定 'L-BFGS-B'。
5. `jac`：计算梯度向量 (vector)的函数（我们的 `gradient`）。提供解析梯度比依赖数值近似效率高得多。
6. `options`：算法特定参数的字典。L-BFGS-B 的重要参数包括：
   - `maxcor`：要存储的最新 (s, y) 对的数量（历史大小 `m`）。默认值为 10。
   - `gtol`：收敛的梯度范数容差。当 $\max | \nabla J(\theta)_i | \leq \text{gtol}$ 时，算法停止。
   - `maxiter`：最大迭代次数。

```python
# --- 运行 L-BFGS ---
print("\n运行 L-BFGS 中...")

# 存储历史数据以供绘图
lbfgs_cost_history = []

def callback_fn(theta):
  """回调函数，用于存储每次迭代的成本。"""
  cost = cost_function(theta, X_train, y_train)
  lbfgs_cost_history.append(cost)
  # 可选：打印进度
  # print(f"迭代 {len(lbfgs_cost_history)}: 成本 = {cost:.6f}")

lbfgs_result = minimize(fun=cost_function,
                        x0=initial_theta,
                        args=(X_train, y_train),
                        method='L-BFGS-B',
                        jac=gradient,
                        callback=callback_fn,
                        options={'maxcor': 15, # 历史大小 m
                                 'gtol': 1e-6, # 梯度容差
                                 'disp': True, # 显示收敛信息
                                 'maxiter': 200
                                }
                       )

# --- 结果 ---
print("\nL-BFGS 结果:")
if lbfgs_result.success:
    print(f"优化成功: {lbfgs_result.message}")
    print(f"最终成本: {lbfgs_result.fun:.6f}")
    print(f"迭代次数: {lbfgs_result.nit}")
    print(f"函数评估次数: {lbfgs_result.nfev}")
    print(f"梯度评估次数: {lbfgs_result.njev}")
    # optimal_theta_lbfgs = lbfgs_result.x
else:
    print(f"优化失败: {lbfgs_result.message}")

# 为绘图目的，在开头添加初始成本
lbfgs_cost_history.insert(0, cost_function(initial_theta, X_train, y_train))
```

运行此代码将输出收敛信息、最终成本以及迭代次数和函数/梯度评估次数。请注意，对于逻辑回归这样表现良好的问题，与一阶方法相比，L-BFGS 通常在相对较少的迭代次数内收敛。

### 与梯度下降 (gradient descent)法的比较

为了体会 L-BFGS 的效率，我们来实现在相同问题上运行一个简单的梯度下降 (GD) 算法。我们需要选择一个学习率 $\alpha$，并运行固定次数的迭代或直到收敛。

```python
# --- 简单的梯度下降实现 ---
def gradient_descent(X, y, initial_theta, learning_rate, n_iterations):
    theta = initial_theta.copy()
    cost_history = []
    for i in range(n_iterations):
        cost = cost_function(theta, X, y)
        cost_history.append(cost)
        grad = gradient(theta, X, y)
        theta = theta - learning_rate * grad
        if i % 100 == 0: # 偶尔打印进度
             print(f"GD 迭代 {i}: 成本 = {cost:.6f}")
    print(f"GD 最终成本（经过 {n_iterations} 次迭代后）: {cost_history[-1]:.6f}")
    return theta, cost_history

print("\n运行梯度下降中...")
learning_rate = 0.1
n_iterations_gd = 500 # 通常需要比 L-BFGS 多得多的迭代次数

# 运行 GD
gd_theta, gd_cost_history = gradient_descent(X_train, y_train, initial_theta, learning_rate, n_iterations_gd)
```

### 收敛过程可视化

现在，让我们使用 Plotly 可视化 L-BFGS 和梯度下降 (gradient descent)每次迭代的成本下降情况。这清晰展示了收敛行为的差异。



![L-BFGS 与梯度下降法的收敛比较](plots/1257-0.json)



> 逻辑回归成本最小化的收敛比较。与标准梯度下降法相比，L-BFGS 通常需要少得多的迭代次数才能达到较低的成本值，即便梯度下降法使用了合理调整的学习率。

正如图示通常显示的那样，L-BFGS 在迭代次数方面收敛得快得多。虽然每次 L-BFGS 迭代涉及的计算（两阶段循环迭代和通常更精密的线搜索）比 GD 迭代更多，但达到所需精度所需的梯度评估总次数和总时间通常会显著减少，尤其是在梯度计算成本高昂或曲率信息为搜索方向提供了有力指引的问题上。

### L-BFGS 的实际考量

- **历史大小（`m` 或 `maxcor`）：** 这是主要的调整参数 (parameter)。更大的 `m` 意味着每次迭代需要更多内存和计算，但可能形成更好的 Hessian 矩阵近似并可能更快收敛（更少迭代次数）。典型值介于 5 到 20 之间。值过大则相比 BFGS 内存节省效果会降低。
- **线搜索：** L-BFGS 依赖线搜索算法（例如满足 Wolfe 条件的算法）来确定沿计算出的搜索方向 $p_k$ 的步长。库的实现会在内部处理这一点，但其质量会影响性能。不佳的线搜索可能导致收敛缓慢或不稳定。
- **梯度精度：** L-BFGS 对梯度计算的精度敏感。请确保您的梯度实现是正确的。使用数值近似计算梯度是可行的，但通常效率低得多，并可能阻碍收敛。
- **缩放：** 和大多数优化算法一样，L-BFGS 可以从特征缩放（例如，标准化）中受益，这可以改善问题的条件数，使损失面更均匀，更容易找到路径。

本次实际练习演示了如何通过 SciPy 等标准库有效使用 L-BFGS 这一强大的拟牛顿法。它能够在不产生过高内存成本的情况下整合曲率信息，这使其成为各种中到大型机器学习 (machine learning)优化问题的一个有价值的工具，通常比一阶方法提供显著更快的收敛速度。

## 参考资料

- [Updating Quasi-Newton Matrices with Limited Storage](https://www.ams.org/journals/mcom/1980-35-151/S0025-5718-1980-0572855-7/S0025-5718-1980-0572855-7.pdf) — Jorge Nocedal (1980)
  Journal: Mathematics of Computation; Volume: 35; Pages: 773-782; DOI: [10.2307/2007613](https://doi.org/10.2307/2007613)
  介绍L-BFGS算法的原始论文，详细阐述了其数学基础和内存高效方法。
- [Numerical Optimization](https://www.springer.com/aam/book/978-0-387-30303-1) — Jorge Nocedal and Stephen J. Wright (2006)
  Publisher: Springer
  一本标准教科书，提供了优化算法的广泛理论和实践处理，包括关于拟牛顿法和L-BFGS的专门章节。
- [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) — SciPy Developers (2024)
  SciPy官方文档，用于本实践中使用的函数，解释了其参数和L-BFGS-B的用法。
- [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) — Stephen Boyd and Lieven Vandenberghe (2004)
  Publisher: Cambridge University Press
  本书对凸优化进行了严谨处理，凸优化是许多机器学习问题（包括逻辑回归）的基础，并为L-BFGS等算法提供了理论背景。
