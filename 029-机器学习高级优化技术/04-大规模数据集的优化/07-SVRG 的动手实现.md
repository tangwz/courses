# SVRG 的动手实现

来源：[原文](https://apxml.com/zh/courses/optimization-techniques-ml/chapter-4-large-scale-ml-optimization/hands-on-svrg-implementation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

随机方差削减梯度（SVRG）是一种有效方法，用于应对标准 SGD 中固有高方差。SVRG 实现的实践指南提供了观察其表现和了解实际考虑因素的机会。重点在于通过清晰的实现来突出算法的核心机制。

回顾我们对 SVRG 的讨论中的核心思想：定期计算整个数据集的完整梯度作为参考点。然后，在后续的随机步骤中，通过比较在当前参数 (parameter)和参考参数下评估的*相同*数据点的梯度来调整标准随机梯度。这种调整大幅降低了梯度估计的方差，通常使得收敛速度比 SGD 更快，特别是在按数据遍历次数衡量时。

### SVRG 算法结构

我们来概述一个典型的 SVRG 实现中涉及的步骤，用于最小化函数 $f(w) = \frac{1}{n} \sum_{i=1}^n f_i(w)$，其中 $f_i(w)$ 是数据点 $i$ 的损失：

1. **初始化：** 选择初始参数 (parameter)向量 (vector) $w_0$、学习率 $\eta > 0$ 和内循环迭代次数 $m$。值 $m$ 通常对应于在完整梯度计算之间执行的有效随机更新次数，常设置为与 $n$ 成比例（例如，$m=n$ 或 $m=2n$）。
2. **外循环：** 迭代 $S$ 个周期（即完整梯度计算的遍历）。
   a. **快照参数：** 将当前参数向量存储为快照 $\tilde{w} = w_{s-1}$（其中 $s$ 是外循环索引）。
   b. **计算完整梯度：** 使用快照参数计算整个数据集的平均梯度：$\mu = \nabla f(\tilde{w}) = \frac{1}{n} \sum_{i=1}^n \nabla f_i(\tilde{w})$。这是每个外循环执行一次的计算成本较高步骤。
   c. **初始化内循环参数：** 设置内循环的起始点：$w_0 = \tilde{w}$。
   d. **内循环：** $t$ 从 $0$ 迭代到 $m-1$。
   i. **采样数据点：** 从 $\{1, ..., n\}$ 中随机选择一个索引 $i_t$。
   ii. **计算随机梯度：** 计算选定数据点在*当前*参数 $w_t$ 和*快照*参数 $\tilde{w}$ 处的梯度：
   \* $\nabla f_{i_t}(w_t)$
   \* $\nabla f_{i_t}(\tilde{w})$
   iii. **更新参数：** 应用 SVRG 更新规则：
   $w_{t+1} = w_t - \eta \left( \nabla f_{i_t}(w_t) - \nabla f_{i_t}(\tilde{w}) + \mu \right)$
   e. **更新外循环参数：** 将内循环的结果设置为下一次外循环迭代的参数：$w_s = w_m$。（或者，可以从 $\{w_0, ..., w_m\}$ 中随机选择 $w_s$ 或使用平均值）。使用 $w_s = w_m$ 很常见。

### 为线性回归实现 SVRG

我们来实现 SVRG，以解决一个简单的最小二乘线性回归问题。对于数据 $(X, y)$，其中 $X \in \mathbb{R}^{n \times d}$ 且 $y \in \mathbb{R}^n$，我们的目标函数是：


$$
f(w) = \frac{1}{2n} \| Xw - y \|_2^2 = \frac{1}{n} \sum_{i=1}^n \frac{1}{2} (x_i^T w - y_i)^2
$$


这里，$f_i(w) = \frac{1}{2} (x_i^T w - y_i)^2$。单个数据点 $(x_i, y_i)$ 的梯度是：


$$
\nabla f_i(w) = (x_i^T w - y_i) x_i
$$


完整梯度是：


$$
\nabla f(w) = \frac{1}{n} X^T (Xw - y)
$$


下面是使用 NumPy 的 Python 实现。

```python
import numpy as np

def linear_loss(w, X, y):
  """计算均方误差损失。"""
  n_samples = X.shape[0]
  predictions = X.dot(w)
  loss = (1 / (2 * n_samples)) * np.sum((predictions - y)**2)
  return loss

def full_gradient(w, X, y):
  """计算线性回归的完整梯度。"""
  n_samples = X.shape[0]
  predictions = X.dot(w)
  gradient = (1 / n_samples) * X.T.dot(predictions - y)
  return gradient

def stochastic_gradient(w, X, y, index):
  """计算单个数据点的随机梯度。"""
  x_i = X[index]
  y_i = y[index]
  prediction_i = x_i.dot(w)
  # 如果它是平的，将其转换为列向量
  if x_i.ndim == 1:
      x_i = x_i[:, np.newaxis] # Make it a column vector if it's flat

  # 计算梯度: (预测值 - 实际值) * 特征向量
  # 确保标量 (prediction_i - y_i) 正确地乘以向量 x_i
  # 结果应与 w 具有相同的形状
  gradient_i = (prediction_i - y_i) * x_i.flatten() # Flatten x_i back if necessary or ensure result shape matches w

  # 确保 gradient_i 与 w 具有相同的形状（例如，(d,)）
  if w.ndim == 1 and gradient_i.shape != w.shape:
       gradient_i = gradient_i.reshape(w.shape)
  elif w.ndim > 1 and gradient_i.shape != w.shape: # 例如 w 是 (d, 1)
       gradient_i = gradient_i.reshape(w.shape)

  return gradient_i

def svrg_optimizer(X, y, n_outer_epochs, m, learning_rate):
  """实现线性回归的 SVRG 算法。"""
  n_samples, n_features = X.shape

  # 确保 y 是与预测一致的一维数组或列向量
  if y.ndim == 1:
      y = y # Keep as 1D array
  elif y.ndim == 2 and y.shape[1] == 1:
       y = y.flatten() # Make it 1D for easier calculations with predictions
  else:
       raise ValueError("y 必须是一维数组或列向量")

  # 初始化参数（例如，全零）
  w = np.zeros(n_features) 
  # 如果您偏好列向量参数:
  # w = np.zeros((n_features, 1))

  loss_history = []

  print(f"SVRG 开始：{n_outer_epochs} 个外层周期, m={m}, lr={learning_rate}")

  for s in range(n_outer_epochs):
    # 1. 快照与完整梯度计算
    w_snapshot = np.copy(w)
    mu = full_gradient(w_snapshot, X, y)

    # 存储当前损失（在内循环前使用快照计算）
    current_loss = linear_loss(w_snapshot, X, y)
    loss_history.append(current_loss)
    if s % 10 == 0 or s == n_outer_epochs - 1:
        print(f"外层周期 {s+1}/{n_outer_epochs}, 损失: {current_loss:.6f}")

    # 2. 内循环
    w_inner = np.copy(w_snapshot) # 从快照开始内循环
    for t in range(m):
      # 选择随机样本
      i_t = np.random.randint(0, n_samples)

      # 计算梯度
      grad_current = stochastic_gradient(w_inner, X, y, i_t)
      grad_snapshot = stochastic_gradient(w_snapshot, X, y, i_t)

      # SVRG 更新
      w_inner = w_inner - learning_rate * (grad_current - grad_snapshot + mu)

    # 更新外循环参数
    w = np.copy(w_inner) # 使用内循环的结果

  print("SVRG 完成。")
  return w, loss_history

# --- 示例用法 ---
# 生成合成数据
np.random.seed(42)
n_samples, n_features = 1000, 10
X = np.random.randn(n_samples, n_features)
true_w = np.random.randn(n_features) * 5 
y = X.dot(true_w) + np.random.randn(n_samples) * 0.5 # 添加一些噪声

# SVRG 参数
n_outer_epochs = 50  # 我们计算完整梯度的次数
m = n_samples        # 内循环步数（每个外层周期 1 次有效遍历）
learning_rate = 0.1 

# 运行 SVRG
w_svrg, svrg_loss_history = svrg_optimizer(X, y, n_outer_epochs, m, learning_rate)

print("\n学习到的权重 (SVRG):", w_svrg)
```

### 结果分析

为了解 SVRG 的表现，比较其与标准 SGD 的收敛情况是很有益的。您通常会为 SGD 实现一个类似的循环（仅使用 $\nabla f_{i_t}(w_t)$ 进行更新），并运行等效数量的梯度计算或有效数据遍历。

我们来可视化收敛速度的潜在差异。我们将绘制损失与有效数据遍历次数的关系图。对于 SVRG，每个外层周期包含一次完整梯度计算（相当于一次遍历）加上 $m$ 次随机梯度计算。如果 $m=n$，每个外层周期大约对应两次有效数据遍历（一次用于完整梯度，一次有效遍历用于 $n$ 次随机更新）。对于标准 SGD，$n$ 次随机更新构成一次有效遍历。



[交互图表：SVRG 与 SGD 收敛比较 (示意)](https://apxml.com/zh/courses/optimization-techniques-ml/chapter-4-large-scale-ml-optimization/hands-on-svrg-implementation#plot-5h80yx)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "scatter",
      "mode": "lines",
      "name": "SVRG",
      "x": [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18
      ],
      "y": [
        10,
        4,
        2,
        0.8,
        0.3,
        0.1,
        0.05,
        0.02,
        0.01,
        0.005
      ],
      "line": {
        "color": "#1c7ed6"
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "name": "SGD (示意)",
      "x": [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100
      ],
      "y": [
        15,
        8,
        5,
        3.5,
        2.8,
        2.4,
        2.1,
        1.9,
        1.8,
        1.75,
        1.7
      ],
      "line": {
        "color": "#f03e3e",
        "dash": "dot"
      }
    }
  ],
  "layout": {
    "title": "SVRG 与 SGD 收敛比较 (示意)",
    "xaxis": {
      "title": "有效数据遍历次数"
    },
    "yaxis": {
      "title": "损失 (MSE)",
      "type": "log"
    },
    "legend": {
      "x": 0.7,
      "y": 0.98
    }
  }
}
```

</details>



> SVRG 与典型 SGD 运行中每次有效数据遍历的损失降低示意性比较。SVRG 通常表现出更快的收敛速度，特别是在损失的对数尺度下。请注意，SGD 曲线是为示意目的手动创建的。请运行您自己的 SGD 实现，以便在相同数据上进行直接比较。

### 实现细节与调优

- **`m` 的选择：** 内循环迭代次数 $m$ 是一个重要的超参数 (parameter) (hyperparameter)。
  - 小的 $m$：完整梯度计算更频繁。这会保持低方差，但增加每次有效数据遍历的开销。
  - 大的 $m$：完整梯度计算不那么频繁。外层周期成本更低，但参数 $w_t$ 可能会偏离快照 $\tilde{w}$ 更远，如果 $m$ 过大，可能会减慢收敛。
  - 常见选择是 $m=n$ 或 $m=2n$，即在完整梯度计算之间有效执行 1 或 2 次随机更新。理论分析通常表明 $m$ 应与 $n$ 成比例。实验很重要。
- **学习率 $\eta$：** SVRG 通常可以容忍，甚至可能需要比 SGD 更大的学习率才能快速收敛。方差削减允许更激进的步骤。然而，过大的学习率仍可能导致不稳定。调整 $\eta$ 是必要的，通常通过网格搜索或在问题结构允许的情况下基于理论指导（例如，平滑常数知识）。
- **计算成本：** 主要成本是完整梯度计算（对于许多标准机器学习 (machine learning)模型为 $\mathcal{O}(nd)$），每个外层周期执行一次。内循环涉及 $m$ 次随机梯度计算，这通常便宜得多（每步为 $\mathcal{O}(d)$）。当完整梯度的成本定期可控时，SVRG 最有效，并且更快的收敛（更少的总有效遍历次数）带来的好处超过了与 SGD 较慢、高方差的进程相比的这种周期性成本。

此实践实现应为将 SVRG 应用于更大、更复杂的问题提供坚实基础。请记住，通常需要仔细调整 $m$ 和 $\eta$ 以获得最佳性能。在您自己的数据集上实验 SVRG 并将其与 SGD 进行比较，是了解其优点和实际表现的最佳方式。

## 参考资料

- [Stochastic Variance Reduced Gradient Methods for Large-scale Learning](https://proceedings.neurips.cc/paper_files/paper/2013/hash/c399862d3b9d6b7bd4d98a5fd98e7279-Paper.pdf) — Rie Johnson and Tong Zhang (2013)
  Journal: Advances in Neural Information Processing Systems 26 (NeurIPS 2013); Publisher: Neural Information Processing Systems Foundation; Volume: 26; Pages: 2697-2705; DOI: [10.5591/978-1-55860-968-3.neurips2013.C1](https://doi.org/10.5591/978-1-55860-968-3.neurips2013.C1)
  介绍SVRG算法及其理论收敛性保证的基础论文。
- [Convex Optimization: Algorithms and Complexity](https://www.nowpublishers.com/article/Details/MAL-050) — Sébastien Bubeck (2015)
  Journal: Foundations and Trends in Machine Learning; Publisher: Now Publishers; Volume: 8; Pages: 231-357; DOI: [10.1561/2200000050](https://doi.org/10.1561/2200000050)
  对凸优化算法进行严谨的理论处理，包含SVRG收敛性分析的专门章节。

---

[上一节](06-%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C%E7%AD%96%E7%95%A5.md) · [下一节](../05-%E5%88%86%E5%B8%83%E5%BC%8F%E4%BC%98%E5%8C%96%E7%AD%96%E7%95%A5/01-%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E7%9A%84%E5%8A%A8%E6%9C%BA.md)
