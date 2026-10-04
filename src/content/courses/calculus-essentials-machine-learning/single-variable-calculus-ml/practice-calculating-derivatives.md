---
course: "calculus-essentials-machine-learning"
chapter: "single-variable-calculus-ml"
lesson: "practice-calculating-derivatives"
sourceId: 1368
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-2-single-variable-calculus-ml/practice-calculating-derivatives"
title: "动手实践：使用 Python 计算导数"
description: "使用 SymPy 或 NumPy 等 Python 库计算函数的导数。"
order: 7
plots: []
sourceHash: "8491497e8fb72a4560c6509889950303d1e5cecc75e3b4f1f3b74daaac9a329d"
sourceCorrections: []
---

虽然理解微分规则并进行手动计算是必要的，但在机器学习 (machine learning)中运用这些原理时，常会遇到复杂的函数或在大型算法中需要重复计算。幸运的是，Python 提供了强大的库来自动计算导数，既可以符号计算（提供精确的解析式），也可以数值计算（近似导数的值）。这里将说明如何使用这些工具。

### 使用 SymPy 进行符号微分

SymPy 是一个用于符号数学的 Python 库。它允许我们使用符号定义数学表达式，并执行微分、积分和方程求解等运算，同时保持精确的符号表示。这对于找到导数的精确形式非常有益。

首先，请确保安装了 SymPy (`pip install sympy`)。让我们从定义一个符号变量和一个函数开始。

```python
import sympy

# 将 'x' 定义为符号变量
x = sympy.symbols('x')

# 定义一个函数，例如 f(x) = x^3 + 2*x^2 - 5*x + 1
f = x**3 + 2*x**2 - 5*x + 1

print(f"原始函数: f(x) = {f}")
```

现在，我们可以使用 SymPy 的 `diff` 函数来计算 $f(x)$ 对 $x$ 的导数。

```python
# 计算 f 对 x 的导数
f_prime = sympy.diff(f, x)

print(f"导数: f'(x) = {f_prime}")
```

SymPy 正确应用微分规则（幂法则、和法则），为我们提供了精确的解析导数：$f'(x) = 3x^2 + 4x - 5$。

我们还可以使用 SymPy 在特定点评估导数。让我们找到 $x = 2$ 处切线的斜率。

```python
# 在 x = 2 处评估导数
value_at_2 = f_prime.subs(x, 2)

print(f"导数在 x = 2 处的值: f'(2) = {value_at_2}")
```

结果 $f'(2) = 3(2)^2 + 4(2) - 5 = 12 + 8 - 5 = 15$，这表示函数 $f(x)$ 在 $x=2$ 点的瞬时变化率。

SymPy 也可以处理涉及三角函数、指数函数和对数函数部分的更复杂函数。

```python
import sympy
import numpy as np # 经常与 SymPy 一起使用

# 将 x 定义为符号
x = sympy.symbols('x')

# 定义一个更复杂的函数: g(x) = sin(x) * exp(-x^2 / 2)
g = sympy.sin(x) * sympy.exp(-x**2 / 2)
print(f"\n另一个函数: g(x) = {g}")

# 计算其导数
g_prime = sympy.diff(g, x)
print(f"导数: g'(x) = {g_prime}")

# 在 x = 0 处评估导数
# 如果需要数值评估，使用 evalf()
value_at_0 = g_prime.subs(x, 0).evalf()
print(f"导数在 x = 0 处的值: g'(0) = {value_at_0}")
```

当你需要导数的精确数学表达式或高精度时，符号微分非常适合。

### 使用 SciPy 进行数值微分

有时，我们可能没有函数的解析表达式（它可能是复杂模拟或外部系统的结果），或者符号推导的计算成本可能非常高。在这种情况下，我们可以数值地近似导数。

主要思想是使用导数的定义，$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$，但我们不取极限，而是使用一个非常小且非零的 $h$ 值。一种常见且通常更准确的方式是中心差分公式：

$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$

SciPy 库（通常与 NumPy 一起安装，`pip install scipy`）为数值微分提供了便捷的函数。`scipy.derivative` 是一个不错的选择。

让我们使用相同的函数 $f(x) = x^3 + 2x^2 - 5x + 1$，并近似其在 $x = 2$ 处的导数。

```python
import numpy as np
from scipy.misc import derivative # 或者在新版本中使用 from scipy import derivative

# 将函数 f(x) 定义为常规 Python 函数
def f_numeric(x):
  return x**3 + 2*x**2 - 5*x + 1

# 评估导数的点
point = 2.0

# 步长（公式中的 h）。较小的值通常能提供更高的准确度，
# 但过小可能导致浮点错误。
step_size = 1e-6

# 使用 SciPy 计算数值导数
f_prime_numeric = derivative(f_numeric, point, dx=step_size)

print(f"\nf'(x) 在 x = {point} 处的数值近似")
print(f"f'({point}) ≈ {f_prime_numeric}")

# 与 SymPy 的精确值比较
exact_value = 15
print(f"精确值: f'({point}) = {exact_value}")
print(f"近似误差: {abs(f_prime_numeric - exact_value)}")
```

如您所见，数值近似值非常接近我们使用 SymPy 找到的精确值 (15)。这种小差异是由于有限差分方法固有的近似以及潜在的浮点不准确性。

当处理没有直接符号表示或不切实际的函数时，数值微分尤其有意义。但是，请注意步长 (`dx`) 的选择和潜在的精度限制。

### 与优化算法的关联

为什么计算导数在机器学习 (machine learning)中如此有必要？正如本章前面和第 1 章中讨论的那样，优化算法包括找到能够最小化成本函数或损失函数 (loss function)（如模型中的权重 (weight)和偏差）的参数 (parameter)。这个成本函数通常衡量模型表现的差劲程度。

导数（或其多变量对应物，梯度，我们将在下一章中介绍）显示了成本函数的斜率。通过计算导数，梯度下降 (gradient descent)等优化算法能够知道调整参数的方向，以降低成本，有效地“沿着”斜率向最小值“下降”。因此，能够高效地计算这些导数，无论是符号计算还是数值计算，对于训练许多机器学习模型来说都必不可少。

这种使用 SymPy 和 SciPy 等 Python 库计算导数的实际能力，使我们摆脱了繁琐的手动计算，并让我们能够专注于构建和优化机器学习模型。当我们转向多变量函数时，这些计算方法将变得更重要。

## 参考资料

- [SymPy Documentation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHk9e3lp4_9uA-g3Vsu7jRVyK6j0K866kth-K_PHj5jeSrLVePRXqhM6_s7CtblZbVUTqFSrj_E1jswm6jkNHazlJTzo2-32h-gJfN4c2fBgFU=) — SymPy Development Team (2025)
  使用 SymPy 进行符号数学计算（包括微分和表达式操作）的官方资料。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面解释了微积分，特别是导数和梯度在梯度下降等机器学习优化算法中的应用。
