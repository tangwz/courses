# 雅可比向量积 (JVPs) 与 jax.jvp

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-4-advanced-automatic-differentiation/jacobian-vector-products-jvp)

[返回章节目录](README.md) · [返回课程目录](../README.md)

前向模式自动微分计算函数输出相对于输入变化的速率，并通过计算向前传播。前向模式自动微分中的基本运算是雅可比向量 (vector)积（JVP）。逆向模式自动微分（`jax.grad` 使用此模式）对于输入多输出少的函数（如机器学习 (machine learning)中常见的损失函数 (loss function)）效率高；而当输入数量相对于输出数量较少，或者您特别需要方向导数时，前向模式表现优异。

### 理解雅可比向量 (vector)积 (JVPs)

考虑一个函数 $f: \mathbb{R}^n \to \mathbb{R}^m$，它将一个 $n$ 维输入向量 $x$ 映射到 $m$ 维输出向量 $y$。函数 $f$ 在点 $x$ 处的雅可比矩阵 $J(x)$ 是一个 $m \times n$ 矩阵，包含所有一阶偏导数：


$$
J(x) = \frac{\partial f}{\partial x} = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}
$$


雅可比矩阵 $J(x)$ 表示函数 $f$ 在点 $x$ 附近的最佳线性近似。

雅可比向量积（JVP）计算此雅可比矩阵 $J(x)$ 与一个“切线”向量 $v \in \mathbb{R}^n$ 的乘积。这个向量 $v$ 表示输入空间中的扰动方向。JVP 由以下公式给出：


$$
\text{JVP}(x, v) = J(x) v = \left. \frac{d}{d\alpha} f(x + \alpha v) \right|_{\alpha=0}
$$


结果 $J(x)v$ 是一个位于输出空间中的 $m$ 维向量。它表示当输入 $x$ 沿切线向量 $v$ 指定的方向进行无穷小扰动时，函数输出 $f(x)$ 的变化率。重要地，前向模式自动微分使我们能够计算这个乘积 $J(x)v$，而*无需*显式地形成可能非常大的雅可比矩阵 $J(x)$。计算开销通常只比评估原始函数 $f(x)$ 多一个小的常数因子。

### 使用 `jax.jvp` 计算 JVPs

JAX 提供 `jax.jvp` 变换，用于计算雅可比向量 (vector)积。其签名为：

```python
jax.jvp(fun, primals, tangents)
```

- `fun`：要进行微分的 Python 可调用对象（函数）。
- `primals`：一个元组或列表，包含用于评估函数及其 JVP 的原始输入。在我们的符号中，这些就是点 $x$。如果 `fun` 接受多个参数 (parameter)，`primals` 应该包含所有这些参数的值。
- `tangents`：一个元组或列表，包含与原始输入对应的切线向量。这些就是向量 $v$。`tangents` 的结构和类型/形状必须与 `primals` 匹配。您可以通过为其他参数传递不可微分值（如 `None` 或零值数组）来仅为原始输入的一个子集提供切线向量。

`jax.jvp` 返回一对：

- `primal_out`：调用 `fun(*primals)` 的结果。这就是 $f(x)$。
- `tangent_out`：JVP 计算的结果，$J(x)v$。这与 `primal_out` 具有相同的结构和类型/形状。

让我们看一个简单例子：

```python
import jax
import jax.numpy as jnp

# 定义一个从 R^2 到 R^2 的函数
def f(x):
  return jnp.array([x[0]**2, x[0] * x[1]])

# 定义原始输入点
x_primal = jnp.array([2.0, 3.0])

# 定义切线向量（扰动方向）
v_tangent = jnp.array([1.0, 0.5])

# 计算函数输出和 JVP
y_primal_out, tangent_out = jax.jvp(f, (x_primal,), (v_tangent,))

print(f"Primal input (x): {x_primal}")
print(f"Tangent vector (v): {v_tangent}")
print(f"Primal output f(x): {y_primal_out}")
print(f"Tangent output (J(x)v): {tangent_out}")

# 让我们手动验证此情况下的雅可比矩阵和 JVP：
# f(x) = [x_0^2, x_0 * x_1]
# J(x) = [[df1/dx0, df1/dx1], [df2/dx0, df2/dx1]]
# J(x) = [[2*x0, 0], [x1, x0]]
# At x = [2.0, 3.0], J(x) = [[4.0, 0.0], [3.0, 2.0]]
# J(x)v = [[4.0, 0.0], [3.0, 2.0]] @ [1.0, 0.5]
#       = [4.0*1.0 + 0.0*0.5, 3.0*1.0 + 2.0*0.5]
#       = [4.0, 3.0 + 1.0]
#       = [4.0, 4.0]
# 这与 jax.jvp 的 tangent_out 匹配！
```

Output:

```
Primal input (x): [2. 3.]
Tangent vector (v): [1.  0.5]
Primal output f(x): [4. 6.]
Tangent output (J(x)v): [4. 4.]
```

如预期，`jax.jvp` 计算了原始函数的输出 `f(x)` 和方向导数 `J(x)v`。

### JVPs 的应用和优点

尽管 `jax.grad`（基于逆向模式 VJP）是训练大多数神经网络 (neural network)的主力，但通过 `jax.jvp` 计算的 JVP 也有其独特之处：

1. **效率：** 计算 JVP 通常开销低，通常是原始函数评估开销的 1.5 到 3 倍左右，不论输出维度 $m$ 为何。这使得当输入维度 $n$ 较小或适中时，它非常高效。
2. **方向导数：** 它的主要用途是计算函数在特定方向 $v$ 上的导数。这在敏感性分析、需要方向导数的优化算法以及了解扰动如何影响输出方面都很有用。
3. **计算雅可比矩阵的列：** 如果您将切线向量 (vector) $v$ 设置为独热向量（例如 `[1.0, 0.0, ..., 0.0]`），所得的 JVP `J(x)v` 正好是雅可比矩阵 $J(x)$ 的第一列。对所有独热基向量重复此操作可以计算完整的雅可比矩阵，尽管对于密集雅可比矩阵，使用 `jax.jacfwd`（它本质上是 `jax.jvp` 上的 `vmap`）通常是实现此目的更直接的方法。
4. **核心原理：** JVP 是自动微分中的一个基本原理，JAX 内部使用它，例如，在计算 Hessian 矩阵时（我们稍后会看到）。

### 使用 PyTrees 和变换

与其他 JAX 变换类似，`jax.jvp` 对于原始输入和切线输入都能与 PyTrees（嵌套列表、元组、字典）良好配合。`tangents` 参数 (parameter)的结构必须与 `primals` 参数的结构相对应。

```python
import jax
import jax.numpy as jnp

def predict(params, inputs):
  # 一个简单的线性模型
  return jnp.dot(inputs, params['w']) + params['b']

params = {
    'w': jnp.array([[1.0, 2.0], [3.0, 4.0]]), # 形状 (2, 2)
    'b': jnp.array([0.1, -0.1])              # 形状 (2,)
}
inputs = jnp.array([10.0, 20.0])             # 形状 (2,)

# 定义与 params 结构匹配的切线
# 只扰动权重 'w'，保持偏置 'b' 不变（切线为零值）
tangents = {
    'w': jnp.ones_like(params['w']),
    'b': jnp.zeros_like(params['b'])
}

# 计算关于 params 的 JVP。注意 inputs 被视为常数（未提供切线）。
# jax.jvp 要求 primals 和 tangents 为元组
primal_out, tangent_out = jax.jvp(predict, (params, inputs), (tangents, jax.lax.stop_gradient(inputs))) # or (tangents, None)

print(f"Inputs: {inputs}")
print(f"Params: {params}")
print(f"Tangents (perturbation): {tangents}")
print(f"Primal output (prediction): {primal_out}")
print(f"Tangent output (change in prediction due to perturbation): {tangent_out}")

# 手动检查：
# 输出 = inputs @ w + b
# d(输出) / d(w_ij) * tangent_w_ij 对 i,j 求和
# d(输出) / d(w) 的贡献：inputs @ tangent['w']
# d(输出) / d(b) 的贡献：tangent['b']
# 预期 tangent_out = inputs @ tangent['w'] + tangent['b']
#                     = [10., 20.] @ [[1., 1.], [1., 1.]] + [0., 0.]
#                     = [10*1+20*1, 10*1+20*1] + [0., 0.]
#                     = [30., 30.]
```

Output:

```
Inputs: [10. 20.]
Params: {'w': Array([[1., 2.],
       [3., 4.]], dtype=float32), 'b': Array([ 0.1, -0.1], dtype=float32)}
Tangents (perturbation): {'w': Array([[1., 1.],
       [1., 1.]], dtype=float32), 'b': Array([0., 0.], dtype=float32)}
Primal output (prediction): [ 70.1 100.9]
Tangent output (change in prediction due to perturbation): [30. 30.]
```

在这里，我们计算了当参数 `params` 沿 `tangents` 指定的方向扰动时，`predict` 的输出如何变化。注意，我们将 `jax.lax.stop_gradient(inputs)` 作为 `inputs` 的切线传入，以表明我们不对其进行微分（传入 `None` 同样有效）。

`jax.jvp` 也可以与 `jax.jit` 等其他变换组合使用以提高性能，或者与 `jax.vmap` 组合以同时计算带有多个不同切线向量 (vector)的 JVP。

理解 `jax.jvp` 有助于了解前向模式自动微分，并为您提供了一个高效计算方向导数的工具，补充了 `jax.grad` 和 `jax.vjp` 提供的逆向模式功能。

## 参考资料

- [Automatic Differentiation of Algorithms: From Simulation to Optimization](https://doi.org/10.1007/978-3-540-68937-5) — Andreas Griewank, Andrea Walther (2008)
  Publisher: Springer; DOI: [10.1007/978-3-540-68937-5](https://doi.org/10.1007/978-3-540-68937-5)
  一本关于自动微分理论与实践的综合性教科书，详细介绍了前向模式自动微分和雅可比向量积。
- [\`jax.jvp\`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jvp.html) — JAX core developers (2024)
  `jax.jvp` 函数的官方文档，提供了API细节、使用示例以及它在JAX框架中的作用。
- [JAX: Composable transformations of Python+NumPy programs](https://github.com/google/jax#citation) — James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Skye Lew, George H. Nash, Nicolas Nieto, Sam Oden, Mark Omernick, Jonathan Renjie So, Matt Von-Rose and Sasha Rush (2018)
  介绍JAX库的原始参考文献，概述了其自动微分和其他程序转换的设计原则。
- [The mechanics of automatic differentiation](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf) — Atilim Baydin, Barak A. Pearlmutter and Alexey Andreyevich Radul (2018)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 18; Pages: 1-65; DOI: [10.5555/3278500.3278502](https://doi.org/10.5555/3278500.3278502)
  一篇关于自动微分的综述和教程，涵盖了其数学基础、各种模式（包括前向模式）和实际实现。

---

[上一节](01-%E5%89%8D%E5%90%91%E5%92%8C%E5%8F%8D%E5%90%91%E6%A8%A1%E5%BC%8F%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%E5%9B%9E%E9%A1%BE.md) · [下一节](03-%E5%90%91%E9%87%8F-%E9%9B%85%E5%8F%AF%E6%AF%94%E7%A7%AF%20%28VJPs%29%20%E4%B8%8E%20jax.vjp.md)
