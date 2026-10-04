---
course: "advanced-jax"
chapter: "advanced-automatic-differentiation"
lesson: "vector-jacobian-products-vjp"
sourceId: 2870
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-4-advanced-automatic-differentiation/vector-jacobian-products-vjp"
title: "向量-雅可比积 (VJPs) 与 jax.vjp"
description: "使用 `jax.vjp` 计算 VJP，这是 `jax.grad` 的根基，并理解其工作原理。"
order: 3
plots: []
sourceHash: "223129dc79beff6858028557250716ea1614b88541efc9e75256114f3dbde1f4"
sourceCorrections: []
---

反向模式自动微分是深度学习 (deep learning)中梯度下降 (gradient descent)的核心工具。它能高效计算一个标量输出（比如损失函数 (loss function)）相对于可能数百万个输入（模型参数 (parameter)）的梯度。尽管 `jax.grad` 提供了一种用户友好的方式来获取这些梯度，但理解其运作原理，即向量 (vector)-雅可比积 (VJP)，能够实现更高级的功能。`jax.vjp` 是 JAX 中用于计算 VJP 的函数。

### 反向模式的要旨：向量 (vector)-雅可比积

回顾一下，对于一个函数 $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$，它在点 $x \in \mathbb{R}^n$ 处的雅可比矩阵 $J_f(x)$ 包含所有偏导数 $\frac{\partial f_i}{\partial x_j}$。雅可比矩阵是一个 $m \times n$ 的矩阵。

反向模式自动微分计算一个 *余切* 向量 $v \in \mathbb{R}^m$ 与雅可比矩阵的乘积，得到 $v^T J_f(x)$。这个乘积就是向量-雅可比积 (VJP)。结果是一个 $\mathbb{R}^n$ 中的向量（或者严格来说，是一个 $1 \times n$ 的行向量）。

这为什么有用？考虑一个典型的机器学习 (machine learning)情境，它涉及到一系列函数组合，最终会产生一个标量损失 $L$。令 $f$ 是这条链中的最后一个函数，它将某个中间表示 $y \in \mathbb{R}^m$ 映射到最终的标量损失 $L \in \mathbb{R}$。因此，$L = f(y)$。梯度 $\nabla_y L$ 是一个行向量 $\frac{\partial L}{\partial y} \in \mathbb{R}^{1 \times m}$。如果 $y = g(x)$ 且 $x \in \mathbb{R}^n$，链式法则告诉我们 $\nabla_x L = \nabla_y L \cdot J_g(x)$。这正是一个 VJP！向量 $v^T$ 是从输出反向传播 (backpropagation)的梯度（或余切），而 $J_g(x)$ 是我们想要其输入梯度的函数 $g$ 的雅可比矩阵。

反向模式通过在计算图中反向传播这些余切向量来工作，对于一个标量损失函数 (loss function) $L$，从初始余切 $\frac{dL}{dL} = 1.0$ 开始。

### 使用 `jax.vjp` 计算 VJP

JAX 通过 `jax.vjp` 提供 VJP 计算的直接访问接口。其基本签名如下：

```python
y, vjp_fun = jax.vjp(fun, *primals)
```

我们来分解一下：

1. `fun`: 要进行微分的 Python 可调用对象（函数）。
2. `*primals`: 用于评估函数和计算 VJP 的输入参数 (parameter)（`x`, `y` 等）。这些是隐式计算雅可比矩阵的点。
3. `y`: 函数在原像（primals）处评估的输出，即 `y = fun(*primals)`。这是前向传播的结果。
4. `vjp_fun`: 一个计算 VJP 的 *新函数*（一个闭包）。这个函数接受一个参数，即余切向量 (vector) `v`，它必须与输出 `y` 具有相同的结构（形状和数据类型）。当作为 `vjp_fun(v)` 调用时，它返回 VJP：$v^T J_{\text{fun}}(\text{primals})$。结果是一个元组，包含针对每个原像输入的计算梯度。

### 示例：标量函数

让我们从一个简单的函数 $f(x, y) = x^2 \sin(y)$ 开始。

```python
import jax
import jax.numpy as jnp

def f(x, y):
  return x**2 * jnp.sin(y)

# 定义原始输入
x_primal = 3.0
y_primal = jnp.pi / 2.0

# 计算 VJP
y_out, vjp_fun = jax.vjp(f, x_primal, y_primal)

print(f"前向传播输出 y: {y_out}")

# 输出 y 是标量，所以余切 v 也是标量。
# 我们使用 v = 1.0，这相当于计算 d(1.0 * y) / dx 和 d(1.0 * y) / dy
# 这等同于找到 y 的梯度。
cotangent_v = 1.0
primal_grads = vjp_fun(cotangent_v)

print(f"关于 x 的梯度: {primal_grads[0]}") # 应该为 2 * x * sin(y) = 2 * 3.0 * sin(pi/2) = 6.0
print(f"关于 y 的梯度: {primal_grads[1]}") # 应该为 x^2 * cos(y) = 3.0^2 * cos(pi/2) = 0.0

# 与 jax.grad 比较
grad_f = jax.grad(f, argnums=(0, 1)) # 获取关于 x 和 y 的梯度
grads_via_grad = grad_f(x_primal, y_primal)

print(f"通过 jax.grad 获得的梯度: {grads_via_grad}")
```

在此情况下，调用 `vjp_fun(1.0)` 得到的结果与 `jax.grad(f, argnums=(0, 1))(x, y)` 相同。这表明了一个基本联系：针对标量函数的 `jax.grad` 实质上是 `jax.vjp` 的一个便捷封装，它自动提供初始余切 `1.0`。

### 示例：非标量函数

现在考虑一个向量 (vector)输出的函数：$g(x) = (x_0^2, x_1^3)$，其中 $x = (x_0, x_1)$。

```python
import jax
import jax.numpy as jnp

def g(x):
  return jnp.array([x[0]**2, x[1]**3])

# 定义原始输入
x_primal = jnp.array([2.0, 3.0])

# 计算 VJP
y_out, vjp_fun = jax.vjp(g, x_primal)

print(f"前向传播输出 y: {y_out}") # 应该为 [4., 27.]

# 输出 y 是一个形状为 (2,) 的向量，因此余切 v 也必须是形状为 (2,) 的向量
cotangent_v = jnp.array([10.0, 20.0]) # 一个任意的余切向量

# 计算 VJP: v^T J
primal_grad = vjp_fun(cotangent_v)

print(f"VJP 结果 (关于 x 的梯度): {primal_grad}")
# 预期：
# 雅可比矩阵 J = [[2*x0, 0], [0, 3*x1^2]] = [[4, 0], [0, 27]]
# v^T J = [10, 20] * [[4, 0], [0, 27]] = [10*4, 20*27] = [40, 540]
# jax.vjp 返回一个元组，即使只有一个原始输入，所以结果是 ([40., 540.],)
```

在这里，`vjp_fun` 接收向量 `cotangent_v` 并将其乘以在 `x_primal` 处评估的函数 `g` 的雅可比矩阵，从而产生返回给 `x` 的梯度贡献。余切的形状必须与函数输出 `y_out` 的形状匹配。

### 为什么直接使用 `jax.vjp`？

尽管 `jax.grad` 足以应对需要标量损失梯度的标准梯度下降 (gradient descent)情况，但 `jax.vjp` 提供更多控制：

1. **非标量梯度的效率：** 如果你需要同时处理多个输出的梯度或想操作中间梯度，VJP 有时可以比重复调用 `grad` 更高效地组织。
2. **自定义梯度规则：** 正如您将在本章后面看到的那样，`jax.custom_vjp` 允许您为特定函数定义定制的 VJP 规则。理解 `jax.vjp` 是此项操作的必要条件。
3. **实现复杂的 AD 逻辑：** 对于需要复杂梯度操作的任务（例如，某些优化算法，高级自动微分研究），直接访问 VJP 是必不可少的。
4. **教学理解：** 使用 `jax.vjp` 可以清楚地了解反向模式 AD 如何运作以及 `jax.grad` 如何构建。

总而言之，`jax.vjp` 为 JAX 强大的反向模式自动微分系统提供了一个更低级的接口。它计算向量 (vector)-雅可比积，这代表了反向传播 (backpropagation)的核心操作。尽管 `jax.grad` 处理标量损失梯度的常见情况，但掌握 `jax.vjp` 会为您提供处理更复杂的微分任务和更全面理解该系统的工具。

## 参考资料

- [jax.vjp](https://jax.readthedocs.io/en/latest/_autosummary/jax.vjp.html) — JAX documentation (2024)
  提供 `jax.vjp` 函数的官方API参考和使用详情，用于直接计算向量-雅可比积。
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://epubs.siam.org/doi/book/10.1137/1.9780898717761) — Andreas Griewank and Andrea Walther (2008)
  Publisher: Society for Industrial and Applied Mathematics; DOI: [10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)
  一本关于算法微分的权威教科书，严谨地阐述了反向模式自动微分及其计算效率的数学原理。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第6章解释了反向传播，这是反向模式自动微分的一个具体应用，对于训练深度神经网络至关重要。
