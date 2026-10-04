# 介绍 `jax.grad`

来源：[原文](https://apxml.com/zh/courses/getting-started-with-jax/chapter-3-automatic-differentiation-grad/introducing-jax-grad)

[返回章节目录](README.md) · [返回课程目录](../README.md)

计算梯度对许多任务来说是基础，尤其是在优化机器学习 (machine learning)模型时。JAX 提供了一种强大而简洁的方法，通过函数转换来执行自动微分。它的主要工具是 `jax.grad`。

`jax.grad` 本身就是一个函数。它的主要作用是将一个返回标量值的数值型 Python 函数，转换成一个计算原函数梯度的新函数。可以这样理解：你给 `jax.grad` 一个函数 $f$，它会返回一个函数 $∇f$。

我们来看一个简单的例子。考虑数学函数 $f(x) = x^3$。我们从微积分中知道，它的导数（一维梯度）是 $f'(x) = 3x^2$。让我们看看如何使用 JAX 来计算它。

首先，我们使用标准 Python 语法定义函数 $f$，可能还会用到 `jax.numpy` 进行数值运算：

```python
import jax
import jax.numpy as jnp

def f(x):
  """计算 x 的三次方。"""
  return x**3

# 测试原函数
print(f"f(2.0) = {f(2.0)}")
```

```text
f(2.0) = 8.0
```

现在，我们将 `jax.grad` 转换应用于函数 `f`：

```python
# 使用 jax.grad 获得梯度函数
grad_f = jax.grad(f)

print(f"Type of f: {type(f)}")
print(f"Type of grad_f: {type(grad_f)}")
```

```text
Type of f: <class 'function'>
Type of grad_f: <class 'function'>
```

请注意，`jax.grad(f)` 返回一个新的 Python 函数，我们将其命名为 `grad_f`。这个新函数 `grad_f` 已准备好计算梯度。要在特定点（例如 $x=2.0$）获得梯度值，我们使用该值调用 `grad_f`：

```python
# 计算函数 f 在 x=2.0 处的梯度
gradient_at_2 = grad_f(2.0)

print(f"Gradient of f at x=2.0: {gradient_at_2}")
```

```text
Gradient of f at x=2.0: 12.0
```

结果是 $12.0$，这与我们手动计算的结果一致：$f'(x) = 3x^2$，所以 $f'(2.0) = 3 \times (2.0)^2 = 3 \times 4.0 = 12.0$。

记住输入和输出类型很重要：

1. 你提供给 `jax.grad` 一个 Python 函数，它接受一个或多个参数 (parameter)并返回一个**单一标量**值。
2. `jax.grad` 返回一个 *新* 的 Python 函数。
3. 这个 *新* 函数接受与原函数相同的参数，但返回原函数输出相对于其**第一个**位置参数的梯度，并在提供的参数值处进行评估。

如果函数接受多个参数会怎样？我们定义 $g(x, y) = x^2 \times y$：

```python
def g(x, y):
  """计算 x 的平方乘以 y。"""
  return (x**2) * y

# 获得函数 g 的梯度函数
grad_g = jax.grad(g)

# 计算在 (x=2.0, y=3.0) 处的梯度
# 默认情况下，这是相对于第一个参数 (x) 的梯度
gradient_g_wrt_x = grad_g(2.0, 3.0)

print(f"g(2.0, 3.0) = {g(2.0, 3.0)}")
print(f"Gradient of g w.r.t x at (2.0, 3.0): {gradient_g_wrt_x}")
```

```text
g(2.0, 3.0) = 12.0
Gradient of g w.r.t x at (2.0, 3.0): 12.0
```

这里，`grad_g(2.0, 3.0)` 计算的是偏导数 $\frac{\partial g}{\partial x}$ 在 $x=2.0, y=3.0$ 处的值。手动计算，$\frac{\partial g}{\partial x} = 2xy$。在 $(2.0, 3.0)$ 处评估得到 $2 \times 2.0 \times 3.0 = 12.0$，与 JAX 输出一致。我们将在后续章节中说明如何计算相对于 *其他* 参数的梯度。

`jax.grad` 的一个基本要求是，你进行微分的函数必须返回一个单一的标量数值（例如整数或浮点数，尽管浮点数通常用于微分）。如果你的函数返回数组、元组或任何非标量输出，直接使用 `jax.grad` 将会导致错误。这与梯度的数学定义相符，梯度适用于标量场（将向量 (vector)或数字映射到标量的函数）。存在对多输出函数进行微分的方法（例如 `jax.jacobian`），这些方法通常构建在 `jax.grad` 之上。

最后一点：自动微分通常对浮点数进行运算。尽管 JAX 有时可能允许整数输入，但通常会希望确保你进行微分的函数（以及梯度函数）的输入是浮点数，以获得有意义的导数结果。

```python
# 函数中使用 jax.numpy 的例子
def h(x):
  """计算 sin(x)。"""
  return jnp.sin(x)

grad_h = jax.grad(h)

# 计算在 pi/2 处的梯度（其中 cos(x) = 0）
# 使用 jnp.pi 保证精度并确保浮点数输入
gradient_h_at_pi_half = grad_h(jnp.pi / 2.0)

print(f"h(pi/2) = {h(jnp.pi / 2.0)}")
print(f"Gradient of h at x=pi/2: {gradient_h_at_pi_half}")
```

```text
h(pi/2) = 1.0
Gradient of h at x=pi/2: 0.0
```

正如预期，$\sin(x)$ 的导数是 $\cos(x)$，且 $\cos(\pi/2) = 0$。

总而言之，`jax.grad` 是 JAX 中进行自动微分的核心工具。它将一个返回标量值的 Python 函数转换为一个新函数，该新函数计算原函数相对于第一个参数的梯度。这种转换自动处理了微分的复杂性，使你能够专注于定义自己的计算。

## 参考资料

- [jax.grad](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) — JAX core team (2024)
  Publisher: JAX Project
  `jax.grad`函数用法、行为和要求的官方文档。
- [JAX: Composable Transformations of Python+NumPy Programs](https://proceedings.mlsys.org/paper_files/paper/2021/file/7ff82695843105777a7f41f7c3276b24-Paper.pdf) — James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Learner, Scott Manifold, Brendan M. Moore, J. H. Park, Dilip Ramakrishnan, Jonathan Sagynbekov, Sasha Rush, Marc van der Maaten, and Google Research (2021)
  Journal: Proceedings of Machine Learning Systems (MLSys 2021); Publisher: Conference on Machine Learning Systems; Volume: 3; Pages: 1-13; DOI: [10.48550/arXiv.1810.03361](https://doi.org/10.48550/arXiv.1810.03361)
  阐述了 JAX 的核心设计原则，包括其通过函数变换实现自动微分的方法。
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://epubs.siam.org/doi/book/10.1137/1.9780898717761) — Andreas Griewank, Andrea Walther (2008)
  Publisher: Society for Industrial and Applied Mathematics (SIAM); Pages: 459; DOI: [10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)
  一本关于自动微分方法理论基础和实际应用方面的全面参考书。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供了梯度、反向传播及其在机器学习优化中的重要性的基本概念。

---

[上一节](01-%E7%90%86%E8%A7%A3%E6%A2%AF%E5%BA%A6.md) · [下一节](03-%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%E7%9A%84%E5%B7%A5%E4%BD%9C%E6%96%B9%E5%BC%8F%EF%BC%9A%E5%8F%8D%E5%90%91%E6%A8%A1%E5%BC%8F.md)
