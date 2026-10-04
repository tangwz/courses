# 高阶导数（`grad`的`grad`）

来源：[原文](https://apxml.com/zh/courses/getting-started-with-jax/chapter-3-automatic-differentiation-grad/grad-higher-order-derivatives)

[返回章节目录](README.md) · [返回课程目录](../README.md)

`jax.grad` 函数接受一个计算标量值的 Python 函数 $f$，并返回一个计算梯度 $\nabla f$ 的新 Python 函数。JAX 设计的一个重要特点是，它的变换（包括 `jax.grad`）本身就是用 Python 实现的，并且作用于 Python 函数。这意味着我们可以对其他变换的*输出*施加变换。

如果我们对 `jax.grad` 产生的梯度函数再次应用 `jax.grad`，会发生什么？我们将得到一个计算二阶导数的函数。

让我们考虑一个简单的标量函数：


$$
f(x) = x^3
$$


它的第一导数是 $f'(x) = 3x^2$，它的第二导数是 $f''(x) = 6x$。

我们可以使用 `jax.grad` 计算第一导数函数：

```python
import jax
import jax.numpy as jnp

def f(x):
  return x**3

# 获取计算第一导数的函数
grad_f = jax.grad(f)

# 在 x=2.0 处计算原函数及其第一导数
x_val = 2.0
print(f"f({x_val}) =", f(x_val))
print(f"f'({x_val}) =", grad_f(x_val))
```

运行结果如下：

```
f(2.0) = 8.0
f'(2.0) = 12.0
```

这与我们的解析计算结果一致：$f(2) = 2^3 = 8$ 和 $f'(2) = 3(2^2) = 12$。

现在，由于 `grad_f` 只是另一个 Python 函数（碰巧计算梯度），我们也可以对它求导：

```python
# 获取计算二阶导数的函数
grad_grad_f = jax.grad(grad_f)

# 在 x=2.0 处计算二阶导数
print(f"f''({x_val}) =", grad_grad_f(x_val))
```

输出结果是：

```
f''(2.0) = 12.0
```

这也与我们的解析结果一致：$f''(2) = 6(2) = 12$。

我们可以通过嵌套调用 `jax.grad` 来更简洁地表达：

```python
# 直接定义二阶导数函数
grad_grad_f_nested = jax.grad(jax.grad(f))

print(f"f''({x_val}) using nested grad =", grad_grad_f_nested(x_val))
```

```
f''(2.0) using nested grad = 12.0
```

这个过程可以继续用于计算三阶、四阶甚至更高阶导数，仅受计算成本和数值稳定性的限制。

### 多元函数的高阶导数

这种嵌套同样适用于具有多个参数 (parameter)的函数。在处理多元标量函数时，两次应用 `grad` 可以用来计算 Hessian 矩阵的元素。标量函数 $f(x_1, x_2, ..., x_n)$ 的 Hessian 矩阵 $H$ 是二阶偏导数的方阵：


$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$


例如，考虑 $f(x, y) = x^2 y + y^3$。让我们找出 $\frac{\partial^2 f}{\partial x^2}$。我们首先对 $x$ 求导（将 $y$ 视为常数），然后再次对结果对 $x$ 求导。

```python
def f_multi(x, y):
  return x**2 * y + y**3

# 对 x 的一阶导数 (参数 0)
grad_f_wrt_x = jax.grad(f_multi, argnums=0)

# 二阶导数：对 grad_f_wrt_x 再次对 x 求导 (它的第一个参数，索引 0)
grad_grad_f_wrt_xx = jax.grad(grad_f_wrt_x, argnums=0)

# 解析计算：
# df/dx = 2xy
# d^2f/dx^2 = 2y

x_val, y_val = 2.0, 3.0
print(f"d^2f/dx^2 at ({x_val}, {y_val}) =", grad_grad_f_wrt_xx(x_val, y_val))
print(f"Analytical result (2*y):", 2 * y_val)
```

```
d^2f/dx^2 at (2.0, 3.0) = 6.0
Analytical result (2*y): 6.0
```

为了计算混合偏导数，例如 $\frac{\partial^2 f}{\partial y \partial x}$，我们在第二次求导时改变 `argnums` 参数：

```python
# 二阶导数：对 grad_f_wrt_x 再次对 y 求导 (它的第二个参数，索引 1)
grad_grad_f_wrt_xy = jax.grad(grad_f_wrt_x, argnums=1)

# 解析计算：
# df/dx = 2xy
# d^2f/dydx = 2x

print(f"d^2f/dydx at ({x_val}, {y_val}) =", grad_grad_f_wrt_xy(x_val, y_val))
print(f"Analytical result (2*x):", 2 * x_val)
```

```
d^2f/dydx at (2.0, 3.0) = 4.0
Analytical result (2*x): 4.0
```

尽管嵌套 `grad` 对于计算特定的二阶导数甚至更高阶导数非常有效，JAX 也提供了便捷函数，例如 `jax.hessian`，用于直接计算完整的 Hessian 矩阵，如果你需要所有二阶偏导数，这可能会更有效率。但是，理解 `grad(grad(...))` 的组合方式对于理解 JAX 变换的可组合性是基础的。

这种随意组合 `grad` 的能力表明了 JAX 的函数式特性。每个变换都接收一个函数并返回一个新函数，准备好被使用或进一步变换。

## 参考资料

- [JAX: Autodiff Cookbook (Higher-Order Derivatives)](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#Higher-order-derivatives) — JAX developers (2024)
  JAX 的官方文档，详细说明了如何使用 `jax.grad` 进行高阶微分，并提供了实际代码示例。
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://epubs.siam.org/doi/book/10.1137/1.9780898717765) — Andreas Griewank, Andrea Walther (2008)
  Publisher: Society for Industrial and Applied Mathematics (SIAM); DOI: [10.1137/1.9780898717765](https://doi.org/10.1137/1.9780898717765)
  一本基础的学术著作，涵盖了自动微分的理论方面和算法，包括计算高阶导数的方法。
- [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5) — Jorge Nocedal and Stephen J. Wright (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
  一本标准教科书，讨论了优化算法中一阶和二阶导数（如Hessian矩阵）的作用。

---

[上一节](04-%E5%85%B3%E4%BA%8E%E5%8F%82%E6%95%B0%E6%B1%82%E5%AF%BC.md) · [下一节](06-%E5%80%BC%E5%92%8C%E6%A2%AF%E5%BA%A6%20%28%60jax.value_and_grad%60%29.md)
