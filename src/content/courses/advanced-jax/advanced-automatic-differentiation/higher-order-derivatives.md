---
course: "advanced-jax"
chapter: "advanced-automatic-differentiation"
lesson: "higher-order-derivatives"
sourceId: 2874
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-4-advanced-automatic-differentiation/higher-order-derivatives"
title: "高阶导数"
description: "通过组合 `grad`、`jvp` 和 `vjp`，计算二阶导数（海森矩阵）及更高阶导数。"
order: 4
plots: []
sourceHash: "fd16ccbcd7dfb59527e6836681ddcb61fa75ee5189cab5e65bca5bf765aba062"
sourceCorrections: []
---

JAX 的函数变换，包括 `grad`、`jvp` 和 `vjp`，被设计为可组合的。这意味着您可以将变换应用于本身就是其他变换结果的函数。这一强大功能使得计算高阶导数成为可能，为高级的分析和优化方法提供了途径。

### 组合求导函数

考虑高阶导数的最直接方式是重复应用求导函数。对于一个简单的标量函数 $f: \mathbb{R} \to \mathbb{R}$，计算其二阶导数就像应用 `jax.grad` 两次一样简单：

```python
import jax
import jax.numpy as jnp

def f(x):
  return x**3 + 2*x**2 - 3*x + 1

# 一阶导数
grad_f = jax.grad(f)

# 二阶导数
grad_grad_f = jax.grad(grad_f) # 或者等价地，jax.grad(jax.grad(f))

# 在 x = 2.0 处求值
x_val = 2.0
first_deriv = grad_f(x_val)
second_deriv = grad_grad_f(x_val)

print(f"f(x) = x^3 + 2x^2 - 3x + 1")
print(f"f'(x) = 3x^2 + 4x - 3")
print(f"f''(x) = 6x + 4")
print(f"f'({x_val}) = {first_deriv}")   # 预期结果: 3*(2^2) + 4*2 - 3 = 12 + 8 - 3 = 17
print(f"f''({x_val}) = {second_deriv}")  # 预期结果: 6*2 + 4 = 12 + 4 = 16
```

这种组合方式与您在微积分中预期的一致。`grad(f)` 返回一个计算一阶导数的函数，对*那个*函数应用 `grad` 会得到一个计算二阶导数的函数。这个原理可以很容易地扩展到标量函数的三阶甚至更高阶导数。

### 海森向量 (vector)积 (HVPs)

对于多元函数 $f: \mathbb{R}^n \to \mathbb{R}$，简单地将 `grad` 组合两次并不能直接得到海森矩阵（二阶偏导数矩阵）。相反，`grad(grad(f))` 将计算梯度函数的梯度，这并非完全是海森结构。

在优化算法中（例如牛顿法或截断牛顿法），我们通常不需要完整的海森矩阵 $H$。相反，我们需要计算海森矩阵与向量 $v$ 的乘积，这被称为海森向量积 (HVP)：$Hv$。计算 HVP 通常比构建完整的海森矩阵效率高很多，尤其是在 $n$ 较大时。

JAX 允许通过组合正向模式 (`jvp`) 和反向模式 (`vjp` 或 `grad`) 自动微分来高效地计算 HVP。回忆一下，`jax.grad(f)` 本质上是基于 `jax.vjp` 构建的。

计算 $H v$ 有两种主要方法：

1. **正向-反向叠加：** 计算梯度函数的 JVP。
   令 $g(x) = \nabla f(x)$。我们希望计算 $g$ 与切向量 $v$ 的 JVP。

   
   $$
   \text{JVP}(g, x, v) = \frac{\partial g(x)}{\partial x} v = \frac{\partial (\nabla f(x))}{\partial x} v = H(x) v
   $$
   

   在 JAX 中，这表示为：

   ```python
   def hvp_forward_over_reverse(f, primals, tangents):
     # 使用 jvp(grad(f)) 计算 H @ v
     return jax.jvp(jax.grad(f), primals, tangents)[1]
   ```

   `[1]` 选择了 `jvp` 结果的输出部分，对应于乘积项。
2. **反向-正向叠加：** 计算应用于固定向量 $v$ 的 JVP 函数的 VJP。这种方式稍微不那么直接。
   考虑函数 $h(x) = (\nabla f(x))^\top v = \text{jnp.dot}(\nabla f(x), v)$。$h(x)$ 的梯度是 $\nabla h(x) = H(x) v$。
   在 JAX 中，这表示为：

   ```python
   def hvp_reverse_over_reverse(f, primals, tangents):
       # 使用 grad(lambda x: jnp.dot(grad(f)(x), v)) 计算 H @ v
       x, = primals
       v, = tangents
       # 需要 lambda 来捕获 v
       return jax.grad(lambda x_lambda: jnp.dot(jax.grad(f)(x_lambda), v))(x)
   ```

   请注意，这里使用了 `lambda` 函数来正确捕获梯度计算中的向量 `v`。

通常，对于 HVP 而言，**正向-反向叠加**方法 (`jvp(grad(f), ...)`) 更受推荐，因为它在计算上通常更高效，并且与数学定义更直接对应。

我们来看一个例子：

```python
import jax
import jax.numpy as jnp

def func(x):
  # f(x,y) = x^2 * y + y^3
  return x[0]**2 * x[1] + x[1]**3

x_primal = jnp.array([1.0, 2.0])
v_tangent = jnp.array([1.0, 0.0]) # 与海森矩阵相乘的向量

# 使用 jvp(grad(f)) - 正向-反向叠加
hvp_val = jax.jvp(jax.grad(func), (x_primal,), (v_tangent,))[1]

print(f"函数: f(x,y) = x^2 * y + y^3")
# 梯度: nabla f = [2xy, x^2 + 3y^2]
# 海森矩阵: H = [[2y, 2x], [2x, 6y]]
# 在 (1, 2) 处: H = [[4, 2], [2, 12]]
# H @ v = [[4, 2], [2, 12]] @ [1, 0] = [4, 2]

print(f"原变量 (x): {x_primal}")
print(f"切向量 (v): {v_tangent}")
print(f"海森向量积 (H @ v): {hvp_val}") # 预期结果: [4., 2.]
```

### 计算完整的海森矩阵

虽然 HVP 对于许多应用而言是高效的，但有时需要完整的海森矩阵 $H$ 的显式表示。海森矩阵是梯度函数的雅可比矩阵：$H(x) = J_{\nabla f}(x)$。

我们可以运用 JAX 计算雅可比矩阵的函数 (`jacfwd` 和 `jacrev`)，将其应用于梯度函数 (`jax.grad(f)`) 来获得海森矩阵。

1. **使用 `jax.hessian`：** JAX 提供了一个便捷函数 `jax.hessian`，可以直接计算海森矩阵。

   ```python
   hessian_matrix = jax.hessian(func)(x_primal)
   print("\n使用 jax.hessian 获得的完整海森矩阵:")
   print(hessian_matrix)
   # 预期结果: [[4., 2.], [2., 12.]]
   ```

   在内部，`jax.hessian` 通常会结合正向和反向模式自动微分（例如 `jacfwd(jacrev(f))` 或 `jacrev(jacfwd(f))`）以提高效率，这类似于 `jacfwd` 和 `jacrev` 计算雅可比矩阵的方式。
2. **使用 `jacfwd(grad(f))`：** 这通过将正向模式自动微分 (`jacfwd`) 应用于梯度函数来计算海森矩阵。它通过 JVP 计算海森矩阵的每一列。

   ```python
   hessian_jacfwd = jax.jacfwd(jax.grad(func))(x_primal)
   print("\n使用 jacfwd(grad(f)) 获得的完整海森矩阵:")
   print(hessian_jacfwd)
   ```
3. **使用 `jacrev(grad(f))`：** 这通过将反向模式自动微分 (`jacrev`) 应用于梯度函数来计算海森矩阵。它通过 VJP 计算海森矩阵的每一行。

   ```python
   hessian_jacrev = jax.jacrev(jax.grad(func))(x_primal)
   print("\n使用 jacrev(grad(f)) 获得的完整海森矩阵:")
   print(hessian_jacrev)
   ```

`jacfwd(grad(f))` 和 `jacrev(grad(f))` 之间的选择可能对性能有影响，这取决于涉及的相对维度，类似于一阶雅可比矩阵在 `jacfwd` 和 `jacrev` 之间的选择。`jax.hessian` 会尝试自动选择一个合理的策略。

### 二阶

组合原理自然地得到扩展。您可以通过根据需要进一步组合 `grad`、`jvp`、`jacfwd` 和 `jacrev` 来计算三阶导数、四阶导数以及各种混合偏导数。例如，可以计算海森矩阵的梯度，尽管在标准机器学习 (machine learning)中，直接应用比一阶和二阶导数不那么常见。

例子：初始标量函数 `f` 的三阶导数：

```python
# 三阶导数
grad_grad_grad_f = jax.grad(grad_grad_f) # 或者 jax.grad(jax.grad(jax.grad(f)))

third_deriv = grad_grad_grad_f(x_val)
print(f"\nf'''(x) = 6")
print(f"f'''({x_val}) = {third_deriv}") # 预期结果: 6.0
```

随意组合求导函数的能力是 JAX 自动微分系统的标志性特点，为实现高级数值和机器学习算法提供了显著的灵活性。

## 参考资料

- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://epubs.siam.org/doi/book/10.1137/1.9780898717721) — Andreas Griewank, Andrea Walther (2008)
  Publisher: SIAM; DOI: [10.1137/1.9780898717721](https://doi.org/10.1137/1.9780898717721)
  自动微分的理论基础书籍，涵盖了与 Hessian-向量积等高阶导数相关的正向、反向和混合模式 AD 的理论基础。
- [Numerical Optimization](https://www.springer.com/gp/book/9780387303031) — Jorge Nocedal and Stephen J. Wright (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
  优化算法的经典教材，详细解释了牛顿法等利用二阶导数和 Hessian-向量积的方法。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面介绍了自动微分及其在深度学习中的应用，包括对二阶优化技术的讨论。
