---
course: "advanced-jax"
chapter: "advanced-automatic-differentiation"
lesson: "computing-full-jacobians-hessians"
sourceId: 2877
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-4-advanced-automatic-differentiation/computing-full-jacobians-hessians"
title: "计算完整的雅可比矩阵和海森矩阵"
description: "使用 `vmap` 或 `jacfwd`/`jacrev` 高效计算完整雅可比矩阵或海森矩阵的策略。"
order: 5
plots: []
sourceHash: "88ba79c015c5c01dbd9ce46b16eaae0ff03de6db4f8a89717e720d1d493da797"
sourceCorrections: []
---

尽管雅可比向量 (vector)积（JVP）和向量雅可比积（VJP）在许多应用中计算效率高，尤其是在只需要乘积的基于梯度的优化中，但有时你需要完整的雅可比矩阵或海森矩阵。这对于某些二阶优化算法、敏感度分析或了解函数局部几何形状可能是必需的。JAX 提供了方便的函数来计算这些完整的矩阵。这些函数通常利用底层的 JVP 和 VJP 机制。然而，请注意，计算和存储这些完整的矩阵可能比计算乘积耗费更多的计算资源和内存，特别是对于机器学习 (machine learning)中常见的高维函数。

## 计算完整的雅可比矩阵

向量 (vector)值函数 $f: \mathbb{R}^n \to \mathbb{R}^m$ 的雅可比矩阵 $J$ 包含所有一阶偏导数。它的元素 $J_{ij}$ 表示第 $i$ 个输出分量 $f_i$ 对第 $j$ 个输入分量 $x_j$ 的偏导数：


$$
J_{ij} = \frac{\partial f_i}{\partial x_j}
$$


因此，完整的雅可比矩阵是一个 $m \times n$ 的矩阵。JAX 提供了两种主要方法来计算它，基于前向模式和反向模式自动微分。

### 使用前向模式（`jax.jacfwd`）

函数 `jax.jacfwd` 使用前向模式自动微分计算雅可比矩阵。它为对应于输入维度的每个标准基向量 (vector)计算 JVP。如果输入 $x \in \mathbb{R}^n$，它为每个基向量 $e_j$（其中 $e_j$ 在索引 $j$ 处为 1，其他地方为 0）计算 $J \cdot e_j$。结果 $J \cdot e_j$ 给出雅可比矩阵 $J$ 的第 $j$ 列。

```python
import jax
import jax.numpy as jnp

# 示例函数：R^3 -> R^2
def func(x):
  return jnp.array([x[0]**2 * x[1], jnp.sin(x[2])])

# 输入点
x_in = jnp.array([1.0, 2.0, jnp.pi / 2])

# 使用前向模式自动微分计算雅可比矩阵
jacobian_fwd = jax.jacfwd(func)(x_in)

print("输入:", x_in)
print("输出:", func(x_in))
print("雅可比矩阵 (jacfwd):\n", jacobian_fwd)
# 预期形状：(2, 3) -> (输出维度, 输入维度)
```

前向模式自动微分的计算成本通常与输入数量成正比。因此，当输入数量 ($n$) 小于输出数量 ($m$) 时，即对于“高”雅可比矩阵，`jax.jacfwd` 可能更高效。然而，由于它逐列计算雅可比矩阵，当函数本身在前向模式下多次求值成本较低时，其优势通常更明显。

### 使用反向模式（`jax.jacrev`）

另一种方法是，`jax.jacrev` 使用反向模式自动微分计算雅可比矩阵。这种方法使用 VJP。它为对应于*输出*维度的每个标准基向量 (vector)计算 VJP。如果输出 $f(x) \in \mathbb{R}^m$，它为每个基向量 $e_i$（其中 $e_i$ 在索引 $i$ 处为 1，其他地方为 0）计算 $e_i^T \cdot J$。结果 $e_i^T \cdot J$ 给出雅可比矩阵 $J$ 的第 $i$ 行。

```python
import jax
import jax.numpy as jnp

# 示例函数：R^3 -> R^2
def func(x):
  return jnp.array([x[0]**2 * x[1], jnp.sin(x[2])])

# 输入点
x_in = jnp.array([1.0, 2.0, jnp.pi / 2])

# 使用反向模式自动微分计算雅可比矩阵
jacobian_rev = jax.jacrev(func)(x_in)

print("输入:", x_in)
print("输出:", func(x_in))
print("雅可比矩阵 (jacrev):\n", jacobian_rev)
# 预期形状：(2, 3) -> (输出维度, 输入维度)
```

反向模式自动微分的计算成本通常与输出数量成正比。因此，当输出数量 ($m$) 小于输入数量 ($n$) 时，即对于“宽”雅可比矩阵，`jax.jacrev` 通常更高效。这是机器学习 (machine learning)中常见的情况，其中损失函数 (loss function)将高维参数 (parameter)映射到标量损失 ($m=1$)。计算梯度（`jax.grad`）是 `jax.jacrev` 在标量输出函数上的一个特例。

### 使用 `vmap` 手动计算（示例性说明）

你也可以通过将 `vmap` 应用于 `jvp` 或 `vjp`（或对于标量输出使用 `grad`）来手动构建雅可比矩阵。尽管 `jacfwd` 和 `jacrev` 通常因其优化实现而被优先使用，但了解 `vmap` 方法可以提供帮助。

对于函数 $f: \mathbb{R}^n \to \mathbb{R}^m$，将 `vmap` 应用于 `jvp` 并遍历标准基输入切线，可以得到雅可比矩阵的列：

```python
import jax
import jax.numpy as jnp

# 示例函数：R^3 -> R^2
def func(x):
  return jnp.array([x[0]**2 * x[1], jnp.sin(x[2])])

# 输入点
x_in = jnp.array([1.0, 2.0, jnp.pi / 2])
n = x_in.shape[0] # 输入维度

# 输入切线的标准基向量
basis_vectors_in = jnp.eye(n)

# 使用 vmap 在 jvp 上逐列计算雅可比矩阵
# jax.jvp 需要 primal_in 和 tangent_in
# 我们固定 primal_in 并映射 tangent_in
primals_out, jac_cols = jax.vmap(lambda tangent: jax.jvp(func, (x_in,), (tangent,)), \
                                 in_axes=0)(basis_vectors_in)

# 转置以得到标准的 m x n 雅可比矩阵
jacobian_vmap_jvp = jac_cols.T 

print("雅可比矩阵 (vmap + jvp):\n", jacobian_vmap_jvp)
```

类似地，将 `vmap` 应用于 `vjp` 并遍历标准基输出余切向量 (vector)，可以得到雅可比矩阵的行：

```python
import jax
import jax.numpy as jnp

# 示例函数：R^3 -> R^2
def func(x):
  return jnp.array([x[0]**2 * x[1], jnp.sin(x[2])])

# 输入点
x_in = jnp.array([1.0, 2.0, jnp.pi / 2])
primals_out, vjp_fun = jax.vjp(func, x_in)
m = primals_out.shape[0] # 输出维度

# 输出余切向量的标准基向量
basis_vectors_out = jnp.eye(m)

# 使用 vmap 在 vjp 上逐行计算雅可比矩阵
jac_rows = jax.vmap(vjp_fun, in_axes=0)(basis_vectors_out)

# 结果已经是 m x n 的雅可比矩阵（vmap 的每个输出都是一行）
# 注意：vjp_fun 返回一个元组，我们需要第一个元素
jacobian_vmap_vjp = jac_rows[0] 

print("雅可比矩阵 (vmap + vjp):\n", jacobian_vmap_vjp)
```

对于标量值函数 ($f: \mathbb{R}^n \to \mathbb{R}$)，雅可比矩阵就是梯度（一个行向量，或者其转置，即梯度向量）。你可以直接使用 `jax.grad` 计算它，或者通过 `jax.jacrev`（对于标量输出通常优先于 `jax.jacfwd`）来实现。

### `jacfwd` 和 `jacrev` 的选择

- 当函数输出维度 ($m$) 远小于输入维度 ($n$) 时，使用 `jax.jacrev`。这在机器学习 (machine learning)的损失函数 (loss function)中很常见 ($m=1$)。
- 当函数输入维度 ($n$) 远小于输出维度 ($m$) 时，使用 `jax.jacfwd`。
- 当 $n$ 和 $m$ 相近时，性能差异可能取决于计算的具体结构和底层的 XLA 实现。通过性能分析（例如 `%timeit` 或第 2 章中介绍的更高级工具）是针对特定用例确定最佳选择的方法。

## 计算完整的海森矩阵

标量值函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的海森矩阵 $H$ 包含所有二阶偏导数。它的元素 $H_{ij}$ 由以下式子给出：


$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$


海森矩阵是一个 $n \times n$ 的矩阵。对于具有连续二阶导数的函数（这在机器学习 (machine learning)环境中很常见），海森矩阵是对称的 ($H_{ij} = H_{ji}$)。

JAX 通过组合微分变换来计算海森矩阵。具体来说，海森矩阵是梯度函数的雅可比矩阵。

### 使用 `jax.hessian`

计算海森矩阵最直接的方法是使用 `jax.hessian`：

```python
import jax
import jax.numpy as jnp

# 示例标量函数：R^2 -> R
def scalar_func(x):
  # f(x, y) = x^2 * y + y^3
  return x[0]**2 * x[1] + x[1]**3

# 输入点
x_in = jnp.array([1.0, 2.0])

# 计算海森矩阵
hessian_matrix = jax.hessian(scalar_func)(x_in)

print("输入:", x_in)
print("输出:", scalar_func(x_in))
print("海森矩阵:\n", hessian_matrix)
# 预期形状：(2, 2) -> (输入维度, 输入维度)
```

在内部，`jax.hessian(f)` 通常实现为 `jax.jacfwd(jax.grad(f))`。它首先使用反向模式自动微分（`jax.grad`）计算梯度函数 ($g = \nabla f$)，然后使用前向模式自动微分（`jax.jacfwd`）计算此梯度函数的雅可比矩阵。你也可以将其计算为 `jax.jacrev(jax.grad(f))`。外层调用选择 `jacfwd` 还是 `jacrev`，其逻辑与雅可比矩阵相同：由于梯度函数 $g: \mathbb{R}^n \to \mathbb{R}^n$ 具有相同的输入和输出维度，因此在这里 `jacfwd` 可能会略微优先。

### 性能考量

计算完整的海森矩阵涉及计算 $O(n^2)$ 个二阶导数。随着输入维度 $n$ 的增加，这会很快变得计算成本过高。存储 $n \times n$ 矩阵也需要大量内存 ($O(n^2)$)。

对于许多应用，尤其是在优化中，不需要完整的海森矩阵。相反，算法通常依赖于海森向量 (vector)积（HvP），它计算给定向量 $v$ 的 $H \cdot v$。HvP 可以更高效地计算，而无需显式地形成 $H$，通常通过结合前向和反向模式自动微分来实现。例如，计算 $H \cdot v = \nabla_x ((\nabla_x f(x)) \cdot v)$ 的一种方法涉及一次前向模式遍历和一次反向模式遍历，成本大约相当于两次梯度计算。

```python
import jax
import jax.numpy as jnp

# 示例标量函数：R^2 -> R
def scalar_func(x):
  # f(x, y) = x^2 * y + y^3
  return x[0]**2 * x[1] + x[1]**3

# 输入点和向量
x_in = jnp.array([1.0, 2.0])
v = jnp.array([0.5, -0.5])

# 方法1：先计算完整海森矩阵再相乘（效率低）
hessian_matrix = jax.hessian(scalar_func)(x_in)
hvp_explicit = hessian_matrix @ v
print("完整海森矩阵:\n", hessian_matrix)
print("显式HvP:", hvp_explicit)

# 方法2：高效海森向量积
# 首先计算梯度函数
grad_f = jax.grad(scalar_func)
# 计算梯度函数的JVP
# jax.jvp(grad_f, (x_in,), (v,)) 返回 (grad_f(x_in), H @ v)
_, hvp_efficient = jax.jvp(grad_f, (x_in,), (v_)) 
print("高效HvP:", hvp_efficient) 
```

因此，虽然 JAX 提供了 `jax.hessian` 以方便使用，但请始终考虑海森向量积是否足以满足你的任务需求，因为它能为较大的 $n$ 带来显著的性能优势。

## 何时使用完整矩阵

计算完整雅可比矩阵和海森矩阵适用于以下情况：

1. **需要明确的矩阵结构：** 某些算法，如优化中的牛顿法，明确需要求解涉及海森矩阵的线性系统，或者需要分析雅可比矩阵的结构。
2. **维度较低：** 输入和输出维度 ($n, m$) 足够小，使得雅可比矩阵的 $O(n \times m)$ 成本或海森矩阵的 $O(n^2)$ 成本（计算和存储）可以接受。
3. **调试或分析：** 检查完整矩阵有助于理解函数行为或调试实现。

在许多大规模机器学习 (machine learning)场景中，直接计算和存储这些矩阵是不可行的。依赖于 JVP、VJP 和海森向量 (vector)积的技术是标准方法，使得微分计算能够有效扩展。了解 `jax.jacfwd`、`jax.jacrev` 和 `jax.hessian` 的工作原理，即使你在实践中主要使用梯度计算或向量积，也能对 JAX 的微分能力有很大帮助。

## 参考资料

- [JAX Documentation](https://jax.readthedocs.io/en/latest/) — JAX core developers (2024)
  JAX 自动微分功能的官方综合文档，包含计算雅可比矩阵和海森矩阵的函数。
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://epubs.siam.org/doi/book/10.1137/1.9780898717765) — Andreas Griewank, Andrea Walther (2008)
  Publisher: SIAM; DOI: [10.1137/1.9780898717765](https://doi.org/10.1137/1.9780898717765)
  一本涵盖自动微分理论和算法的学术著作，包括前向模式、反向模式、雅可比矩阵和海森矩阵。
- [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5) — Jorge Nocedal and Stephen J. Wright (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
  一本数值优化标准教材，解释了各种优化算法中一阶和二阶导数（雅可比矩阵、海森矩阵）的应用。
