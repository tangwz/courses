---
course: "advanced-jax"
chapter: "advanced-automatic-differentiation"
lesson: "custom-jvp-rules"
sourceId: 2883
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-4-advanced-automatic-differentiation/custom-jvp-rules"
title: "jax.custom_jvp 自定义求导规则"
description: "使用 `jax.custom_jvp` 定义自定义前向模式微分规则。"
order: 7
plots: []
sourceHash: "79f115a3225352f6c2cd75685bd5e0caeed2e3aae9b132c4a1fb2125f10b63e5"
sourceCorrections: []
---

虽然 `jax.grad` 和 `jax.vjp` 为反向模式自动微分提供支持，这对于训练大多数神经网络 (neural network)很重要，但在某些情况下，需要直接控制前向模式微分（雅可比向量 (vector)积）。前向模式计算输入变化（由“切线”向量表示）如何通过函数向前传播以影响输出。`jax.jvp` 函数直接计算此值。

有时，JAX 自动推导 JVP 规则的结果可能不是最佳的，甚至无法进行。这可能由以下几个原因引起：

- **数值稳定性：** 标准导数计算对于某些输入可能存在浮点数问题（例如，当输入接近于零时，对于 `sqrt(x)` 或 `log(x)` 等函数）。
- **性能：** 您可能知道一种计算 JVP 的更有效方法，优于 JAX 自动推导的方法。
- **不可微分点：** 您可能想在函数数学上不可微分的点定义特定的梯度行为（例如，在 `jnp.clip` 的边界处）。
- **外部代码：** 如果您的函数调用外部代码（例如通过 `jax.pure_callback`），JAX 无法看到其内部以计算导数。

在这些情况下，您可以使用 `jax.custom_jvp` 装饰器定义自己的 JVP 规则。这使您能够精确地告诉 JAX 如何计算函数的前向模式导数。

### 定义自定义 JVP

要定义自定义 JVP，您需要使用 `@jax.custom_jvp` 装饰您的函数，然后使用 `@[您的函数名].defjvp` 装饰器定义一个与该函数关联的单独的 JVP 规则函数。

让我们通过一个简单例子进行说明：实现一个数值稳定的 `jnp.sqrt` 版本，它处理 $x=0$ 处的导数。

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def stable_sqrt(x):
  """计算 sqrt(x)，在接近零时使用一个小的 epsilon 保证稳定性。"""
  is_zero = (x == 0.0)
  # 如果 x 恰好为零，避免在梯度计算中出现除以零的情况
  safe_x = jnp.where(is_zero, 1.0, x) 
  return jnp.sqrt(safe_x)

# 为 stable_sqrt 定义自定义 JVP 规则
@stable_sqrt.defjvp
def stable_sqrt_jvp(primals, tangents):
  """stable_sqrt 的自定义 JVP 规则。"""
  x, = primals
  t, = tangents # t 是切线向量 dx

  # 原始计算（前向传播结果）
  primal_out = stable_sqrt(x)

  # 切线计算（导数部分）
  # sqrt(x) 的导数是 0.5 * x**(-0.5)
  # 显式处理 x=0 情况，避免出现 NaN/inf
  is_zero = (x == 0.0)
  # 在 0 处使用一个大的有限数作为导数，或者使用 0， 
  # 取决于所需行为。这里，我们使用 0。
  safe_x = jnp.where(is_zero, 1.0, x) # 避免除以零
  deriv = 0.5 * jax.lax.rsqrt(safe_x) 
  tangent_out = jnp.where(is_zero, 0.0, deriv * t) # 在 x=0 处将导数设置为 0

  return primal_out, tangent_out

# 示例用法
x_val = jnp.array(0.0)
t_val = jnp.array(1.0) # 切线向量

# 使用自定义规则计算 JVP
primal_output, tangent_output = jax.jvp(stable_sqrt, (x_val,), (t_val,))

print(f"输入 x: {x_val}")
print(f"输入切线 t: {t_val}")
print(f"原始输出 (stable_sqrt(x)): {primal_output}")
print(f"切线输出 (x 处导数 * t): {tangent_output}") 

x_val_pos = jnp.array(4.0)
primal_output_pos, tangent_output_pos = jax.jvp(stable_sqrt, (x_val_pos,), (t_val,))

print(f"\n输入 x: {x_val_pos}")
print(f"输入切线 t: {t_val}")
print(f"原始输出 (stable_sqrt(x)): {primal_output_pos}") 
print(f"切线输出 (x 处导数 * t): {tangent_output_pos}") # 应为 0.5 * 4**(-0.5) * 1 = 0.25
```

### 理解 JVP 规则函数

使用 `.defjvp` 装饰的函数（此处为 `stable_sqrt_jvp`）接受两个参数 (parameter)：

1. `primals`：一个元组，包含原始函数 (`stable_sqrt`) 的原始输入。在我们的示例中，`primals` 是 `(x,)`。
2. `tangents`：一个元组，包含与原始输入对应的切线值。这些值表示微分的“方向”。在我们的示例中，`tangents` 是 `(t,)`，而 `t` 对应于 `x`。

JVP 规则函数必须返回一对值：

1. `primal_out`：计算原始函数的结果（前向传播）。通常通过调用原始函数本身来计算，如上所示：`primal_out = stable_sqrt(x)`。
2. `tangent_out`：雅可比向量 (vector)积的结果。这表示输出如何响应输入沿 `tangents` 指定的方向变化。在数学上，如果 $f$ 是函数，$x$ 是输入，它计算 $J(x) v$，其中 $J(x)$ 是 $f$ 在 $x$ 处的雅可比矩阵，$v$ 是输入切线向量。在示例中，这是 `jnp.where(is_zero, 0.0, deriv * t)`。

### 与 `jax.custom_vjp` 的关系

`jax.custom_jvp` 定义前向模式微分规则。其对应项 `jax.custom_vjp` 定义反向模式规则（向量 (vector)雅可比积），这是 `jax.grad` 的依据。

- 如果您只需要前向模式微分（例如，直接使用 `jax.jvp`，或通过 `jax.jacfwd` 计算完整雅可比矩阵），则仅定义 `jax.custom_jvp` 可能就足够了。
- 如果您需要反向模式微分（例如，使用 `jax.grad`），通常需要使用 `jax.custom_vjp` 定义自定义 VJP 规则。
- JAX 有时可以通过自动转置从已定义的 JVP 自动推导出 VJP（反之亦然）。然而，此过程可能并非总是高效或数值稳定的。为了获得最大控制和稳定性，尤其是在处理复杂函数或数值稳定性问题时，如果您的函数需要前向和反向模式能力，通常最佳做法是明确定义 **`jax.custom_jvp` 和 `jax.custom_vjp` 两者**。

### 用例回顾

定义自定义 JVP 规则在以下情况下特别有用：

1. **提升数值稳定性：** 手动指定问题点处的导数。
2. **优化性能：** 提供手动优化的 JVP 计算。
3. **处理不连续性：** 明确定义梯度行为（例如，将 `clip` 的导数设置为 0 或 1）。
4. **与非 JAX 代码接口：** 为 JAX 无法自动微分的函数提供导数信息。

与大多数 JAX 变换一样，具有自定义 JVP 规则的函数可以与 `jax.jit`、`jax.vmap` 和其他变换正确配合使用。JAX 将在追踪和编译期间使用您的自定义规则。

通过熟练掌握 `jax.custom_jvp`，您可以更好地控制 JAX 的前向模式自动微分，从而为专门的计算任务实现更高效的实现。这补充了 `jax.custom_vjp` 为反向模式微分提供的控制。

## 参考资料

- [\`jax.custom_jvp\`](https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_jvp.html) — JAX developers (2024)
  JAX 中定义自定义雅可比向量积（前向模式微分规则）的官方文档。
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://doi.org/10.1137/1.9780898717765) — Andreas Griewank and George F. Corliss (2008)
  Publisher: Society for Industrial and Applied Mathematics (SIAM); DOI: [10.1137/1.9780898717765](https://doi.org/10.1137/1.9780898717765)
  基础教科书，对自动微分（包括前向模式）进行了透彻的数学和算法处理。
- [Automatic Differentiation in Machine Learning: a Survey](http://jmlr.org/papers/v18/17-468.html) — Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind (2018)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 18; Pages: 1-43
  一篇研究综述，详细阐述了自动微分的原理和各种技术，特别是在机器学习应用背景下。
