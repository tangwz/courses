---
course: "getting-started-with-jax"
chapter: "accelerating-functions-jit"
lesson: "jit-control-flow"
sourceId: 1703
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-2-accelerating-functions-jit/jit-control-flow"
title: "Python 控制流与 `jit`"
description: "在 JIT 编译的函数中处理 Python 的原生控制流（if/else、循环）。"
order: 4
plots: []
sourceHash: "e0e1bf25b5402518b5ceff990601d784cb7f21fd31f01039339482266117b084"
sourceCorrections: []
---

`jax.jit` 通过使用抽象值（代表潜在输入的形状和类型）追踪代码执行，然后编译生成的运算序列（Jaxpr）来加速您的代码。对于给定的函数签名，此追踪过程通常只发生一次。

然而，标准的 Python 控制流语句，如 `if`/`else` 和 `for`/`while` 循环，对此追踪机制提出了挑战。Python 通常根据执行期间变量的*具体值*来评估这些语句。但在 JAX 追踪期间，值通常是抽象的 `Tracer` 对象，而不是具体的数字。这种根本差异导致在尝试 `jit` 编译包含依赖追踪值的原生 Python 控制流的函数时出现问题。

### 条件执行：Python `if` 与 `jax.lax.cond`

考虑一个带有 `if` 语句的简单 Python 函数：

```python
import jax
import jax.numpy as jnp

def conditional_func_py(x):
  if x > 0:
    return x * 2
  else:
    return x / 2

# 尝试使用普通数字运行（工作正常）
print(conditional_func_py(5.0))   # 输出: 10.0
print(conditional_func_py(-4.0))  # 输出: -2.0

# 现在，让我们尝试 JIT 编译它
jitted_conditional_py = jax.jit(conditional_func_py)

# 这将引发错误！
# jitted_conditional_py(jnp.array(5.0))
```

如果您取消注释并运行最后一行，JAX 将引发 `ConcretizationTypeError`。为什么？当 `jax.jit` 追踪 `conditional_func_py` 时，它会为 `x` 使用一个抽象追踪器对象。表达式 `x > 0` 也产生一个代表布尔值的抽象追踪器，而不是具体的 `True` 或 `False`。标准的 Python `if` 语句不知道如何处理这种抽象布尔值；它需要一个具体值才能在*追踪时*决定执行哪个分支。由于分支取决于 `x` 的运行时值，JAX 无法确定要编译的单一执行路径。

要在条件依赖追踪值（如 JAX 数组）的 `jit` 编译函数中处理条件逻辑，您需要使用 JAX 特定的控制流原语。对于 `if`/`else` 逻辑，主要工具是 `jax.lax.cond`。

`jax.lax.cond` 接受以下参数 (parameter)：

1. `pred`: 一个布尔谓词（可以是追踪值）。
2. `true_fun`: 如果 `pred` 为真则执行的函数。
3. `false_fun`: 如果 `pred` 为假则执行的函数。
4. `operand`: 要传递给 `true_fun` 或 `false_fun` 的输入。两个函数必须接受相同类型/结构的输入。

以下是使用 `jax.lax.cond` 重写我们示例的方法：

```python
import jax
import jax.numpy as jnp
from jax import lax

def conditional_func_lax(x):
  # 定义真分支和假分支的函数
  def true_branch(val):
    return val * 2
  def false_branch(val):
    return val / 2

  # 使用 lax.cond 根据 x > 0 来选择应用哪个函数
  return lax.cond(x > 0,  # 条件
                  true_branch, # 如果为真时执行的函数
                  false_branch, # 如果为假时执行的函数
                  x)         # 传递给所选函数的输入

# JIT 编译 lax 版本
jitted_conditional_lax = jax.jit(conditional_func_lax)

# 现在这可以工作了！
result_pos = jitted_conditional_lax(jnp.array(5.0))
result_neg = jitted_conditional_lax(jnp.array(-4.0))

print(result_pos) # 输出: 10.0
print(result_neg) # 输出: -2.0
```

`lax.cond` 指示 JAX 编译器在编译代码中包含*两种*可能的执行路径，并根据谓词 `x > 0` 的实际值在运行时选择合适的路径。这样，追踪成功，并且编译后的代码可以正确处理条件逻辑。

**注意：** 如果 Python `if` 语句中的条件*仅*依赖于静态值（在编译时已知，而不是追踪的 JAX 数组），`jit` *可能*成功编译它。然而，如果静态值在不同调用之间发生变化，这通常会导致重新编译，从而可能抵消性能提升。对于涉及 JAX 数组的条件语句，通常更推荐使用 `lax.cond`。

### 循环：Python `for`/`while` 与 `jax.lax` 原语

标准的 Python `for` 和 `while` 循环也会出现类似问题。

**循环展开：** 如果 Python `for` 循环的迭代次数是静态的（常量或由静态参数 (parameter)确定），`jax.jit` 通常会通过*展开*循环来追踪它。这意味着追踪会显式记录每次迭代的操作。

```python
import jax
import jax.numpy as jnp

def sum_first_n_py(arr, n): # n 在此处是静态的
  total = 0.0
  for i in range(n): # 带有静态范围的 Python 循环
    total += arr[i]
  return total

jitted_sum_first_3 = jax.jit(sum_first_n_py, static_argnums=1) # 告诉 JIT n 是静态的

my_array = jnp.arange(10.0)
print(jitted_sum_first_3(my_array, 3)) # 输出: 3.0 (0.0 + 1.0 + 2.0)

# 追踪实际上变为：
# total = 0.0
# total += arr[0]
# total += arr[1]
# total += arr[2]
# return total
```

虽然展开适用于静态迭代次数，但如果迭代次数很高，它可能导致非常大的计算图和长的编译时间。更重要的是，如果迭代次数（如上面的 `n`）或循环的继续条件依赖于*追踪值*（函数内部计算的 JAX 数组），Python 的 `for` 或 `while` 循环在追踪期间将再次导致 `ConcretizationTypeError`。

JAX 在 `jax.lax` 中提供了结构化的循环原语来处理这些情况：

- `jax.lax.fori_loop(lower_bound, upper_bound, body_fun, init_val)`: 用于循环次数（`upper_bound - lower_bound`）在循环开始前已知（可能作为追踪值）的情况。`body_fun` 接受循环索引 `i` 和当前循环状态（`val`），并返回下一次迭代的更新状态。
- `jax.lax.while_loop(cond_fun, body_fun, init_val)`: 用于继续条件在每一步都评估的循环。`cond_fun` 接受当前状态并返回一个布尔值（追踪值）。`body_fun` 接受当前状态并返回更新后的状态。
- `jax.lax.scan(f, init, xs)`: 一个功能强大的原语，常用于递归计算或在序列中传递状态。它迭代地应用函数 `f`，累积结果。我们将在讨论状态管理时更详细地介绍 `scan`。

让我们使用 `jax.lax.fori_loop` 重写求和示例，允许 `n` 成为一个追踪值：

```python
import jax
import jax.numpy as jnp
from jax import lax

def sum_first_n_lax(arr, n):
  # body_fun 接受循环索引 i 和当前循环携带值（total）
  # 它返回下一次迭代的更新携带值
  def body_fun(i, current_total):
    return current_total + arr[i]

  # 运行循环从 0 到（但不包括）n
  # total 的初始值为 0.0
  initial_val = 0.0
  final_total = lax.fori_loop(0, n, body_fun, initial_val)
  return final_total

# JIT 编译，无需为 n 使用 static_argnums
jitted_sum_lax = jax.jit(sum_first_n_lax)

my_array = jnp.arange(10.0)
n_val = jnp.array(3) # n 现在可以是 JAX 数组

print(jitted_sum_lax(my_array, n_val)) # 输出: 3.0
print(jitted_sum_lax(my_array, jnp.array(5))) # 输出: 10.0 (0+1+2+3+4)
```

`jax.lax.fori_loop` 允许 JAX 编译循环本身的表示，而不是展开它，这使得它即使在迭代次数 `n` 是根据追踪输入动态确定的情况下也适用。

> `jit` 中展开的 Python 循环（左）与 JAX `lax` 循环（右）的区别。Python 展开要求在追踪时迭代次数是固定且已知的。JAX 原语编译一个通用的循环结构，即使迭代计数依赖于追踪值也适用。

### 总结

使用 `jax.jit` 时，请注意 Python 控制流：

- **问题：** 原生的 Python `if`、`for` 和 `while` 语句依赖具体值来确定执行路径。JAX 追踪通常操作抽象的 `Tracer` 值。在这些 Python 构造中依赖追踪值通常会导致 `ConcretizationTypeError`。
- **解决方法：** 当分支或循环依赖于追踪的 JAX 数组值时，请使用 JAX 的 `lax` 控制流原语：
  - `lax.cond` 用于条件执行（`if`/`else`）。
  - `lax.fori_loop` 用于固定迭代次数的循环（迭代次数可以是动态的）。
  - `lax.while_loop` 用于基于条件的循环。
  - `lax.scan` 用于有状态的序列计算。
- **何时可以使用 Python 控制流？** 如果条件或循环迭代*仅*依赖于静态值（常量，或使用 `static_argnums`/`static_argnames` 标记 (token)为静态的参数 (parameter)），Python 控制流*可以*工作。JIT 会有效地为遇到的每组不同的静态值专门化并编译一个版本的函数。然而，这可能导致频繁的重新编译。对于涉及 JAX 数组的逻辑，`lax` 原语是标准且通常更受推荐的方法。

了解这种区别对于编写高效且可组合的 JAX 代码非常重要，它能充分利用 `jit` 编译的全部能力。

## 参考资料

- [JIT compilation and Control Flow Primitives](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-compilation) — The JAX Team (2024)
  JAX官方指南，解释JIT编译的工作原理，包括使用抽象值进行追踪，以及在依赖追踪值的控制流中使用`jax.lax`原语（如`cond`、`fori_loop`、`while_loop`、`scan`）的必要性。
