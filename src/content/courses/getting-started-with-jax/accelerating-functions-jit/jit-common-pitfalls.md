---
course: "getting-started-with-jax"
chapter: "accelerating-functions-jit"
lesson: "jit-common-pitfalls"
sourceId: 1710
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-2-accelerating-functions-jit/jit-common-pitfalls"
title: "`jit` 的常见问题"
description: "识别并避免使用 `jax.jit` 时的常见问题。"
order: 6
plots: []
sourceHash: "c7603ad493bbbca9bc670d0f2f0ba5d1dbdcf135e7eb893954f7e7d77cbb942c"
sourceCorrections: []
---

`jax.jit` 通过编译Python函数，能带来显著的性能提升，但它对追踪的依赖也引入了特定的限制。了解这些限制可以帮助您避免一些常见问题，并有效使用`jit`。JAX会使用抽象值（追踪器）来检查您的函数，这些抽象值代表潜在输入的*形状*和*数据类型*，而非其具体值。这种抽象执行方式会导致一些潜在的问题。

### 依赖追踪值的条件语句

标准Python `if` 语句需要一个具体的 `True` 或 `False` 值来决定执行哪个分支。然而，在追踪期间，JAX通常处理的是抽象追踪器。如果一个条件直接依赖于追踪数组的*值*，Python无法将其解析为一个单一的布尔值。

考虑这个函数：

```python
import jax
import jax.numpy as jnp

def condition_on_value(x):
  if x > 0: # Python 'if' 检查具体值
    return x * 2
  else:
    return x / 2

jitted_condition = jax.jit(condition_on_value)

# 这在追踪期间可能会引发错误
try:
  print(jitted_condition(jnp.array(5.0)))
except Exception as e:
  print(f"Error: {e}")

# 错误输出示例：
# Error: ConcretizationTypeError: Abstract tracer value encountered...
# 问题出现在评估条件(x > 0)时
```

在追踪期间，`x` 是一个抽象的 `Traced<ShapedArray(float32[])>` 对象。Python不知道如何在函数编译*之前*将 `Traced<...> > 0` 解释为具体的 `True` 或 `False` 来选择路径。JAX需要在编译时知道完整的计算图。

**解决方法：** 使用JAX的结构化控制流原语，例如 `jax.lax.cond`。这些原语会追踪条件语句的*两个*分支，并在加速器上运行时选择合适的结果，从而使编译能够进行。

```python
import jax
import jax.numpy as jnp
from jax import lax

def condition_on_value_lax(x):
  # 定义真分支和假分支的函数
  def true_fun(operand):
    return operand * 2
  def false_fun(operand):
    return operand / 2
  # 使用 lax.cond: cond(谓词, 真函数, 假函数, 操作数)
  return lax.cond(x > 0, true_fun, false_fun, x)

jitted_condition_lax = jax.jit(condition_on_value_lax)

# 这可以正常运行
print(jitted_condition_lax(jnp.array(5.0)))   # 输出: 10.0
print(jitted_condition_lax(jnp.array(-4.0)))  # 输出: -2.0
```

请注意，基于*静态*值（常量或标记 (token)为静态的参数 (parameter)）的条件语句可以很好地与标准Python `if` 语句一起使用，因为它们的值在追踪时是已知的。

### 变长迭代的循环

与条件语句类似，标准Python `for` 或 `while` 循环通常根据运行时值来确定它们的迭代次数。如果循环的持续时间依赖于追踪值，JAX在追踪期间无法“展开”循环，因为迭代次数是未知的。

```python
import jax
import jax.numpy as jnp

def variable_loop(x, n):
  # Python 'for' 循环范围依赖于 n
  total = x
  for i in range(n): # 'n' 可能是一个追踪值
    total = total + i
  return total

jitted_loop = jax.jit(variable_loop)

# 如果 'n' 被追踪，这将导致错误
try:
  # 假设 'n' 是从某个追踪计算派生出来的
  traced_n = jnp.array(3) # 在实际情况中，这可能来自另一个 JAX 操作
  print(jitted_loop(jnp.array(10.0), traced_n))
except Exception as e:
  print(f"Error: {e}")

# 错误输出示例：
# Error: ConcretizationTypeError: Abstract tracer value encountered...
# 问题出现在评估循环的 range(n) 时。
```

**解决方法：** 使用JAX的结构化循环原语，如 `jax.lax.fori_loop`（适用于在追踪时已知固定迭代次数的情况）或 `jax.lax.scan`（适用于循环携带状态的递归计算）。

```python
import jax
import jax.numpy as jnp
from jax import lax

def fixed_loop_fori(x, n_static):
  # n_static 必须在编译时已知
  def body_fun(i, current_total):
    return current_total + i
  # 使用 lax.fori_loop: fori_loop(下限, 上限, 循环体函数, 初始值)
  # 注意：上限 'n_static' 必须是静态的（编译时常量）
  return lax.fori_loop(0, n_static, body_fun, x)

# 我们需要告诉 jit 'n_static' 是一个编译时常量
jitted_loop_fori = jax.jit(fixed_loop_fori, static_argnums=(1,))

# 这之所以有效，是因为 'n_static' (3) 被视为静态值
print(jitted_loop_fori(jnp.array(10.0), 3)) # 输出: 13.0 (10 + 0 + 1 + 2)

# 使用 lax.scan 来传递状态（示例：累加和）
def cumulative_sum_scan(xs):
    def scan_op(carry, x):
        new_carry = carry + x
        return new_carry, new_carry # (下一步的传递值, 这一步的输出值)

    _, ys = lax.scan(scan_op, 0.0, xs) # 初始传递值为 0.0
    return ys

jitted_scan = jax.jit(cumulative_sum_scan)
data = jnp.array([1.0, 2.0, 3.0, 4.0])
print(jitted_scan(data)) # 输出: [1. 3. 6. 10.]
```

Python 循环，如果其固定迭代次数由静态值确定，通常与 `jit` 兼容。

### 副作用

JAX 转换（如 `jit`）假定函数是*纯粹的*。纯函数的输出仅取决于其输入，并且没有副作用（例如打印、修改全局变量或写入文件）。

`jit` 会为给定的输入签名追踪函数一次并进行编译。编译后的代码随后会用于具有匹配签名的后续调用。Python函数中的任何副作用只会在*追踪*阶段发生，而不会在编译代码执行时发生。

```python
import jax
import time

@jax.jit
def function_with_side_effect(x):
  print(f"TRACE: Running Python code for x={x}") # 副作用：打印
  # 模拟一些计算
  time.sleep(0.1)
  return x * x

print("首次调用（触发追踪和编译）：")
result1 = function_with_side_effect(jnp.array(3.0))
print(f"Result 1: {result1}\n")

print("第二次调用（使用缓存的编译代码）：")
result2 = function_with_side_effect(jnp.array(4.0)) # 相同的形状/数据类型
print(f"Result 2: {result2}\n")

print("第三次调用（不同输入，但形状/数据类型兼容）：")
result3 = function_with_side_effect(jnp.array(5.0)) # 相同的形状/数据类型
print(f"Result 3: {result3}")

# 输出示例：
# 首次调用（触发追踪和编译）：
# TRACE: Running Python code for x=Traced<ShapedArray(float32[])>with<DynamicJaxprTrace(level=0/1)>
# Result 1: 9.0
#
# 第二次调用（使用缓存的编译代码）：
# Result 2: 16.0
#
# 第三次调用（不同输入，但形状/数据类型兼容）：
# Result 3: 25.0
```

请注意，`TRACE:` 消息只在首次调用函数被追踪和编译时出现一次。后续调用会直接执行优化后的编译代码，跳过Python的 `print` 语句。

**解决方法：** 在您打算 `jit` 的函数中避免副作用。如果您需要管理状态（例如模型参数 (parameter)），请使用显式状态传递模式，我们将在第6章中讲解。对于调试，可以考虑JAX的特定调试工具或暂时禁用 `jit`。

### 动态形状和数据类型

`jit` 会针对特定的输入*形状*和*数据类型*（dtypes）优化代码。当您调用一个 `jit` 编译的函数时，如果传入的参数 (parameter)具有以前未见过的形状或数据类型，JAX会自动为该新签名重新追踪和重新编译函数。

尽管这很方便，但频繁的重新编译会显著影响性能，由于编译开销，可能会使 `jit` 化的版本比原始Python执行更慢。

```python
import jax
import jax.numpy as jnp
import time

@jax.jit
def process_data(x):
  # 一个对输入形状敏感的简单操作
  return jnp.sum(x * 2.0)

print("形状 (3,) 的编译计时：")
start_time = time.time()
process_data(jnp.ones(3))
print(f"首次调用（形状 (3,)）：{time.time() - start_time:.4f} seconds")

print("\n形状 (3,) 的执行计时：")
start_time = time.time()
process_data(jnp.ones(3))
print(f"第二次调用（形状 (3,)）：{time.time() - start_time:.4f} seconds")

print("\n形状 (4,) 的编译计时：")
start_time = time.time()
process_data(jnp.ones(4)) # 新形状触发重新编译
print(f"第三次调用（形状 (4,)）：{time.time() - start_time:.4f} seconds")

print("\n形状 (4,) 的执行计时：")
start_time = time.time()
process_data(jnp.ones(4))
print(f"第四次调用（形状 (4,)）：{time.time() - start_time:.4f} seconds")

# 输出示例（时间会有所不同）：
# 形状 (3,) 的编译计时：
# First call (shape (3,)): 0.1532 seconds

# 形状 (3,) 的执行计时：
# Second call (shape (3,)): 0.0001 seconds

# 形状 (4,) 的编译计时：
# Third call (shape (4,)): 0.0875 seconds

# 形状 (4,) 的执行计时：
# Fourth call (shape (4,)): 0.0001 seconds
```

第一次和第三次调用慢得多，因为它们包含了新形状的编译时间。第二次和第四次调用快速执行了缓存的编译代码。如果您的应用程序频繁切换许多不同的形状，`jit` 可能无法提供预期的加速。

**解决方法：**

- **填充：** 如果可能，将输入填充到一致的最大形状。
- **`static_argnums` / `static_argnames`：** 如果形状依赖于可以作为编译时常量的输入参数，请将其标记 (token)为静态。
- **批处理：** 以固定大小的批次处理数据。
- **重新评估 `jit` 的使用：** 对于本质上处理高度动态形状的函数，`jit` 可能不是最佳工具，或者您可能只将其应用于处理一致形状的子函数。

### 对全局变量的依赖

编译后的函数会捕获它们在*追踪时*引用的任何全局变量的值。如果您在函数编译后修改了全局变量，编译版本将继续使用旧的、已捕获的值。

```python
import jax
import jax.numpy as jnp

learning_rate = 0.01 # 全局变量

@jax.jit
def update_weights(params, grads):
  # 使用在追踪时捕获的全局学习率
  return params - learning_rate * grads

params = jnp.array([1.0, 2.0])
grads = jnp.array([0.5, -0.1])

print(f"初始学习率：{learning_rate}")
updated_params = update_weights(params, grads)
print(f"更新后的参数（第一次调用）：{updated_params}")

# 在编译后修改全局变量
learning_rate = 1000.0
print(f"\n学习率已更改为：{learning_rate}")

# 再次调用 jit 函数 - 它仍然使用旧的学习率！
updated_params_again = update_weights(params, grads)
print(f"更新后的参数（第二次调用）：{updated_params_again}")

# 输出示例：
# 初始学习率：0.01
# 更新后的参数（第一次调用）：[0.995 2.001]

# 学习率已更改为：1000.0
# 更新后的参数（第二次调用）：[0.995 2.001]  <-- 仍然使用 0.01！
```

**解决方法：** 将像超参数 (parameter) (hyperparameter)这样的可变值显式地作为函数参数传递。这使得函数的行为只依赖于其输入，符合函数式编程原则，并确保编译后的函数使用正确的值。

```python
import jax
import jax.numpy as jnp

# 这里不需要全局变量

# 将 learning_rate 作为参数传递
@jax.jit
def update_weights_explicit(params, grads, lr):
  return params - lr * grads

params = jnp.array([1.0, 2.0])
grads = jnp.array([0.5, -0.1])
current_lr = 0.01

print(f"使用学习率：{current_lr}")
updated_params = update_weights_explicit(params, grads, current_lr)
print(f"更新后的参数（第一次调用）：{updated_params}")

# 更改我们传入的学习率值
current_lr = 1000.0
print(f"\n使用学习率：{current_lr}")
updated_params_again = update_weights_explicit(params, grads, current_lr)
print(f"更新后的参数（第二次调用）：{updated_params_again}")

# 输出示例：
# 使用学习率：0.01
# 更新后的参数（第一次调用）：[0.995 2.001]

# 使用学习率：1000.0
# 更新后的参数（第二次调用）：[-499.  120.] <-- 正确使用了 1000.0
```

通过了解追踪如何与Python的动态特性、副作用和全局状态交互，您可以预见并避免这些常见问题，从而使您能够充分利用 `jax.jit` 的全部功能来加速计算。

## 参考资料

- [JIT compilation and \`jax.jit\`](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html) — The JAX team (2024)
  本官方指南详细介绍了 `jax.jit` 的工作原理，包括其追踪机制、函数式纯度要求以及条件控制流和动态形状等常见问题的解决方案。
