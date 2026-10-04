---
course: "getting-started-with-jax"
chapter: "accelerating-functions-jit"
lesson: "hands-on-applying-jit"
sourceId: 1713
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-2-accelerating-functions-jit/hands-on-applying-jit"
title: "动手实践：应用 `jit`"
description: "练习将 `jax.jit` 应用于简单函数并测量性能提升。"
order: 7
plots: ["plots/1713-0.json"]
sourceHash: "ecd6c450b31a2e9b06780739f689a3891c907406a79d0f7ba8facac6635f2866"
sourceCorrections: []
---

将 `jax.jit` 应用到函数中以观察其对执行速度的影响。还将呈现准确测量性能的方法，同时说明初始编译的开销。

### 设置

首先，请确保您已安装 JAX 并导入所需的库。我们将主要使用 `jax`、`jax.numpy`（通常作为 `jnp` 导入）以及 Python 的 `timeit` 模块进行基础计时。

```python
import jax
import jax.numpy as jnp
import timeit
import numpy as np # 我们将使用标准 NumPy 进行比较设置

# 检查可用设备（可选，但推荐）
print(f"JAX devices: {jax.devices()}")

# 用于更精确计时的辅助函数
def time_func(func, *args, num_runs=10, warmup_runs=2):
    """对函数执行进行计时，同时处理预热和 JAX 异步操作。"""
    # 预热运行
    for _ in range(warmup_runs):
        result = func(*args)
        if isinstance(result, jax.Array):
            result.block_until_ready() # 确保 JAX 计算完成

    # 计时运行
    times = []
    for _ in range(num_runs):
        start_time = timeit.default_timer()
        result = func(*args)
        if isinstance(result, jax.Array):
            result.block_until_ready() # 重要：等待 JAX 操作完成
        end_time = timeit.default_timer()
        times.append(end_time - start_time)

    return np.mean(times), np.std(times)

# 在默认设备上创建一些示例数据
key = jax.random.PRNGKey(0)
size = 1000
x_jnp = jax.random.normal(key, (size, size))
y_jnp = jax.random.normal(key, (size, size))

# 确保数据在计时前位于设备上
x_jnp.block_until_ready()
y_jnp.block_until_ready()
```

请注意 `result.block_until_ready()` 的使用。JAX 操作是异步分发的。这意味着当您调用 JAX 函数时，Python 可能会在 GPU/TPU（甚至 CPU）上的实际计算完成*之前*将控制权返回给您的脚本。为了准确计时，我们需要明确等待结果计算完成。

### 示例 1：加速数值函数

让我们定义一个执行多个 `jax.numpy` 操作的函数：

```python
def compute_heavy(a, b):
  """一个包含多个 JAX 操作的示例函数。"""
  c = jnp.dot(a, b)
  d = jnp.sin(c)
  e = jnp.log(jnp.abs(d) + 1e-6) # 添加 epsilon 以提高数值稳定性
  f = jnp.sum(e)
  return f

# 对原始函数计时
mean_time_orig, std_time_orig = time_func(compute_heavy, x_jnp, y_jnp)
print(f"Original function time: {mean_time_orig:.6f} +/- {std_time_orig:.6f} seconds")

# 现在，让我们对函数进行 JIT 编译
compute_heavy_jit = jax.jit(compute_heavy)

# 对 JIT 编译的函数计时
# 注意：*首次*运行将包含编译时间！
print("Timing JIT version (first run will include compilation)...")
mean_time_jit, std_time_jit = time_func(compute_heavy_jit, x_jnp, y_jnp)
print(f"JIT function time:    {mean_time_jit:.6f} +/- {std_time_jit:.6f} seconds")

# 计算加速比
speedup = mean_time_orig / mean_time_jit
print(f"\nApproximate speedup: {speedup:.2f}x")
```

运行此代码时，在初始编译开销之后，您应该看到原始版本和 JIT 编译版本之间的执行时间有显著差异。JIT 编译器将一系列 `jnp` 操作融合到一个经过优化的单一内核中，该内核在目标加速器上运行速度快很多。

确切的加速比取决于您的硬件（CPU 与 GPU/TPU）、数组的大小以及函数的复杂程度。对于这种纯数值代码，`jit` 带来的好处通常很可观。

### 示例 2：JIT 与 Python 控制流

如前所述，`jit` 通过使用抽象值追踪函数来工作。如果控制流取决于被追踪的特定*值*，这有时可能导致意外行为或错误。

考虑此函数：

```python
def conditional_computation(x, threshold):
  """根据条件执行不同的计算。"""
  if jnp.sum(x) > threshold:
    return jnp.dot(x, x.T) * 2
  else:
    return jnp.dot(x, x.T) / 2

# 尝试对其进行 JIT 编译
conditional_computation_jit = jax.jit(conditional_computation)

# 创建一些小型数据
x_small = jnp.array([1.0, 2.0, 3.0])

# 使用满足条件的值运行
print("Running JIT with condition TRUE:")
result_true = conditional_computation_jit(x_small, 5.0)
result_true.block_until_ready()
print(f"Result (True): {result_true}")

# 使用不满足条件的值运行
# 这很可能会触发重新编译，也可能正常工作，具体取决于 JAX 版本/细节
print("\nRunning JIT with condition FALSE:")
try:
    result_false = conditional_computation_jit(x_small, 10.0)
    result_false.block_until_ready()
    print(f"Result (False): {result_false}")
except Exception as e:
    print(f"Caught an error (as expected sometimes): {e}")

# 使用 jax.lax 进行分阶段控制流的例子
import jax.lax

@jax.jit
def conditional_computation_lax(x, threshold):
  """使用 lax.cond 实现 JIT 兼容的条件逻辑。"""
  return jax.lax.cond(
      jnp.sum(x) > threshold,    # 条件
      lambda op: op * 2,         # True 分支函数
      lambda op: op / 2,         # False 分支函数
      jnp.dot(x, x.T)            # 传递给所选分支的操作数
  )

print("\nRunning JIT with lax.cond:")
result_true_lax = conditional_computation_lax(x_small, 5.0)
result_true_lax.block_until_ready()
print(f"Result lax (True): {result_true_lax}")

result_false_lax = conditional_computation_lax(x_small, 10.0)
result_false_lax.block_until_ready()
print(f"Result lax (False): {result_false_lax}")
```

当 JAX 追踪 `conditional_computation` 时，它会遇到 Python 的 `if` 语句。由于条件 `jnp.sum(x) > threshold` 取决于 `x` 的*值*（在追踪期间是抽象的），JAX 可能难以创建适用于 `if` 所有可能结果的单一编译产物。根据具体情况，它可能：

1. 仅编译首次追踪时采取的分支。随后调用采取*其他*分支时，可能会触发重新编译或导致错误。
2. 尝试展开条件（如果可能），但对于依赖于值的条件，这通常会失败。

处理此问题的标准 JAX 方法是使用 `jax.lax.cond`（用于条件）或 `jax.lax.scan`、`jax.lax.fori_loop`（用于循环）等结构化控制流原语。这些函数被设计为可由 `jit` 追踪。`conditional_computation_lax` 示例展示了 `lax.cond`，它将条件逻辑分阶段编译到 XLA 图中，避免了运行时 Python 级别的 `if` 语句问题。

### 示例 3：静态参数 (parameter)与追踪参数

有时，您函数的一个参数会确定计算的*结构*，而不仅仅是作为数据参与。例如，考虑一个重复执行矩阵乘法的函数：

```python
def apply_n_times(x, n):
  """将矩阵乘法应用 n 次。"""
  y = x
  for _ in range(n): # Python 循环！
    y = jnp.dot(y, x)
  return y

# 尝试在不指定静态参数的情况下进行 JIT 编译
apply_n_times_jit = jax.jit(apply_n_times)

x_matrix = jax.random.normal(key, (50, 50))
x_matrix.block_until_ready()

print("Timing apply_n_times_jit (n=2):")
time_func(apply_n_times_jit, x_matrix, 2) # 首次运行，为 n=2 编译

print("\nTiming apply_n_times_jit (n=3):")
# 这很可能会触发重新编译，因为 'n' 改变了！
time_func(apply_n_times_jit, x_matrix, 3)

# 现在，使用 static_argnums 告诉 JIT，'n' 影响计算结构
apply_n_times_jit_static = jax.jit(apply_n_times, static_argnums=(1,)) # 索引 1 对应于 'n'

print("\nTiming apply_n_times_jit_static (n=2):")
time_func(apply_n_times_jit_static, x_matrix, 2) # 首次运行，编译 n=2 版本

print("\nTiming apply_n_times_jit_static (n=3):")
time_func(apply_n_times_jit_static, x_matrix, 3) # *使用 n=3* 首次运行，为 n=3 编译一个单独的版本

print("\nTiming apply_n_times_jit_static (n=2) AGAIN:")
# 现在应该很快，因为使用了 n=2 的缓存编译
time_func(apply_n_times_jit_static, x_matrix, 2)
```

在第一次尝试（`apply_n_times_jit`）中，Python `for` 循环的迭代次数直接取决于 `n` 的*值*。当 `jit` 追踪此代码时，循环会根据追踪期间遇到的特定 `n` 值（例如 `n=2`）进行展开。当您再次使用 `n=3` 调用函数时，追踪是不同的（循环需要展开 3 次），从而强制进行重新编译。

通过使用 `jax.jit(..., static_argnums=(1,))`，我们告诉 JAX，索引 1 处的参数 (`n`) 是*静态的*。这意味着 JAX 不会尝试追踪其值。相反，它在*给定编译*中会将 `n` 视为一个常量。如果您使用*不同*的静态值（例如将 `n` 从 2 更改为 3）调用该函数，JAX 将识别此情况并为该特定 `n` 值编译一个*新的、专门版本*的函数。随后使用相同静态值（再次 `n=2`）的调用将重用缓存的、已编译的版本。这避免了运行时重新编译的开销，同时仍然允许编译后的代码针对由静态参数控制的特定结构变体进行优化。

### 可视化加速效果

让我们可视化示例 1 中的计时差异。

```python
# 用于绘图的数据（使用示例 1 的结果）
labels = ['Original Function', 'JIT Compiled']
mean_times = [mean_time_orig, mean_time_jit]
std_devs = [std_time_orig, std_time_jit]
```



![函数执行时间比较](plots/1713-0.json)



> 原始 Python 函数及其 JIT 编译版本的平均执行时间（对数刻度）比较。误差棒表示多次运行的标准差。越低越好。请注意 JIT 编译后时间的显著减少。

本次动手实践展示了 `jax.jit` 的实际应用。您已看到它如何显著加速数值代码，Python 控制流如何与追踪交互，以及如何使用 `static_argnums` 来管理那些结构依赖于特定参数 (parameter)的函数的编译。请记住，在对 JAX 代码计时时始终使用 `.block_until_ready()`，并注意初始编译开销。随着您构建更复杂的 JAX 程序，`jit` 将是您性能优化工具箱中不可或缺的工具。

## 参考资料

- [JIT Compilation](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-compilation) — JAX documentation (2023)
  此官方文档说明了 `jax.jit` 的作用和性能优势，涵盖了异步执行和初始编译成本。
- [Common Gotchas in JAX: Control Flow](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow-with-lax-cond) — JAX documentation team (2023)
  针对在使用 `jax.jit` 时与 Python 控制流相关的常见问题，提供了关于使用 JAX 结构化控制流原语（如 `jax.lax.cond`）的指导。
- [XLA: Optimizing Compilers for Machine Learning](https://www.tensorflow.org/xla) — Google (2023)
  Publisher: Google
  介绍了 XLA，这是 `jax.jit` 使用的领域特定编译器，用于编译机器学习模型，以在各种加速器上实现高性能执行。
